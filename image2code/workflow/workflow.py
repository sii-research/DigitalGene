"""
Distributed Code Generation

This module provides distributed inference capabilities for the image2code model
to generate code based on given test datasets with data parallelism support.
"""

import json
import os
from argparse import ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm

from utils.inference import inference
from utils.processor import ParameterProcessor
from utils.qwen_hf import QwenVLGenerationModel

class DistributedQwenVLGenerator:
    """
    Distributed code generator using QwenVL model.
    
    This class handles distributed inference across multiple GPUs for efficient
    code generation from images using the QwenVL model.
    """
    
    def __init__(self, args: Namespace):
        """
        Initialize the distributed generator.
        
        Args:
            args: Configuration arguments containing model path, dataset info, etc.
        """
        self.args = args
        self.world_size = args.num_gpus
        
    def setup_distributed(self, rank: int) -> None:
        """
        Setup distributed training environment.
        
        Args:
            rank: Process rank in distributed setup
        """
        os.environ['MASTER_ADDR'] = self.args.master_addr
        os.environ['MASTER_PORT'] = self.args.master_port
        dist.init_process_group("nccl", rank=rank, world_size=self.world_size)
        torch.cuda.set_device(rank)
        
    def cleanup_distributed(self) -> None:
        """Clean up distributed training environment."""
        if dist.is_initialized():
            dist.destroy_process_group()
    
    @staticmethod
    def split_workload(data_list: List, num_parts: int) -> List[List]:
        """
        Split data list into approximately equal parts for distributed processing.
        
        Args:
            data_list: List of data items to split
            num_parts: Number of parts to split into
            
        Returns:
            List of sublists, one for each process
        """
        if not data_list:
            return [[] for _ in range(num_parts)]
        chunk_size = len(data_list) // num_parts
        remainder = len(data_list) % num_parts

        result = []
        start_idx = 0
        for i in range(num_parts):
            end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
            result.append(data_list[start_idx:end_idx])
            start_idx = end_idx
        return result
    
    def align_bounding_box(self, image: Image.Image, target_size: int = 720, box_ratio: float = 0.65) -> Image.Image:
        """
        Align image with bounding box preprocessing.
        
        Args:
            image: Input PIL Image
            target_size: Target image size after processing
            box_ratio: Ratio for box size calculation
            
        Returns:
            Processed PIL Image
        """
        box_size = int(target_size * box_ratio)
        original_width, original_height = image.size
        # Pad to square
        max_dimension = max(original_width, original_height)
        padded_image = Image.new("RGB", (max_dimension, max_dimension), (255, 255, 255))
        paste_x = (max_dimension - original_width) // 2
        paste_y = (max_dimension - original_height) // 2
        padded_image.paste(image, (paste_x, paste_y))
        # Resize to box size
        resized_image = padded_image.resize((box_size, box_size), Image.LANCZOS)
        # Pad to target size
        final_image = Image.new("RGB", (target_size, target_size), (255, 255, 255))
        final_paste_x = (target_size - box_size) // 2
        final_paste_y = (target_size - box_size) // 2
        final_image.paste(resized_image, (final_paste_x, final_paste_y))
        return final_image
    
    def get_output_directory(self) -> Path:
        """
        Get the output directory path based on model name.
        
        Returns:
            Path object for output directory
        """
        model_name = os.path.basename(self.args.model_path)
        return Path(self.args.save_dir) / model_name
    
    def load_test_categories(self) -> List[str]:
        """
        Load test dataset categories.
        
        Returns:
            List of category names
        """
        test_set_path = Path(self.args.test_set_dir)
        if not test_set_path.exists():
            raise FileNotFoundError(f"Test set directory not found: {test_set_path}")
        categories = [
            item for item in os.listdir(test_set_path)
            if (test_set_path / item).is_dir()
        ]
        if not categories:
            raise ValueError("No categories found in test set directory")
        return categories
    
    def process_single_image(self, model: QwenVLGenerationModel, image_path: Path, category: str, output_dir: Path, global_index: int) -> None:
        """
        Process a single image and generate code.
        
        Args:
            model: QwenVL model instance
            image_path: Path to input image
            category: Image category
            output_dir: Output directory for results
            global_index: Global index for this image
        """
        # Create output directory for this item
        item_output_dir = output_dir / str(global_index)
        item_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Load and process image
            image = Image.open(image_path).convert("RGB")
            # Save processed image
            output_image_path = item_output_dir / image_path.name
            image.save(output_image_path)
            # Generate code using model
            inference_results = inference(model, output_image_path, category=category)
            # Save generated code
            code_str, raw_outputs = inference_results
            code_output_path = item_output_dir / "generated_code_before_rec.json"
            with open(code_output_path, "w", encoding="utf-8") as f:
                f.write(code_str)
            # Save raw outputs
            raw_output_path = item_output_dir / "generated_raw_outputs.txt"
            with open(raw_output_path, "w", encoding="utf-8") as f:
                f.write(raw_outputs)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # Create error marker file
            error_path = item_output_dir / "error.txt"
            with open(error_path, "w", encoding="utf-8") as f:
                f.write(f"Error: {str(e)}")
    
    def process_images_worker(self, rank: int) -> None:
        """
        Worker function for processing images on a specific GPU.
        
        Args:
            rank: GPU rank/ID for this worker
        """
        device = torch.device(f"cuda:{rank}")
        # Load test categories
        try:
            categories = self.load_test_categories()
        except Exception as e:
            print(f"Error loading test categories: {e}")
            return
        if rank == 0:
            print(f"Found categories: {categories}")
            print("=" * 50)
            print(f"Loading model: {self.args.model_path}")
        # Initialize model
        try:
            model = QwenVLGenerationModel(Path(self.args.model_path), device=device)
        except Exception as e:
            print(f"Error loading model on GPU {rank}: {e}")
            return
        # Process each category
        for category in categories:
            if rank == 0:
                print(f"Processing category: {category}")
            category_path = Path(self.args.test_set_dir) / category
            # Get all image paths in this category
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
            all_image_paths = [
                category_path / filename
                for filename in os.listdir(category_path)
                if Path(filename).suffix.lower() in image_extensions
            ]
            if not all_image_paths:
                if rank == 0:
                    print(f"No images found in category: {category}")
                continue
            # Split workload among processes
            image_chunks = self.split_workload(all_image_paths, self.world_size)
            worker_images = image_chunks[rank]
            if rank == 0:
                print(f"Total images: {len(all_image_paths)}")
                print(f"Images per GPU: ~{len(worker_images)}")
            # Setup output directory
            output_dir = self.get_output_directory() / category
            output_dir.mkdir(parents=True, exist_ok=True)
            # Process assigned images
            for local_idx, image_path in enumerate(
                tqdm(worker_images, desc=f"GPU {rank}", disable=rank != 0)
            ):
                # Calculate global index
                global_idx = sum(len(chunk) for chunk in image_chunks[:rank]) + local_idx
                self.process_single_image(
                    model=model,
                    image_path=image_path,
                    category=category,
                    output_dir=output_dir,
                    global_index=global_idx
                )
    
    def post_process_results(self) -> None:
        """Post-process generated results to recover parameters."""
        print("Starting post-processing...")
        # Initialize parameter processor
        processor = ParameterProcessor(num_bins=1024, token_start_id=2048)
        # Load statistical information
        stats_path = Path(self.args.stats_dir) / "stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"Stats file not found: {stats_path}")
        with open(stats_path, "r", encoding="utf-8") as f:
            processor.stats = json.load(f)
        print(f"Loaded statistics from: {stats_path}")
        # Process each category
        output_base_dir = self.get_output_directory()
        if not output_base_dir.exists():
            print("No output directory found. Skipping post-processing.")
            return
        categories = [
            item for item in os.listdir(output_base_dir)
            if (output_base_dir / item).is_dir()
        ]
        for category in categories:
            print(f"Post-processing category: {category}")
            category_dir = output_base_dir / category
            item_dirs = [
                category_dir / item
                for item in os.listdir(category_dir)
                if (category_dir / item).is_dir()
            ]
            
            for item_dir in tqdm(item_dirs, desc="Post-processing"):
                input_path = item_dir / "generated_code_before_rec.json"
                output_path = item_dir / "generated_code.json"
                if not input_path.exists():
                    continue
                try:
                    with open(input_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    recovered_data = processor.recover_item(data)
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(recovered_data, f, indent=4, ensure_ascii=False)
                except Exception as e:
                    print(f"Error post-processing {item_dir}: {e}")
    
    def run_distributed_worker(self, rank: int) -> None:
        """
        Main worker function for distributed processing.
        
        Args:
            rank: Process rank
        """
        self.setup_distributed(rank)
        try:
            self.process_images_worker(rank)
        finally:
            self.cleanup_distributed()
    
    def run(self) -> None:
        """Main entry point for running the distributed generation."""
        if self.world_size > 1:
            print(f"Starting distributed inference with {self.world_size} GPUs")
            mp.spawn(
                self.run_distributed_worker,
                args=(),
                nprocs=self.world_size,
                join=True
            )
        else:
            print("Running single GPU inference")
            self.process_images_worker(0)
        # Post-process results
        self.post_process_results()
        print("Generation and post-processing completed!")

def create_argument_parser() -> ArgumentParser:
    """
    Create and configure argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = ArgumentParser(
        description="Generate code from images using QwenVL model with data parallelism",
        formatter_class=ArgumentParser.ArgumentDefaultsHelpFormatter
    )
    
    # Model configuration
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Path to the QwenVL model")
    # Data configuration
    parser.add_argument("--test_set_dir", type=str, required=True, help="Directory containing test dataset")
    parser.add_argument("--stats_dir", type=str, required=True, help="Directory containing statistical information for post-processing")
    # Output configuration
    parser.add_argument("--save_dir", type=str, default="outputs-real/", help="Directory to save generated results")
    # Distributed configuration
    parser.add_argument("--num_gpus", type=int, default=torch.cuda.device_count(), help="Number of GPUs to use for parallel processing")
    parser.add_argument("--master_addr", type=str, default="localhost", help="Master node address for distributed training")
    parser.add_argument("--master_port", type=str, default="12355", help="Master node port for distributed training")
    return parser

def main() -> None:
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    # Validate arguments
    if args.num_gpus <= 0:
        raise ValueError("Number of GPUs must be positive")
    if not Path(args.test_set_dir).exists():
        raise FileNotFoundError(f"Test set directory not found: {args.test_set_dir}")
    if not Path(args.stats_dir).exists():
        raise FileNotFoundError(f"Stats directory not found: {args.stats_dir}")
    # Create output directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    # Run distributed generation
    generator = DistributedQwenVLGenerator(args)
    generator.run()

if __name__ == "__main__":
    main()
