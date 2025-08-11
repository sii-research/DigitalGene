"""
QwenVL Generation Model Implementation.

This module provides a wrapper for the Qwen2.5-VL model for vision-language
generation tasks with support for both single and batch inference.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

class QwenVLGenerationModel:
    """
    Wrapper class for Qwen2.5-VL model with generation capabilities.
    
    This class provides convenient methods for single and batch inference
    using the Qwen2.5-VL vision-language model.
    """
    
    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct", max_new_tokens: int = 8192, device: str = ""):
        """
        Initialize the QwenVL generation model.
        
        Args:
            model_path: Path to the model (local or HuggingFace model ID)
            max_new_tokens: Maximum number of tokens to generate
            device: Device to run the model on ("auto" if empty)
        """
        self._model_path = model_path
        self.device = device if device else "auto"
        self._max_new_tokens = max_new_tokens
        
        # Load model and processor
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype="auto", 
            device_map=self.device
        )
        self._processor = AutoProcessor.from_pretrained(model_path)
        
    @staticmethod
    def _prompt_template(question: str, image_path: str, system_prompt: str = "") -> List[Dict]:
        """
        Create message template for the model.
        
        Args:
            question: User question/instruction
            image_path: Path to the image file
            system_prompt: Optional system prompt
            
        Returns:
            Formatted message list for the model
        """
        user_content = [
            {
                "type": "image",
                "image": f"file://{image_path}",
            },
            {"type": "text", "text": question}
        ]
        # system prompt
        if system_prompt:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}]
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        return messages

    def generate(self, question: str, image_path: str, system_prompt: str = "") -> List[str]:
        """
        Generate response for a single image-question pair.
        
        Args:
            question: Question or instruction about the image
            image_path: Path to the input image
            system_prompt: Optional system prompt for context
            
        Returns:
            List containing the generated text response
        """
        # Create message template
        messages = self._prompt_template(question, image_path, system_prompt)
        # Prepare inputs for inference
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        # Generate response
        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens)
        # Extract only the newly generated tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        # Decode generated tokens to text
        output_text = self._processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        return output_text
    
    def generate_batch(self, items: List[Tuple[str, str, str]]) -> List[str]:
        """
        Generate responses for multiple image-question pairs in batch.
        
        Args:
            items: List of tuples containing (question, image_path, system_prompt)
            
        Returns:
            List of generated text responses
        """
        # Create message templates for all items
        all_messages = []
        for question, image_path, system_prompt in items:
            messages = self._prompt_template(question, image_path, system_prompt)
            all_messages.append(messages)
        # Prepare batch inputs
        texts = [
            self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in all_messages
        ]
        image_inputs, video_inputs = process_vision_info(all_messages)
        inputs = self._processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        # Batch generation
        with torch.no_grad():
            generated_ids = self._model.generate(**inputs, max_new_tokens=self._max_new_tokens)
        # Extract newly generated tokens for each item
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        # Decode all generated texts
        output_texts = self._processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        return output_texts
