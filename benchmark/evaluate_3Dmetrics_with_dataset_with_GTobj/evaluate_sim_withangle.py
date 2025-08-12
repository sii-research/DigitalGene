import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch # For device checking

# Assuming the evaluation script you provided is saved as evaluation_utils.py
# and is in the same directory or accessible via PYTHONPATH
from evaluate_sim_function import evaluate_obj_files, normalize_pc, chamfer_distance_wrapper, compute_fscore, get_rotation_sphere, find_best_coarse_alignment, icp_refine


def find_identifier_and_generated_obj(instance_group_path: Path):
    """
    In a folder like 'checkpoint-60000/Bottle/0/', finds 'generated_obj.obj'
    and extracts an identifier (e.g., 'bottle_041_08') from a .png file.
    Returns (path_to_generated_obj, identifier_string) or (None, None).
    """
    generated_obj_path = instance_group_path / "generated_obj.obj"
    if not generated_obj_path.exists():
        # print(f"Warning: generated_obj.obj not found in {instance_group_path}")
        return None, None

    # Find a .png file to use as the identifier
    png_files = list(instance_group_path.glob("*.png"))
    if not png_files:
        # print(f"Warning: No .png file found in {instance_group_path} to determine identifier.")
        return None, None
    
    # Take the first png file found
    identifier_stem = png_files[0].stem # e.g., "bottle_041_08"
    return generated_obj_path, identifier_stem


def run_evaluation_pipeline(generated_data_dir_str: str, gt_data_dir_str: str,
                            num_points: int, f_threshold_str: str,
                            device_str: str, use_icp_flag: bool, icp_iter: int,
                            output_filename: str = "evaluation_metrics.json"):
    """
    Runs the evaluation pipeline.
    """
    generated_data_root = Path(generated_data_dir_str)
    gt_data_root = Path(gt_data_dir_str)

    if not generated_data_root.is_dir():
        print(f"Error: Generated data directory not found: {generated_data_root}")
        return
    if not gt_data_root.is_dir():
        print(f"Error: Ground truth data directory not found: {gt_data_root}")
        return

    f_threshold_list = [float(t) for t in f_threshold_str.split(',')]
    all_category_metrics = {}
    total_evaluated_pairs = 0
    total_start_time = time.time()

    print(f"Starting evaluation pipeline...")
    print(f"Generated data root: {generated_data_root}")
    print(f"Ground truth data root: {gt_data_root}")
    print(f"Device: {device_str}, Num Points: {num_points}, ICP: {use_icp_flag}, ICP Iters: {icp_iter}")
    print(f"F-score thresholds: {f_threshold_list}")
    print("-" * 50)

    # Iterate through categories in the generated_data_dir (e.g., Bottle, Mug)
    for category_path in generated_data_root.iterdir():
        if not category_path.is_dir():
            continue
        
        category_name_generated = category_path.name # e.g., "Bottle"
        # Normalize category name for matching with GT folder (usually lowercase)
        category_name_gt_match = category_name_generated #.lower()
        print(f"\nProcessing Category: {category_name_generated}")

        if category_name_generated not in all_category_metrics:
            all_category_metrics[category_name_generated] = {
                "cd": [], "cd_accuracy": [], "cd_completeness": [],
                    "rot_angle_deg": [],        # NEW
                "yaw_deg": [], "pitch_deg": [], "roll_deg": [],             # NEW
                "f_scores": [[] for _ in f_threshold_list] # list of lists for f-scores
            }
        
        gt_category_path = gt_data_root / category_name_gt_match
        if not gt_category_path.is_dir():
            print(f"  Warning: GT category directory not found for '{category_name_generated}' (expected at {gt_category_path}). Skipping.")
            continue

        # Iterate through instance groups (e.g., "0", "1")
        for instance_group_path in category_path.iterdir():
            output_json_path = instance_group_path / output_filename
            pair_duration= -1
            if not instance_group_path.is_dir():
                continue
            if output_json_path.exists():
                print(f'file exist {output_json_path} , will pass',flush=True)
                with open(output_json_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)  # 将JSON文件内容加载到Python字典
            else:
                generated_obj_path, identifier = find_identifier_and_generated_obj(instance_group_path)

                if not generated_obj_path or not identifier:
                    print(f"  Skipping {instance_group_path}: Missing generated_obj.obj or identifier PNG.")
                    continue

                # Construct path to the corresponding GT Scan.obj
                # Identifier is like 'bottle_041_08'
                # GT path is like 'collected_pair_data/bottle/bottle_041_08/Scan.obj'
                gt_obj_path = gt_category_path / Path(str(identifier)  +"/Scan.obj")
                # if not 'box_039' in str(gt_obj_path):
                #     continue

                if not gt_obj_path.exists():
                    print(f"  Warning: GT Scan.obj not found for identifier '{identifier}' (expected at {gt_obj_path}). Skipping.")
                    continue
                
                print(f"  Evaluating: {generated_obj_path.name} ({identifier}) vs {gt_obj_path.relative_to(gt_data_root)}")
                
                pair_start_time = time.time()
                results = evaluate_obj_files(
                    str(generated_obj_path), str(gt_obj_path),
                    num_points, f_threshold_list, device_str,
                    use_icp_flag, icp_iter
                )
                pair_duration = time.time() - pair_start_time

            if results:
                print(results)
                total_evaluated_pairs += 1
                # Save results to JSON in the generated object's directory
                try:
                    with open(output_json_path, 'w') as f:
                        json.dump(results, f, indent=4)
                    print(f"    Saved results to {output_json_path} (Took {pair_duration:.2f}s)")
                except IOError as e:
                    print(f"    Error saving results to {output_json_path}: {e}")

                # Store for category aggregation
                all_category_metrics[category_name_generated]["cd"].append(results["cd"])
                all_category_metrics[category_name_generated]["cd_accuracy"].append(results["cd_accuracy"])
                all_category_metrics[category_name_generated]["cd_completeness"].append(results["cd_completeness"])
                for i, f_score_val in enumerate(results["f_scores"]):
                    all_category_metrics[category_name_generated]["f_scores"][i].append(f_score_val)
                #  add just below the existing appends
                all_category_metrics[category_name_generated]["rot_angle_deg"] \
                    .append(results["rot_angle_deg"])

                yaw_deg, pitch_deg, roll_deg = results["yaw_pitch_roll_deg"]
                all_category_metrics[category_name_generated]["yaw_deg"].append(yaw_deg)
                all_category_metrics[category_name_generated]["pitch_deg"].append(pitch_deg)
                all_category_metrics[category_name_generated]["roll_deg"].append(roll_deg)

            else:
                print(f"    Evaluation failed for pair: {generated_obj_path.name} and {gt_obj_path.name} ")
            # break
                

    print("-" * 50)
    print("\n--- Category Average Metrics ---")
    if not all_category_metrics or total_evaluated_pairs == 0:
        print("No pairs were successfully evaluated.")
    else:
        for category, metrics_data in all_category_metrics.items():
            num_samples = len(metrics_data["cd"])
            if num_samples == 0:
                print(f"\nCategory: {category} (0 samples evaluated)")
                continue

            print(f"\nCategory: {category} ({num_samples} samples)",flush=True)
            avg_cd = np.mean(metrics_data["cd"]) if metrics_data["cd"] else float('nan')
            avg_cd_acc = np.mean(metrics_data["cd_accuracy"]) if metrics_data["cd_accuracy"] else float('nan')
            avg_cd_comp = np.mean(metrics_data["cd_completeness"]) if metrics_data["cd_completeness"] else float('nan')
            
            avg_rot = np.mean(metrics_data["rot_angle_deg"])
            avg_yaw = np.mean(np.abs(metrics_data["yaw_deg"]))       # abs so sign doesn’t cancel
            avg_pitch = np.mean(np.abs(metrics_data["pitch_deg"]))
            avg_roll  = np.mean(np.abs(metrics_data["roll_deg"]))
            print(f"  Avg Rotation Angle       : {avg_rot:.2f}°",flush=True)
            print(f"  Avg |Yaw|, |Pitch|, |Roll|: {avg_yaw:.2f}°, "
                f"{avg_pitch:.2f}°, {avg_roll:.2f}°",flush=True)
            print(f"  Average Chamfer Distance (CD): {avg_cd:.6f}",flush=True)
            print(f"    Average CD Accuracy: {avg_cd_acc:.6f}",flush=True)
            print(f"    Average CD Completeness: {avg_cd_comp:.6f}",flush=True)
            
            print("  Average F-scores @ d:",flush=True)
            for i, threshold in enumerate(f_threshold_list):
                avg_f_score = np.mean(metrics_data["f_scores"][i]) if metrics_data["f_scores"][i] else float('nan')
                print(f"    F-score @ {threshold:.4f}: {avg_f_score:.6f}",flush=True)
    
    total_duration = time.time() - total_start_time
    print("-" * 50)
    print(f"Pipeline finished. Evaluated {total_evaluated_pairs} pairs.",flush=True)
    print(f"Total execution time: {total_duration:.2f} seconds.",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation pipeline for generated OBJ files against GT OBJ files.")
    parser.add_argument("--generated_data_dir", type=str,default='3d_benchmark/32B',
                        help="Root directory of the generated data (e.g., './checkpoint-60000').")
    ### example of directory  3d_benchmark/32B
    ### ```tree 3d_benchmark/32B```
    # |-- Bottle
    # |   |-- 0
    # |   |   |-- bottle_059_20.png
    # |   |   |-- evaluation_metrics.json
    # |   |   |-- generated.png
    # |   |   |-- generated_code.json
    # |   |   |-- generated_code_before_rec.json
    # |   |   |-- generated_obj.obj
    # |   |   `-- generated_raw_outputs.txt
    # |   |-- 1
    # |   |   |-- bottle_047_09.png
    # |   |   |-- evaluation_metrics.json
    # |   |   |-- generated.png
    # |   |   |-- generated_code.json
    # |   |   |-- generated_code_before_rec.json
    # |   |   |-- generated_obj.obj
    # |   |   `-- generated_raw_outputs.txt
    #  ....
    # |-- Box
    # |   |-- 0
    # |   |   |-- box_040_19.png
    # |   |   |-- evaluation_metrics.json
    # |   |   |-- generated.png
    # |   |   |-- generated_code.json
    # |   |   |-- generated_code_before_rec.json
    # |   |   |-- generated_obj.obj
    # |   |   `-- generated_raw_outputs.txt
    # |   |-- 1
    # |   |   |-- box_025_05.png
    # |   |   |-- evaluation_metrics.json
    # |   |   |-- generated.png
    # |   |   |-- generated_code.json
    # |   |   |-- generated_code_before_rec.json
    # |   |   |-- generated_obj.obj
    # |   |   `-- generated_raw_outputs.txt
    # ...
    parser.add_argument("--gt_data_dir", type=str,default='/inspire/hdd/project/robot-dna/public/dataset/image2code_proj/OmniObject3D/omniobject3d___OmniObject3D-New/raw/collected_pair_data_rotation',
                        help="Root directory of the ground truth data (e.g., './collected_pair_data').")
    # ```tree /inspire/hdd/project/robot-dna/public/dataset/image2code_proj/OmniObject3D/omniobject3d___OmniObject3D-New/raw/collected_pair_data_rotation```
    # |-- Bottle
    # |   |-- bottle_001_01
    # |   |   |-- Scan.obj
    # |   |   |-- frame.png
    # |   |   |-- material.mtl
    # |   |   `-- material_0.006.jpeg
    # |   |-- bottle_001_02
    # |   |   |-- Scan.obj
    # |   |   |-- frame.png
    # |   |   |-- material.mtl
    # |   |   `-- material_0.006.jpeg
    # ...
    # |-- Box
    # |   |-- box_001_01
    # |   |   |-- Scan.obj
    # |   |   |-- frame.png
    # |   |   |-- material.mtl
    # |   |   `-- material_0.jpeg
    # |   |-- box_001_02
    # |   |   |-- Scan.obj
    # |   |   |-- frame.png
    # |   |   |-- material.mtl
    # |   |   `-- material_0.jpeg
    # ...
    # Arguments mirroring those in evaluation_utils.py
    parser.add_argument("--num_points", type=int, default=10000,
                        help="Number of points to sample from each mesh.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use ('cuda' or 'cpu').")
    parser.add_argument("--f_thresholds", type=str, default="0.005,0.01,0.02,0.05",
                        help="Comma-separated F-score thresholds.")
    parser.add_argument("--use_icp", action='store_true',
                        help="Enable ICP refinement after brute-force alignment.")
    parser.set_defaults(use_icp=False) # Default to False if not specified
    parser.add_argument("--icp_iterations", type=int, default=1000,
                        help="Number of iterations for ICP.")
    parser.add_argument("--output_filename", type=str, default="evaluation_metrics.json",
                        help="Filename for the JSON output in each evaluated instance's directory.")
    
    args = parser.parse_args()

    run_evaluation_pipeline(
        args.generated_data_dir,
        args.gt_data_dir,
        args.num_points,
        args.f_thresholds,
        args.device,
        args.use_icp,
        args.icp_iterations,
        args.output_filename
    )