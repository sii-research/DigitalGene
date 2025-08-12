import numpy as np
import torch
import trimesh # For loading OBJ files and sampling points
import argparse
from external.chamfer3D.dist_chamfer_3D import chamfer_3DDist
import math
from pathlib import Path

# -------- Kabsch（SVD）求刚体变换 --------
def kabsch_alignment(src: torch.Tensor, tgt: torch.Tensor):
    """
    src, tgt: [N,3] (same N) after PCA 粗对齐 & 最近邻对应
    Returns: R (3×3), t (3,)
    """
    assert src.shape == tgt.shape
    src_mean = src.mean(dim=0, keepdim=True)      # [1,3]
    tgt_mean = tgt.mean(dim=0, keepdim=True)

    src_c = src - src_mean
    tgt_c = tgt - tgt_mean

    H = src_c.T @ tgt_c                           # 3×3
    U, _, V = torch.linalg.svd(H)
    R = V @ U.T
    if torch.det(R) < 0:                          # 保证右手系
        V[:, 2] *= -1
        R = V @ U.T
    t = tgt_mean.squeeze(0) - R @ src_mean.squeeze(0)
    return R, t                                   # torch tensors
# -------- PCA 建立初始旋转 --------
def pca_rotation(pc: torch.Tensor):
    """
    pc: [N,3]  —— 已经去过中心也没关系
    返回: 3×3 的旋转矩阵，其列为 3 个主轴（最大到最小特征向量）
    """
    # 去中心
    c = pc.mean(dim=0, keepdim=True)
    Xc = pc - c
    # 协方差(3×3)
    cov = Xc.T @ Xc / Xc.shape[0]
    # 对称矩阵 => eigen decomposition
    eigvals, eigvecs = torch.linalg.eigh(cov)     # eigvecs column-wise
    # 按特征值从大到小排序
    idx = torch.argsort(eigvals, descending=True)
    R = eigvecs[:, idx]                           # 3×3
    # 如果 R 不是右手系, 调整一下
    if torch.det(R) < 0:
        R[:, 2] *= -1
    return R


def chamfer_distance_wrapper( X1, X2):
    assert(X1.shape[2]==3)
    Chamfer_3D = chamfer_3DDist().to(X1.device)
    dist_1, dist_2, idx_1, idx_2 = Chamfer_3D(X1, X2)
    return dist_1.sqrt(), dist_2.sqrt(), idx_1, idx_2
def rotation_matrix_to_angle_deg(R: torch.Tensor) -> float:
    """
    Geodesic rotation angle (degrees) from a 3×3 rotation matrix.
    """
    # clamp to avoid numerical drift outside [-1,1]
    trace_val = torch.clamp((torch.trace(R) - 1.) / 2., -1., 1.)
    angle_rad = torch.acos(trace_val)
    return float(angle_rad * 180.0 / math.pi)

def rotation_matrix_to_euler_zyx_deg(R: torch.Tensor) -> tuple[float, float, float]:
    """
    Convert 3×3 rotation matrix to Z-Y-X (yaw-pitch-roll) Euler angles in degrees.
    Assumes a right-handed, Z-up convention (common for OmniObject3D).
    """
    # numerical guard for gimbal-lock
    sy = torch.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        yaw   = torch.atan2(R[1, 0], R[0, 0])
        pitch = torch.atan2(-R[2, 0], sy)
        roll  = torch.atan2(R[2, 1], R[2, 2])
    else:  # pitch ≈ ±90°
        yaw   = torch.atan2(-R[0, 1], R[1, 1])
        pitch = torch.atan2(-R[2, 0], sy)
        roll  = torch.tensor(0., device=R.device)
    return (float(yaw * 180.0 / math.pi),
            float(pitch * 180.0 / math.pi),
            float(roll * 180.0 / math.pi))

@torch.no_grad()
def normalize_pc(pc):
    assert pc.ndim == 3
    pc_centered = pc - pc.mean(dim=1, keepdim=True)                 # 平移到质心
    # 计算 X/Y/Z 轴长
    lengths = pc_centered.max(dim=1, keepdim=True)[0] - pc_centered.min(dim=1, keepdim=True)[0]  # [B,1,3]
    max_len  = lengths.max(dim=-1, keepdim=True)[0]                 # [B,1,1]
    pc_norm  = pc_centered / (max_len + 1e-7)
    return pc_norm
    # pc_mean = pc.mean(dim=1, keepdim=True)
    # pc_zmean = pc - pc_mean
    # length_x = pc_zmean[:, :, 0].max(dim=-1, keepdim=True)[0] - pc_zmean[:, :, 0].min(dim=-1, keepdim=True)[0]
    # length_y = pc_zmean[:, :, 1].max(dim=-1, keepdim=True)[0] - pc_zmean[:, :, 1].min(dim=-1, keepdim=True)[0]
    # length_max = torch.stack([length_x, length_y], dim=-1).max(dim=-1)[0].unsqueeze(-1)
    # pc_normalized = pc_zmean / (length_max + 1.e-7)
    # return pc_normalized
import trimesh     # 若想用 trimesh 也行，见注释

# -------- 1. 工具函数 --------
def save_point_cloud_obj(pc: torch.Tensor, filename: str):
    """
    将点云写成 OBJ（只有顶点、没有面），兼容 Meshlab / CloudCompare 查看。
    pc : [N,3]  (torch 或 numpy)
    """
    if isinstance(pc, torch.Tensor):
        pc = pc.detach().cpu().numpy()  # -> (N,3)

    with open(filename, 'w') as f:
        for x, y, z in pc:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
    print(f"Saved point cloud to {filename}")
def get_rotation_sphere(azim_sample=8, elev_sample=8, roll_sample=8, scales=None):
    if scales is None:
        scales = [1.0]
    rotations = []
    for scale in scales:
        for azim_angle in np.linspace(0, 2 * np.pi, azim_sample, endpoint=False):
            for elev_angle in np.linspace(-np.pi / 2, np.pi / 2, elev_sample, endpoint=True):
                num_roll_samples = 1 if np.isclose(np.abs(elev_angle), np.pi/2) else roll_sample
                for roll_angle in np.linspace(0, 2 * np.pi, num_roll_samples, endpoint=False):
                    ca, sa = np.cos(azim_angle), np.sin(azim_angle)
                    ce, se = np.cos(elev_angle), np.sin(elev_angle)
                    cr, sr = np.cos(roll_angle), np.sin(roll_angle)
                    R_x = torch.tensor([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=torch.float32)
                    R_y = torch.tensor([[ce, 0, se], [0, 1, 0], [-se, 0, ce]], dtype=torch.float32)
                    R_z = torch.tensor([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=torch.float32)
                    R = R_z @ R_y @ R_x
                    rotations.append(R)
    return torch.stack(rotations)

def compute_fscore(dist1, dist2, thresholds):
    fscores = []
    for threshold_d in thresholds:
        precision = torch.mean((dist1 < threshold_d).float(), dim=1)
        recall = torch.mean((dist2 < threshold_d).float(), dim=1)
        fscore = 2 * precision * recall / (precision + recall)
        fscore[torch.isnan(fscore) | torch.isinf(fscore)] = 0
        fscores.append(fscore)
    fscores_tensor = torch.stack(fscores, dim=1)
    return fscores_tensor

@torch.no_grad()
def icp_refine(pc_pred_initial, pc_gt_target, num_iter=50, device="cuda"):
    """
    Refines the alignment of pc_pred_initial to pc_gt_target using ICP.
    Assumes inputs are single point clouds [N, 3] or [M, 3], already normalized.
    """
    X1 = pc_pred_initial.unsqueeze(0).to(device) # [1, N, 3]
    X2 = pc_gt_target.unsqueeze(0).to(device)    # [1, M, 3]

    for _ in range(num_iter):
        # chamfer_distance_wrapper returns sqrt(dist_sq), idx1, idx2
        # We only need idx from X1 to X2 for correspondences
        _, _, idx1_to_2, _ = chamfer_distance_wrapper(X1, X2) # idx1_to_2: [1, N]
        # print(idx1_to_2,'idx1_to_2')
        idx1_to_2 = idx1_to_2.long()
        # Create X2_corresp based on found correspondences
        # X2_corresp will have shape [1, N, 3]
        X2_corresp = torch.gather(X2, 1, idx1_to_2.unsqueeze(-1).expand(-1, -1, 3))

        t1 = X1.mean(dim=1, keepdim=True) # [1, 1, 3]
        t2 = X2_corresp.mean(dim=1, keepdim=True) # [1, 1, 3]

        X1_centered = X1 - t1
        X2_corresp_centered = X2_corresp - t2

        # Covariance matrix H = X1_centered^T @ X2_corresp_centered
        # H shape should be [1, 3, 3]
        H = X1_centered.transpose(1, 2) @ X2_corresp_centered # [1, 3, N] @ [1, N, 3] -> [1, 3, 3]

        try:
            U, S, V = torch.linalg.svd(H) # U: [1,3,3], S: [1,3], V: [1,3,3] (V is V, not V.T)
        except torch.linalg.LinAlgError:
             # SVD may fail for degenerate cases, e.g. colinear points.
             # In such cases, we can't improve alignment, so return current X1.
             print("Warning: SVD in ICP failed. Returning current alignment.")
             return X1.squeeze(0)


        R = V @ U.transpose(1, 2) # R: [1, 3, 3]

        # Ensure proper rotation (no reflection)
        # If det(R) == -1, flip the sign of the last column of V (or U) and recompute R
        # A simpler way: R[torch.det(R) < 0, :, 2] *= -1 (modifies last column of R directly)
        # Or more robustly for V U.T:
        if torch.det(R) < 0:
            # print("ICP: Correcting reflection in rotation matrix.")
            V_prime = V.clone()
            V_prime[:, :, 2] *= -1 # Flip last column of V
            R = V_prime @ U.transpose(1, 2)
            if torch.det(R) < 0: # Should not happen now unless S has zeros
                print("Warning: ICP reflection correction failed. Using original R.")
                R = V @ U.transpose(1,2) # Revert if still bad


        # Apply transformation: X1_new = (X1 - t1) @ R.T + t2
        # Transpose R because points are row vectors and R from SVD typically transforms column vectors (p' = Rp)
        # So for row vectors P: P' = P @ R_transpose
        X1 = (X1_centered @ R.transpose(1,2)) + t2

    return X1.squeeze(0) # [N, 3]

def save_two_clouds_ply(pred: torch.Tensor,
                        gt  : torch.Tensor,
                        out_pred: str = "pred_norm.ply",
                        out_gt  : str = "gt_norm.ply"):
    """
    把归一化后的 pred / gt 点云各自保存成 .ply，方便用 Meshlab / CloudCompare 查看。
    """
    # 生成随机颜色或固定颜色
    col_pred = np.tile(np.array([[255,  32,  32]]), (pred.shape[0], 1))  # 红
    col_gt   = np.tile(np.array([[ 32, 255,  32]]), (gt.shape[0],   1))  # 绿

    trimesh.PointCloud(pred.cpu().numpy(), col_pred).export(out_pred)
    trimesh.PointCloud(gt .cpu().numpy(), col_gt  ).export(out_gt)
    print(f"Saved to {out_pred} / {out_gt}")
@torch.no_grad()
def normalize_pair_max_edge(pc_pred: torch.Tensor,
                            pc_gt:   torch.Tensor,
                            eps: float = 1e-7):
    """
    最长轴归一化（max-edge），保证 pred / gt 位于同一尺寸框内
    Args:
        pc_pred, pc_gt : [B, N, 3]  (B 可以不同，但通常 B=1)
    Returns:
        pc_pred_norm, pc_gt_norm : 同 shape
    """
    # 1. 拼接求共同质心
    pc_all      = torch.cat([pc_pred, pc_gt], dim=1)
    center      = pc_all.mean(dim=1, keepdim=True)            # [B,1,3]

    # 2. 计算共同最长轴
    pc_all_c    = pc_all - center
    lengths     = pc_all_c.max(dim=1, keepdim=True)[0] - pc_all_c.min(dim=1, keepdim=True)[0]  # [B,1,3]
    max_len     = lengths.max(dim=-1, keepdim=True)[0]        # [B,1,1]

    # 3. 分别归一化
    pc_pred_n   = (pc_pred - center) / (max_len + eps)
    pc_gt_n     = (pc_gt   - center) / (max_len + eps)
    # save_two_clouds_ply(pc_pred_n,pc_gt_n)
    return pc_pred_n, pc_gt_n
def normalize_pc_max_edge_indiv(pc: torch.Tensor, eps: float = 1e-7):
    """
    单个点云归一化——最长轴缩放到 1
    Args:  pc  [B, N, 3]
    Returns: 同 shape
    """
    center = pc.mean(dim=1, keepdim=True)
    pc_c   = pc - center
    lengths = pc_c.max(dim=1, keepdim=True)[0] - pc_c.min(dim=1, keepdim=True)[0]   # [B,1,3]
    max_len = lengths.max(-1, keepdim=True)[0]                                      # [B,1,1]
    return pc_c / (max_len + eps)
@torch.no_grad()
def find_best_coarse_alignment(pc_pred_orig: torch.Tensor,
                               pc_gt_orig:   torch.Tensor,
                               device: str = "cuda"):
    """
    Brute-force rotation grid search.
    Args:
        pc_pred_orig : [N_pred, 3]  predicted cloud
        pc_gt_orig   : [N_gt , 3]  ground-truth cloud
    Returns:
        best_pc_pred_norm : [N_pred, 3]  (rotated & normalized)
        best_pc_gt_norm   : [N_gt , 3]  (normalized with same center+scale)
    """
    pc_pred = pc_pred_orig.unsqueeze(0).to(device)   # [1, N_pred, 3]
    pc_gt   = pc_gt_orig.unsqueeze(0).to(device)     # [1, N_gt , 3]

    rotations     = get_rotation_sphere(azim_sample=24,
                                        elev_sample=24,
                                        roll_sample=24,
                                        scales=[1.0]).to(device)
    batch_size_rot = 8

    best_cd_val               = float("inf")
    best_pc_pred_norm         = pc_pred.clone().squeeze(0)   # placeholder
    best_pc_gt_norm           = pc_gt.clone().squeeze(0)

    for i in range(0, len(rotations), batch_size_rot):
        R_batch  = rotations[i:i + batch_size_rot]           # [B_r, 3, 3]
        B_r      = R_batch.shape[0]

        # rotate pred
        pc_pred_rot = torch.bmm(pc_pred.repeat(B_r, 1, 1),   # [B_r,N,3]
                                R_batch.transpose(1, 2))

        # pair-normalization  (共享 center & scale)
        pc_gt_rep = pc_gt.repeat(B_r, 1, 1)
        pc_pred_norm = normalize_pc_max_edge_indiv(pc_pred_rot)
        pc_gt_norm     = normalize_pc_max_edge_indiv(pc_gt_rep)

        # pc_pred_norm, pc_gt_norm = normalize_pair_max_edge(pc_pred_rot,
        #                                                    pc_gt_rep)
        # CD
        acc, comp, _, _ = chamfer_distance_wrapper(pc_pred_norm,
                                                   pc_gt_norm)
        cd_batch = (acc.mean(dim=1) + comp.mean(dim=1)) / 2   # [B_r]

        # keep best
        min_cd, min_idx = torch.min(cd_batch, dim=0)
        if min_cd < best_cd_val:
            best_cd_val       = min_cd.item()
            best_pc_pred_norm = pc_pred_norm[min_idx].clone()
            best_pc_gt_norm   = pc_gt_norm[min_idx].clone()
            best_R            = R_batch[min_idx].clone()      # ②  remember R
            
    # save_two_clouds_ply(best_pc_pred_norm, best_pc_gt_norm)
    for name, pc in [("pred", best_pc_pred_norm), ("gt", best_pc_gt_norm)]:
        bbox = pc.max(0).values - pc.min(0).values
        print(name, bbox)        # 三个轴的边长应非常接近
    return best_pc_pred_norm, best_pc_gt_norm, best_R     
# ==== 1.  欧拉角 -> 旋转矩阵  ========================================
def euler_zyx_deg_to_rotmat(yaw_deg: float, pitch_deg: float, roll_deg: float,
                            device: str = "cpu") -> torch.Tensor:
    """Z-Y-X (yaw-pitch-roll) to 3×3 rotation matrix."""
    y, p, r = map(math.radians, [yaw_deg, pitch_deg, roll_deg])
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)

    Rz = torch.tensor([[cy, -sy, 0],
                       [sy,  cy, 0],
                       [ 0,   0, 1]], dtype=torch.float32, device=device)
    Ry = torch.tensor([[cp, 0, sp],
                       [ 0, 1, 0],
                       [-sp, 0, cp]], dtype=torch.float32, device=device)
    Rx = torch.tensor([[1, 0,  0],
                       [0, cr, -sr],
                       [0, sr,  cr]], dtype=torch.float32, device=device)
    return Rz @ Ry @ Rx                      # 3×3



def evaluate_obj_files(obj_file1_path, obj_file2_path, num_points, f_thresholds,
                       device, use_icp=False, icp_iterations=500,save_obj=False):
    """
    Main evaluation function.
    """
    print(f"Loading OBJ files: {obj_file1_path}, {obj_file2_path}")
    try:
        mesh1 = trimesh.load_mesh(obj_file1_path, force='mesh')
        mesh2 = trimesh.load_mesh(obj_file2_path, force='mesh')
    except Exception as e:
        print(f"Error loading OBJ files: {e}")
        return None

    if not (isinstance(mesh1, trimesh.Trimesh) and len(mesh1.vertices) > 0 and
            isinstance(mesh2, trimesh.Trimesh) and len(mesh2.vertices) > 0):
        print("Error: One or both OBJ files could not be loaded as valid meshes or are empty.")
        return None

    print(f"Sampling {num_points} points from each mesh...")
    try:
        points1_np = mesh1.sample(num_points)
        points2_np = mesh2.sample(num_points)
        # points2_np = mesh2
    except Exception as e:
        print(f"Error sampling points: {e}")
        return None
    
    if points1_np.shape[0] < num_points * 0.5 or points2_np.shape[0] < num_points * 0.5:
        print(f"Warning: Sampling resulted in fewer points than expected. "
              f"Mesh1: {points1_np.shape[0]}, Mesh2: {points2_np.shape[0]}. Check mesh integrity.")
        if points1_np.shape[0] == 0 or points2_np.shape[0] == 0:
            print("Error: Zero points sampled from at least one mesh.")
            return None

    points1 = torch.from_numpy(points1_np).float().to(device)
    points2 = torch.from_numpy(points2_np).float().to(device)

    print("Performing coarse alignment (brute-force search)...")
    # pc_pred_coarse_aligned is [N_pred, 3], pc_gt_normalized is [N_gt, 3]
    pc_pred_coarse_aligned, pc_gt_normalized,best_R = find_best_coarse_alignment(points1, points2, device)
    R_to_apply = best_R            # 这是 Blender 需要的矩阵
    yaw2, pitch2, roll2 = rotation_matrix_to_euler_zyx_deg(R_to_apply)
    print('yaw2, pitch2, roll2',yaw2, pitch2, roll2)
    rotation_deg = rotation_matrix_to_angle_deg(best_R)
    yaw_deg, pitch_deg, roll_deg = rotation_matrix_to_euler_zyx_deg(best_R)
    final_pc_pred_aligned = pc_pred_coarse_aligned
    if save_obj:
        save_point_cloud_obj(pc_pred_coarse_aligned, "pred_aligned.obj")
        save_point_cloud_obj(pc_gt_normalized,       "gt_normalized.obj")
    # if use_icp:
    # print(f"Performing ICP refinement with {icp_iterations} iterations...")
    # final_pc_pred_aligned = icp_refine(pc_pred_coarse_aligned, pc_gt_normalized,
    #                                     num_iter=icp_iterations, device=device)
    
    # Calculate final metrics
    # Add batch dimension for chamfer_distance_wrapper and compute_fscore
    final_pc_pred_aligned_batch = final_pc_pred_aligned.unsqueeze(0)
    pc_gt_normalized_batch = pc_gt_normalized.unsqueeze(0)

    acc_dist, comp_dist, _, _ = chamfer_distance_wrapper(
        final_pc_pred_aligned_batch,
        pc_gt_normalized_batch
    )

    f_scores_values = compute_fscore(acc_dist, comp_dist, f_thresholds) # [1, num_thresholds]
    
    cd_accuracy_val = acc_dist.mean().item()
    cd_completeness_val = comp_dist.mean().item()
    chamfer_dist_val = (cd_accuracy_val + cd_completeness_val) / 2.0

    return {
        "cd": chamfer_dist_val,
        "cd_accuracy": cd_accuracy_val,
        "cd_completeness": cd_completeness_val,
        "f_scores": f_scores_values.squeeze(0).cpu().tolist(), # list of f-scores
        "yaw_pitch_roll_deg": (yaw_deg, pitch_deg, roll_deg),
        "rot_angle_deg": rotation_deg,


    }
def save_rotated_mesh(src_obj_path: str,
                      axis: str,
                      angle_deg: float,
                      out_path: str):
    """
    读取 OBJ → 旋转 → 保存新 OBJ
    axis ∈ {'x','y','z'}
    """
    mesh = trimesh.load_mesh(src_obj_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f'Cannot load mesh from {src_obj_path}')

    angle_rad = math.radians(angle_deg)
    if axis.lower() == 'x':
        R = trimesh.transformations.rotation_matrix(angle_rad, [1,0,0])
    elif axis.lower() == 'y':
        R = trimesh.transformations.rotation_matrix(angle_rad, [0,1,0])
    elif axis.lower() == 'z':
        R = trimesh.transformations.rotation_matrix(angle_rad, [0,0,1])
    else:
        raise ValueError('axis must be x, y or z')

    mesh.apply_transform(R)
    mesh.export(out_path)
    print(f'Saved rotated mesh: {out_path}')

def main():
    import time
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Evaluate Chamfer Distance and F-score between two OBJ files, with optional ICP.")
    parser.add_argument("--obj_file1", type=str,default='/inspire/hdd/project/robot-dna/public/dataset/image2code_proj/OmniObject3D/omniobject3d___OmniObject3D-New/raw/collected_pair_data/kettle/kettle_005_02/Scan.obj', help="Path to the first OBJ file (e.g., prediction).")
    parser.add_argument("--obj_file2", type=str, default='/inspire/hdd/project/robot-dna/public/eval/qwen25vl7b_base_30catspose0_image2code_30cats_multipose3_white,internvl_sft_data,sharegpt-4o,llava_nodes_8_device_batch_4_grad_accu_1_lr_5.0e-6/ood/checkpoint-10000/Kettle/73/generated_obj.obj', help="Path to the second OBJ file (e.g., ground truth).")
    parser.add_argument("--num_points", type=int, default=30000, help="Number of points to sample from each mesh.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use ('cuda' or 'cpu').")
    parser.add_argument("--f_thresholds", type=str, default="0.005,0.01,0.02,0.05",
                        help="Comma-separated F-score thresholds.")
    parser.add_argument("--use_icp", action='store_true', help="Enable ICP refinement after brute-force alignment.")
    parser.add_argument("--icp_iterations", type=int, default=2000, help="Number of iterations for ICP.")
    parser.add_argument("--debug_angle", type=float, default=45.0,
                    help="Rotation angle (deg) used when --debug_rotate is on.")
    args = parser.parse_args()

    f_threshold_list = [float(t) for t in args.f_thresholds.split(',')]
    
    # print(f"Using device: {args.device}")
    results = evaluate_obj_files(
        args.obj_file1, args.obj_file2, args.num_points, f_threshold_list,
        args.device, args.use_icp, args.icp_iterations,save_obj = True
    )

    if results:
        print("\n--- Evaluation Results ---")
        print(f"Chamfer Distance (CD): {results['cd']:.6f}")
        print(f"  CD Accuracy (Pred -> GT): {results['cd_accuracy']:.6f}")
        print(f"  CD Completeness (GT -> Pred): {results['cd_completeness']:.6f}")
        
        print("\nF-scores @ d:")
        for i, threshold in enumerate(f_threshold_list):
            print(f"  F-score @ {threshold:.4f}: {results['f_scores'][i]:.6f}")
        print("\n--- Pose difference (pred ➜ GT canonical) ---")
        print(f"Rotation angle   : {results['rot_angle_deg']:.2f}°")
        y, p, r = results['yaw_pitch_roll_deg']
        print(f"Yaw / Pitch / Roll: {y:.2f}° / {p:.2f}° / {r:.2f}°")
    print(t0-time.time())
    
    
    # axes = ['x','y','z']
    # for ax in axes:
    #     rotated_path = f"{Path(args.obj_file1).stem}_rot{ax.upper()}.obj"
    #     save_rotated_mesh(args.obj_file1, ax, args.debug_angle, rotated_path)

    #     print(f"\n===== Evaluate {ax}-axis {args.debug_angle}° rotation =====")
    #     res = evaluate_obj_files(rotated_path, args.obj_file2,
    #                              args.num_points, f_threshold_list,
    #                              args.device, args.use_icp, args.icp_iterations)
    #     if res:
    #         y,p,r = res['yaw_pitch_roll_deg']
    #         print(f"Expect around {args.debug_angle}° on {ax.upper()}  axis.")
    #         print(f"Predicted Yaw/Pitch/Roll: {y:.2f} / {p:.2f} / {r:.2f}")
    #         print(f"Predicted overall angle : {res['rot_angle_deg']:.2f}°")
    #     else:
    #         print("Evaluation failed for rotated mesh.")
if __name__ == "__main__":
    main()
