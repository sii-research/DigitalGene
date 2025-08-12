import json
import numpy as np
from scipy.spatial.transform import Rotation as R

# ----------------------------------------------------------------------
# 0.  NVIDIA helper (原样复制)
# ----------------------------------------------------------------------
def line_distance(a_o, a_d, b_o, b_d):
    n = np.cross(a_d, b_d); n_len = np.linalg.norm(n)
    if n_len < 1e-6:                                      # 平行
        return np.linalg.norm(np.cross(b_o - a_o, a_d))
    return abs(np.dot(n, a_o - b_o)) / n_len

# ----------------------------------------------------------------------
# 1.  读取 GT 轴   ------------------------------------------------------
# ----------------------------------------------------------------------
gt_json = json.loads(
    '{"input":{"joint_id":0,"motion":{"type":"rotate","rotate":[0.0,-45.0],"translate":[0.0,0.0]}},'
    '"trans_info":{"axis":{"o":[-0.2405585040450098,0.3870803675651547,0.0],"d":[0.0,0.0,1.0]},'
    '"rotate":{"l":0.0,"r":-45.0},"type":"rotate"}}'
)  #from paris dataset, USB_100109/gt/trans.json

o_gt = np.array(gt_json['trans_info']['axis']['o'])      # 世界坐标
d_gt = np.array(gt_json['trans_info']['axis']['d'])      # (0,0,1)

# ----------------------------------------------------------------------
# 2.  读取 *预测* JSON 并构造轴   ---------------------------------------
# ----------------------------------------------------------------------
with open('predictions/USB/1/generated_code.json', 'r') as f:      # the json file that corresponding to the predicted object file
    pred_data = json.load(f)

# ---- (2.1) 全局刚体位姿 ----------------------------------------------
t_global = np.array(pred_data['pose']['global_position'])       # (x,y,z)  (m)
R_global = R.from_euler(
    'xyz', np.deg2rad(pred_data['pose']['global_rotation'])     # Euler XYZ, °→rad
)

# ---- (2.2) 找到 Regular_Connector 以读取 shaft_offset[0] -------------
conn = next(item['parameters'] for item in pred_data['conceptualization']
            if item['template'] == 'Regular_Connector')

shaft_offset = conn['thickness'][0]                      # **关键字段 → pivot z**

# ---- (2.3) 在 *USB 局部坐标* 下的轴 -------------------------------
p_local = np.array([0.0, 0.0, shaft_offset])             # 轴上一点
d_local = np.array([0.0, 1.0, 0.0])                      # 平行 y 轴

# ---- (2.4) 变到世界坐标 -------------------------------------------
o_pred = R_global.apply(p_local) + t_global
d_pred = R_global.apply(d_local)
d_pred /= np.linalg.norm(d_pred)

# ----------------------------------------------------------------------
# 3.  Axis Ang Err (°) & Axis Pos Err (×0.1 m)  -------------------------
# ----------------------------------------------------------------------
# --- 角度 ---
cos_val = np.dot(d_pred, d_gt) / (np.linalg.norm(d_pred) * np.linalg.norm(d_gt))
ang_raw = np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0)))
axis_ang_err = min(ang_raw, 180.0 - ang_raw)             # 折叠到 0–90°

# --- 位置 ---
dist_m = line_distance(o_pred, d_pred, o_gt, d_gt)       # 单位 m
axis_pos_err = dist_m * 10.0                             # 转 “0.1 m”

print(f'AxisAngErr = {axis_ang_err:.2f}°')
print(f'AxisPosErr = {axis_pos_err:.2f}  (×0.1 m)')
