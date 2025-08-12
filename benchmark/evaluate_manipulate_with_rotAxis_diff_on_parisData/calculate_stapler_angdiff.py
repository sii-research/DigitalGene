"""
Stapler ‒ hinge‐axis evaluation
==============================

* **GT axis** comes from the JSON snippet you provided (already loaded below).
* **Predicted axis** is derived from the *Simplified_Cover* block of the
  Stapler AOT file with the rule you specified:

      local_pivot = cover.position + [ 0,
                                       -size[1]/2,
                                       -size[2]/2 ]
      axis_dir    = parallel to +X (cover’s local frame)

  Then both point **and** direction are transformed by the cover’s own Euler
  rotation **and** the stapler’s global pose.

Outputs
-------
`AxisAngErr` (in degrees, 0–90)  
`AxisPosErr` (in units of 0.1 m, identical to NVIDIA “line_distance” metric)
"""

import json
import numpy as np
from scipy.spatial.transform import Rotation as R

# ----------------------------------------------------------------------
# 0.  NVIDIA helper (verbatim)
# ----------------------------------------------------------------------
def line_distance(a_o, a_d, b_o, b_d):
    n = np.cross(a_d, b_d)
    n_len = np.linalg.norm(n)
    if n_len < 1e-6:                                   # parallel lines
        return np.linalg.norm(np.cross(b_o - a_o, a_d))
    return abs(np.dot(n, a_o - b_o)) / n_len

# ----------------------------------------------------------------------
# 1.  Ground-truth axis  (pre-loaded from your GT snippet)
# ----------------------------------------------------------------------
gt_json = json.loads(
    '{"input":{"joint_id":1,"motion":{"type":"rotate","rotate":[0.0,-40.0],"translate":[0.0,0.0]}},'
    '"trans_info":{"axis":{"o":[0.0,0.752066445189761,0.10500870623151695],"d":[1.0,0.0,0.0]},'
    '"rotate":{"l":0.0,"r":-40.0},"type":"rotate"}}'
) #from paris dataset, stapler_103111/gt/trans.json
o_gt = np.array(gt_json['trans_info']['axis']['o'])
d_gt = np.array(gt_json['trans_info']['axis']['d'])

# ----------------------------------------------------------------------
# 2.  Predicted axis from Stapler AOT JSON  -----------------------------
# ----------------------------------------------------------------------
with open('predictions/Stapler/0/generated_code.json', 'r') as f:     # the json file that corresponding to the predicted object file
    pred = json.load(f)

# --- 2.1 global pose ---------------------------------------------------
t_global = np.array(pred['pose']['global_position'])
R_global = R.from_euler('xyz',
                        np.deg2rad(pred['pose']['global_rotation']))

# --- 2.2 locate Simplified_Cover parameters ---------------------------
cover = next(item['parameters'] for item in pred['conceptualization']
             if item['template'] == 'Simplified_Cover')

size_y, size_z = cover['size'][1], cover['size'][2]
cover_pos      = np.array(cover['position'])
R_cover        = R.from_euler('xyz', np.deg2rad(cover['rotation']))

# --- 2.3 local hinge pivot & direction --------------------------------
pivot_local = cover_pos + np.array([0.0, -0.5*size_y, -0.5*size_z])
d_local     = np.array([1.0, 0.0, 0.0])

# --- 2.4 transform to world -------------------------------------------
pivot_world = R_global.apply(R_cover.apply(pivot_local)) + t_global
d_world     = R_global.apply(R_cover.apply(d_local))
d_world    /= np.linalg.norm(d_world)

o_pred, d_pred = pivot_world, d_world   # alias for clarity

# ----------------------------------------------------------------------
# 3.  Axis-angle & position error   (identical to NVIDIA logic) ---------
# ----------------------------------------------------------------------
# --- angle error (0–90°) ----------------------------------------------
cos_val = np.dot(d_pred, d_gt) / (np.linalg.norm(d_pred)*np.linalg.norm(d_gt))
ang_raw = np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0)))
axis_ang_err = min(ang_raw, 180.0 - ang_raw)

# --- positional error (metres → ×0.1 m) -------------------------------
dist_m = line_distance(o_pred, d_pred, o_gt, d_gt)
axis_pos_err = dist_m * 10.0

print(f'AxisAngErr = {axis_ang_err:.2f}°')
print(f'AxisPosErr = {axis_pos_err:.2f}  (×0.1 m)')
