"""Read one sealed zero trajectory; angular envelope only, no visibility claim."""
import itertools
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'runs/ppo_fsm_reference_p09_stable_v2/video_eval/prior_B/20260915T0525295619176Z_g4a58c0190ef7_615f8fe3cfc64cff81f73ac6a0d302e6/source'
CAMERAS = {'historical': {'eye_m': [1.45,-1.25,.8], 'target_m': [.45,0,.12]},
           'review_v1': {'eye_m': [1.85,-1.65,1.15], 'target_m': [.70,-.15,.15]}}
minimum, maximum = np.full(3, np.inf), np.full(3, -np.inf)
count = 0
for line in (SOURCE / 'physical_observations.jsonl').open(encoding='utf8'):
    row = json.loads(line)
    # Pose-aware collider bounds are exact for base + four wheels. Other leg
    # body origins get 75 mm visual margin, not falsely labelled mesh bounds.
    bounds = row['body_bounds_w_m']
    for name, body in row['bodies'].items():
        if name in bounds:
            lo, hi = bounds[name]['minimum_m'], bounds[name]['maximum_m']
        else:
            p = np.asarray(body['position_w_m'])
            lo, hi = p-.075, p+.075
        minimum = np.minimum(minimum, lo)
        maximum = np.maximum(maximum, hi)
    count += 1
corners = np.asarray(list(itertools.product(*zip(minimum, maximum))))
results = {}
for name, camera in CAMERAS.items():
    eye, target = np.asarray(camera['eye_m']), np.asarray(camera['target_m'])
    forward = target-eye; forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0,0,1]); right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    delta = corners-eye
    depth = delta @ forward
    x, y = (delta @ right)/depth, (delta @ up)/depth
    max_h, max_v = float(np.abs(x).max()), float(np.abs(y).max())
    minimum_hfov = math.degrees(2*math.atan(max(max_h, max_v*16/9)))
    results[name] = {**camera, 'positive_depth': bool((depth>0).all()),
        'horizontal_tangent_bounds': [float(x.min()),float(x.max())],
        'vertical_tangent_bounds': [float(y.min()),float(y.max())],
        'minimum_horizontal_fov_deg_at_16_9': minimum_hfov,
        'minimum_horizontal_fov_deg_for_10pct_each_edge_margin': math.degrees(2*math.atan(max(max_h,max_v*16/9)/.8))}
receipt = {'schema':'wlr50_clean.review_camera_trajectory_envelope.v1', 'source':str(SOURCE),
    'rows':count, 'whole_run_world_envelope_m': {'minimum':minimum.tolist(),'maximum':maximum.tolist()},
    'cameras':results, 'physics_steps':0,
    'limitations':['Projection envelope is not render QA or an occlusion test.',
        'Base/four wheel collider AABBs are measured; other leg body origins have an approximate 75 mm margin.',
        'No focal length change; actual existing viewport lens must be checked on rendered first, FR, FL, RR, RL and ending frames.',
        'New home recovery may exceed the retained trajectory; full new-run visual QA is required.']}
output = Path(__file__).with_name('camera_trajectory_preflight.json')
output.write_text(json.dumps(receipt,indent=2),encoding='utf8')
print(json.dumps(receipt,indent=2))
