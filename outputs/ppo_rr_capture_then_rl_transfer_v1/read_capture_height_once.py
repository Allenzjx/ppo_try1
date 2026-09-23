"""Bounded CPU-only extraction of two sealed sources; stdout, no mutation."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASES = {
    'C_26db': ('runs/ppo_rr_capture_then_rl_transfer_v1/video_eval/validation/20260923T0356402230589Z_g26db2a1946e8_c1a7c95ca1db41d69882389e1e97cd8d/source', [9512,9520,9552,9560,9856,10152,10456,10752,10760,12484]),
    'accepted_N_ref': ('runs/ppo_task_conditioned_hip_wheel_v1/video_eval/prior_B/20260921T0701155546642Z_gee5a9651591d_64985c63292d4a9b966716c3a54659a1/source', [6136,6144,6152,6160,6176,6248]),
}
def value(row, key):
    obj = row.get(key)
    return obj.get('value') if isinstance(obj, dict) else obj

result = {'kind': 'sealed_readonly_descriptive_not_paired_counterfactual', 'cases': {}}
for label, (relative, ticks) in CASES.items():
    source = ROOT / relative
    selected = []
    with (source/'height_diagnostics.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if row['physics_tick'] not in ticks:
                continue
            out = {key: row.get(key) for key in ('physics_tick','simulation_time_s','phase','terminal_sample','clock_unchanged','body_collision_minimum_z_w_m')}
            out['rr_hip_mount_w_m'] = row['rr_hip_mount_w_m']
            out['base'] = value(row,'base')
            out['center_of_mass'] = value(row,'center_of_mass')
            out['joints_canonical'] = value(row,'joints_canonical')
            out['body_world'] = {}
            for body, wrapped in row['fresh_collider_bounds'].items():
                bounds = wrapped['value']
                out['body_world'][body] = None if bounds is None else {
                    key: bounds.get(key) for key in ('link_origin_w_m','minimum_m','maximum_m')}
            selected.append(out)
    metrics = []
    with (source/'phase_metrics.csv').open(encoding='utf-8-sig',newline='') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        for row in reader:
            tick_value = row.get('physics_tick',row.get('tick'))
            if tick_value and int(float(tick_value)) in ticks:
                metrics.append(row)
    result['cases'][label] = {'source': str(source), 'requested_ticks': ticks,
        'height_rows': selected,'phase_metrics_fields': fields,'phase_metrics_rows': metrics}
path=ROOT/'outputs/ppo_p05_hip_only_continuation_v1/CP218496_RR_capture_source_readonly.json'
with path.open(encoding='utf-8') as stream:
    old = json.load(stream)
result['accepted_N_prior_compact_evidence'] = old['accepted_N_ref']
print(json.dumps(result, allow_nan=False, separators=(',',':')))
