"""Bounded saved-data frequency/FK analysis; no policy forward or simulation."""
from pathlib import Path
import ast
import hashlib
import itertools
import json
import math
import sys
import numpy as np
from scipy import signal
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
OUT=Path(__file__).resolve().parent


def rotation(q):
    return Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()


def main():
    source=ROOT/'outputs/diagnostics_v1/formal_P01_CP178432_failure.json'
    data=json.loads(source.read_text())
    rows=[r for r in data['physics_native_rows'] if 3000<=r['tick']<=5800]
    assert [r['tick'] for r in rows]==list(range(3000,5801))
    values={'FL_hip_mapped_N_deg':[r['FL_mapped_N_deg'][0] for r in rows],
        'FL_hip_final_deg':[r['FL_final_target_deg'][0] for r in rows],
        'FL_hip_actual_deg':[r['FL_actual_deg'][0] for r in rows],
        'FL_knee_actual_deg':[r['FL_actual_deg'][1] for r in rows],
        'FL_hip_raw':[r['FL_raw'][0] for r in rows],
        'FL_gap_mm':[r['FL_gap_m']*1000 for r in rows],
        'base_z_mm':[r['base']['position_w_m'][2]*1000 for r in rows]}
    spectra={}
    for key,values_i in values.items():
        f,p=signal.welch(values_i,fs=120,nperseg=1024,detrend='linear')
        indices=np.argsort(p[1:])[-4:][::-1]+1
        spectra[key]=dict(dominant_frequency_Hz=float(f[indices[0]]), peak_frequencies_Hz=f[indices].tolist(),
            peak_PSD=p[indices].tolist(), min=min(values_i),max=max(values_i),std=float(np.std(values_i)))
    phase={}
    for key in ('FL_hip_actual_deg','FL_gap_mm'):
        f,c=signal.csd(values['FL_hip_final_deg'],values[key],fs=120,nperseg=1024,detrend='linear')
        _,co=signal.coherence(values['FL_hip_final_deg'],values[key],fs=120,nperseg=1024)
        i=int(np.argmin(abs(f-7.5)))
        phase[key]=dict(frequency_Hz=float(f[i]), response_relative_to_target_phase_deg=float(np.angle(c[i])*180/np.pi),
                       magnitude_squared_coherence=float(co[i]), closed_loop_not_causal_transfer=True)
    changes=[rows[i]['tick'] for i in range(1,len(rows)) if values['FL_hip_mapped_N_deg'][i]!=values['FL_hip_mapped_N_deg'][i-1]]
    eligible=[r['FL_tracking_channels'][0] for r in rows if r['FL_tracking_channels'][0]['feedback_eligible']]
    reference=dict(feedback_update_interval_ticks=sorted(set(np.diff(changes).tolist())),
        eligible_samples=len(eligible),reference_used=sum(r['reference_used'] for r in eligible),
        clipped_samples=sum(r['desired_reference_clipped'] for r in eligible),
        max_c1_minus_clamped_c0_plus_gain_r_error=max(abs(r['desired_reference_c1_deg']-np.clip(r['desired_original_c0_deg']+8*r['active_reference_deg'],-10,10)) for r in eligible))
    chain_path=Path('C:/robotics_sim/wlr_robot/fsm_50mm_recording_sensor_fsm_v1/src/wlr50/fsm/frozen_usd_kinematic_chain.py')
    tree=ast.parse(chain_path.read_text())
    chain=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FROZEN_USD_KINEMATIC_CHAIN' for t in n.targets))
    asset=Path(chain['usd_asset_path_for_provenance_only'])
    assert hashlib.sha256(asset.read_bytes()).hexdigest()==chain['usd_asset_sha256']
    native_path=Path(data['source'])/'native_tick_audit.jsonl'
    with native_path.open() as stream:
        n=json.loads(next(itertools.islice(stream,2999,3000)))
    assert n['episode_physics_tick']==3000
    tracking=n['native_audit']['tracking_reference_evidence']
    standing=np.array(tracking['standing_pose_deg'][:2])
    geometric=[]
    for row in (rows[0],rows[1000],rows[-1]):
        def fk(q):
            T=np.eye(4);T[:3,:3]=rotation(row['base']['orientation_wxyz']);T[:3,3]=row['base']['position_w_m']
            for joint,angle in zip(chain['joints'][:3],[*q,0.]):
                A=np.eye(4);A[:3,:3]=rotation(joint['localRot0']);A[:3,3]=joint['localPos0']
                B=np.eye(4);B[:3,:3]=rotation(joint['localRot1']);B[:3,3]=joint['localPos1']
                Z=np.eye(4);Z[:3,:3]=Rotation.from_rotvec([0.,0.,math.radians(angle)]).as_matrix()
                T=T@A@Z@np.linalg.inv(B)
            return T[:3,3]
        q=np.array(row['FL_actual_deg'])+standing
        J=np.column_stack([(fk(q+np.eye(2)[i]*.001)-fk(q-np.eye(2)[i]*.001))/(.002)*1000 for i in range(2)])
        geometric.append(dict(tick=row['tick'],actual_canonical_deg=row['FL_actual_deg'],standing_deg=standing.tolist(),
            measured_center_w_m=row['FL_center_w_m'],FK_center_w_m=fk(q).tolist(),
            FK_center_error_mm=((fk(q)-row['FL_center_w_m'])*1000).tolist(),
            xyz_J_mm_per_canonical_degree=J.tolist(), gap_mm=row['FL_gap_m']*1000,
            hip_minus_1p5_estimated_xyz_mm=(J@[-1.5,0.]).tolist(),
            combo_minus1p5_plus1_estimated_xyz_mm=(J@[-1.5,1.]).tolist()))
    # Pure current production reference arithmetic: if a sustained residual is
    # physically achieved, the eligible reference does not ask nominal to undo it.
    from wlr50_clean.ppo.semantic_tracking_reference import build_tracking_reference,CONTEXT_SCHEMA
    ref_tests=[]
    for bias in (-1.5,1.5):
        context=json.loads(json.dumps(tracking));context['schema']=CONTEXT_SCHEMA
        context['mapper_feedback_tick']=context['previous_ack_write_count']=context['previous_feedback_sample_tick']+1
        context['mapper_feedback_tick']+=(-context['mapper_feedback_tick'])%4
        context['previous_ack_write_count']=context['mapper_feedback_tick']
        context['previous_feedback_sample_tick']=context['mapper_feedback_tick']-1
        context['previous_requested_full12'][0]=bias
        nominal=context['mapper_pre_state']['requested_servo_deg'][0]
        context['actual_measured_physical_rad'][0]=math.radians(context['standing_pose_deg'][0]+nominal+bias)
        result=build_tracking_reference(context,requested_command_deg=context['mapper_pre_state']['requested_servo_deg'],tracking_servo_names=['front_left_hip'])
        channel=result['channels'][0]
        assert abs(channel['bounded_desired_c1_deg'])<1e-10
        ref_tests.append(dict(physical_bias_deg=bias,original_c0_deg=channel['desired_original_c0_deg'],
            referenced_desired_c1_deg=channel['bounded_desired_c1_deg'],arithmetic_only_not_physics=True))
    result=dict(schema='wlr50_clean.saved_FL_frequency_and_probe_design.v1',source=str(source),
        sealed_rows_available=3408,spectral_window_ticks=[3000,5800],spectral_samples=len(rows),sample_Hz=120,
        welch=dict(nperseg=1024,detrend='linear',window='hann',frequency_resolution_Hz=120/1024),
        spectra=spectra,phase=phase,tracking_reference=reference,constant_bias_reference_examples=ref_tests,
        FK=dict(frozen_chain_source=str(chain_path),asset_sha256=chain['usd_asset_sha256'],
                semantics='same-measured-base wheel-center FK, not collider minimum or floating/contact response',samples=geometric),
        probes=dict(hip_deg=[-1.5,1.5],optional_combo_deg=[-1.5,1.],ramp_s=8/15,hold_s=32/15,
                    residual_anchor='actual previous filtered requested FL residual at live P05 source endpoint',
                    sign_validated_by_live_same_state_Jacobian=True),
        no_new_simulation_or_policy_forward=True,no_mapper_change=True,
        caveats=['Feedback 30Hz update is not the measured 7.5Hz actual oscillation.',
                 'Closed-loop coherence and phase are association, not open-loop causal identification.',
                 'Positive/negative physical experiments must verify effective offset, q, FL contact and body coupling.',
                 'Old reference arithmetic is distinct from the already repaired recursive nominal-history bug.'])
    with (OUT/'sealed_FL_frequency_and_probe_design.json').open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({'reference':reference,'FK':geometric,'phase':phase},indent=2))


if __name__=='__main__':main()
