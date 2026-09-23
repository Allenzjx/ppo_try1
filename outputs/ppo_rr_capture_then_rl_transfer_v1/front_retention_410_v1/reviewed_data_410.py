"""Same frozen block03 raw labels; explicit, checked 389-to-410 data admission.

This is historical off-policy supervision, not a replayed current trajectory.
Only existing exact389 rows are read. No simulator or optimizer is called.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import torch
import yaml

ROOT=Path(__file__).resolve().parents[3]
LEGACY=ROOT/'outputs/ppo_p05_hip_only_continuation_v1/front_rehearsal_v1'
LEGACY_SHA='3ee15f8ce3bb6c1a88ce9204298cfaa8d6b8b4c162349ee0c459b0071ce3c0dd'
SCHEMA='wlr50_clean.front_retention410_reviewed_data.v1'
SOURCE_FILES={
    'residual_and_projection_audit.jsonl':'3c01be2834dbd43f7fe02af80fe4d48c8bdc0ad129cfd1ccddf4f85fed060639',
    'rollouts/rollout_001558.pt':'9cc4bfd238166faaa49e7089a5a16f20e0fc61d45e0aa3a80fb2e8d661a85490',
    'rollouts/rollout_001559.pt':'f6ac3005d801463747b44a3121c16fd0df3f2fe401c8ff308f57a05e06900cc8',
    'rollouts/rollout_001560.pt':'e65b1a1ba2f57f0d67b669e391642dc2b0533d6ee27c275591b2f93e119dd021',
    'rollouts/rollout_001561.pt':'2e5b42709feb5a999e8761324b478e8763d59ee3338405cbec4489acced1bbec',
    'run_manifest.json':'e8c67aefec1a8c305e91b20db944abea2e0522d02fa364e6446d97b200044ca9',
    'training_manifest.json':'ad0ef71300578509a9ebabb60c8f623296987c707c2d5130f444dbe478020e49',
}


def require(condition,message):
    if not condition: raise ValueError(message)


def sha(path):
    from wlr50_clean.ppo.semantic_migration import file_sha
    return file_sha(Path(path))


def digest(value):
    from wlr50_clean.ppo.semantic_migration import digest as f
    return f(value)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def legacy_reader():
    path=LEGACY/'reviewed_data.py'
    require(sha(path)==LEGACY_SHA,'immutable original data reader changed')
    spec=importlib.util.spec_from_file_location('_front_retention410_legacy_data',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def prove_zero_tail_input(obs,row,previous,*,index):
    """Logical proof from measured fields; never assume a phase-wide zero mask.

    RR assist cannot initialize before P09, and all seven context predicates
    are false when actual RR is unqualified and has no obstacle TOP pair.
    Wheel armed scope is P09/P12 only. The reset row uses its exact saved
    contact/history features instead of fabricating a reset physical frame.
    """
    phase=int(obs[:13].argmax())+1
    require(phase<=6 and float(obs[:13].sum())==1.,'data left pre-RR contiguous source scope')
    require(all(obs[k].item()==0. for k in (149,153,157)),
        'source input has RR qualified/crossed/placed history; zero tail cannot be asserted')
    # Encoder order: FL,FR,RL,RR; each pair is ground then obstacle.
    require(obs[138].item()==0.,'source input has an RR obstacle contact pair')
    if index==0:
        require(previous is None and phase==1 and obs[18].item()==0.,
            'only exact saved natural-P01 reset may use reset implication proof')
        return {'index':0,'phase':'P01','proof':'exact_reset_history149_153_157_and_obstacle_pair138_zero',
            'full_reset_frame_reconstructed':False,'rr_assist_WAIT_induction':True,'tail21_zero':True}
    require(previous is not None and previous['applied_audit']['decision_count']+1==row['applied_audit']['decision_count'],
        'missing adjacent source input evaluator')
    a=previous['applied_audit'];ev=a['semantic_task']['physical_evaluator'];rr=ev['current_legs']['RR']
    require(ev['valid'] is True and ev.get('termination_reason') is None and a['semantic_task'].get('termination_reason') is None,
        'source input is physically invalid')
    require(ev['history']['active_lift']['RR'] is False and ev['history']['front_edge_crossed']['RR'] is False
        and ev['history']['placed']['RR'] is False and rr['obstacle_pair_active'] is False
        and rr['top_surface_contact'] is False and rr['contact_surface']!='TOP',
        'current RR context does not admit all-zero seven task flags')
    require(ev['physics_tick']==round(float(obs[18])*200.*120.),'adjacent evaluator and input clock differ')
    return {'index':index,'phase':f'P{phase:02d}','physics_tick':ev['physics_tick'],
        'RR_active_lift':False,'RR_crossed':False,'RR_placed':False,
        'RR_current_lift_valid':rr['current_lift_valid'],'RR_contact_surface':rr['contact_surface'],
        'RR_ground_contact':rr['ground_contact'],'RR_obstacle_pair_active':False,
        'RR_within_top_xy':rr['within_top_xy'],'RR_gap_m':rr['clearance_m'],
        'RR_front_distance_m':rr['front_distance_m'],'rr_assist_WAIT_induction':True,'tail21_zero':True}


def selection_indices(legacy=None):
    """Omit only the unobserved reset input; original source data is untouched."""
    legacy=legacy or legacy_reader()
    train,val,holdout=legacy.selection_indices()
    require(train==[0,1,*range(2,280,3)] and len(val)==93 and len(holdout)==13,
        'immutable original fixed split changed')
    train=train[1:]
    require(len(train)==94 and train[0]==1 and 0 not in train,
        'unproved reset input must not be a supervised current410 label')
    return train,val,holdout


def load_reviewed_data(metadata,contract):
    from wlr50_clean.ppo.semantic_migration import _version_bytes
    from wlr50_clean.ppo.semantic_policy_distribution import policy_contract
    from wlr50_clean.ppo.semantic_p05_capture_profile import P05_CAPTURE_POLICY,P05_CAPTURE_OBSERVATION_LAYOUT
    from wlr50_clean.ppo.semantic_rr_capture_profile import RR_CAPTURE_POLICY,RR_CAPTURE_OBSERVATION_LAYOUT
    from wlr50_clean.ppo.semantic_supervisor import TaskStageSupervisor,_current_rr_receiver_preparation_retired
    from wlr50_clean.ppo.semantic_rr_capture_assist import RRHipOnlyCaptureAssist,rr_capture_assist_features
    require(metadata['policy_contract']==policy_contract(RR_CAPTURE_POLICY,observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT),
        'candidate requires the actual current410 policy contract')
    legacy=legacy_reader();train,val,holdout=selection_indices(legacy);run=legacy.RUN
    for name,expected in SOURCE_FILES.items(): require(sha(run/name)==expected,'sealed source bytes changed: '+name)
    manifest=read(run/'training_manifest.json')
    require(manifest['lifecycle']=='SUCCEEDED' and manifest['stage']=='full_episode'
        and manifest['actual_policy_decisions']==2048 and not manifest['phase_suffix_curriculum_implemented']
        and not (run/'prefix_evidence.jsonl').exists(),'wrong original natural-P01 source')
    rows=[]
    with (run/'residual_and_projection_audit.jsonl').open(encoding='utf-8') as stream:
        for _ in range(465): rows.append(json.loads(next(stream)))
    xs,ys,ms,ss,ls=[],[],[],[],[];source_contract=None
    source_policy=policy_contract(P05_CAPTURE_POLICY,observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT)
    for update in range(1558,1562):
        batch=torch.load(run/f'rollouts/rollout_{update:06}.pt',map_location='cpu',weights_only=False)
        x=batch['observations']['policy'];y=batch['actions'];mu,sigma=batch['distribution_params']
        require(batch['schema']=='wlr50_clean.semantic_on_policy_rollout.v1' and x.shape==(128,1,389)
            and y.shape==(128,1,12) and torch.equal(x,batch['observations']['critic'])
            and batch['policy_contract']==source_policy and batch['curriculum_epoch']['prefix_request'] is None,
            'source exact389 raw rollout shape/profile changed')
        if source_contract is None: source_contract=batch['runtime_contract']
        require(batch['runtime_contract']==source_contract,'source rollout contracts differ')
        xs.extend(x[:,0]);ys.extend(y[:,0]);ms.extend(mu[:,0]);ss.extend(sigma[:,0]);ls.extend(batch['actions_log_prob'][:,0])
    x,y,mu,sigma,lp=[torch.stack(a[:465]) for a in (xs,ys,ms,ss,ls)]
    likelihood=max(legacy.validate_tensor_row(x[i],y[i],mu[i],sigma[i],lp[i],r,index=i) for i,r in enumerate(rows))
    require([r['applied_audit']['phase_id'] for r in rows[:280]]==['P01']*2+['P02']*278,'fixed front window changed')
    require(all(r['applied_audit']['phase_id'] in ('P01','P02','P03','P04','P05','P06') for r in rows),
        'WAIT induction requires the entire contiguous source prefix before P09')
    # Observation encoding is checked structurally, not by renaming the old contract.
    old_schema=json.loads(_version_bytes(ROOT,source_contract,source_contract['selected_configuration']['observation_schema.json']['path']))
    new_schema=read(ROOT/contract['selected_configuration']['observation_schema.json']['path'])
    require(new_schema['feature_groups'][:-2]==old_schema['feature_groups']
        and all(new_schema[k]==old_schema[k] for k in ('clip','fixed_chassis_to_body_wxyz','maximum_task_duration_s','normalization')),
        'old389 numeric codec changed beyond the declared append')
    for path in ('src/wlr50_clean/ppo/semantic_history_actor.py','src/wlr50_clean/ppo/semantic_p05_capture_actor.py',
                 'src/wlr50_clean/ppo/semantic_receiving_wheel_sigma.py'):
        require(source_contract['files'][path]==contract['files'][path], 'HISTORY/conditional-sigma kernel changed: '+path)
    old_spec=yaml.safe_load(_version_bytes(ROOT,source_contract,source_contract['selected_configuration']['stage_task_spec.yaml']['path']))
    new_spec=yaml.safe_load((ROOT/contract['selected_configuration']['stage_task_spec.yaml']['path']).read_text())
    from wlr50_clean.ppo.semantic_rr_carry_wheel import SEMANTICS as wheel_semantics
    from wlr50_clean.ppo.semantic_rr_postcapture_wheel_migration import WHEEL_SEMANTICS
    require(wheel_semantics==WHEEL_SEMANTICS,'current reviewed P09/P12 wheel scope changed')
    prior=object.__new__(TaskStageSupervisor);prior.spec=old_spec
    current=object.__new__(TaskStageSupervisor);current.spec=new_spec
    require(tuple(rr_capture_assist_features(RRHipOnlyCaptureAssist().snapshot()))==(0.,)*14,'RR WAIT14 is not zero')
    proofs=[];max_phi=0.
    for i,r in enumerate(rows):
        proofs.append(prove_zero_tail_input(x[i],r,rows[i-1] if i else None,index=i))
        before=r['applied_audit']['reward_breakdown']['potential_before']
        require(x[i,17].item()==torch.tensor(before,dtype=torch.float32).item(),'stored Phi differs from actual pre-state')
        if i:
            ev=rows[i-1]['applied_audit']['semantic_task']['physical_evaluator']
            old_phi,new_phi=prior.physical_potential(ev),current.physical_potential(ev)
            max_phi=max(max_phi,abs(new_phi-old_phi))
            require(abs(old_phi-before)<1e-12 and new_phi==old_phi
                and torch.tensor(new_phi,dtype=torch.float32).item()==x[i,17].item()
                and not _current_rr_receiver_preparation_retired(new_spec,'RR',ev),
                'current X17 does not retain the source front-state semantics')
    for index,leg,tick in ((283,'FR',2268),(460,'FL',3687)):
        ev=rows[index]['applied_audit']['semantic_task']['physical_evaluator'];legrow=ev['current_legs'][leg]
        require(ev['history']['placed'][leg] and ev['history']['event_ticks']['placed'][leg]==tick
            and legrow['top_contact'] and legrow['top_surface_contact'] and legrow['contact_surface']=='TOP'
            and legrow['within_top_xy'] and legrow['within_lateral_span'] and not legrow['ground_contact']
            and legrow['bearing_verified'] and legrow['bearing_force_n']>0.,'missing actual later legal front capture')
    expanded=torch.cat((x,torch.zeros((len(x),21),dtype=x.dtype)),dim=-1)
    groups={name:{'indices':ids,'count':len(ids),'phases':sorted(set(rows[i]['applied_audit']['phase_id'] for i in ids)),
        'source389_sha256':legacy.tensor_sha(x[ids]),'current410_sha256':legacy.tensor_sha(expanded[ids]),
        'actual_raw12_sha256':legacy.tensor_sha(y[ids])} for name,ids in (('train',train),('validation',val),('invariance',holdout))}
    receipt={'schema':SCHEMA,'source_run':str(run),'source_files_sha256':SOURCE_FILES,'legacy_reader_sha256':LEGACY_SHA,
        'source_runtime_contract_sha256':digest(source_contract),'current_runtime_contract_sha256':digest(contract),
        'source_current_checkpoint_sha256':metadata['checkpoint_sha256'],'groups':groups,
        'original389_direct_saved':True,'current410_direct_saved':False,'tail21_explicitly_derived_zero':True,
        'derivation':'RR WAIT induction over full contiguous P01-P06; actual RR active-lift false and no obstacle TOP pair; P09/P12 wheel scope inactive',
        'zero_tail_rows_checked':465,'per_row_proof_sha256':digest(proofs),'maximum_phi_semantic_difference':max_phi,
        'current_phi_rows_checked':464,'current_phi_checked_for_every_selected_input':True,
        'original_train_count':95,'excluded_supervised_indices':[0],
        'exclusion_reason':'reset input has no complete adjacent measured evaluator; current Phi equivalence is not claimed',
        'reviewed_static_zero_tail_premises':{'RR_pair_index138_from_unchanged_prefix_codec':True,
            'current_wheel_P09_P12_scope_semantics':wheel_semantics,
            'source_observation_codec_sha256':source_contract['files']['src/wlr50_clean/ppo/semantic_observation.py'],
            'current_observation_codec_sha256':contract['files']['src/wlr50_clean/ppo/semantic_observation.py'],
            'current_wheel_module_sha256':contract['files']['src/wlr50_clean/ppo/semantic_rr_carry_wheel.py']},
        'reset_row_proof':'historical saved history/pair provenance only; excluded from supervised current410 Phi admission',
        'raw_targets_original_unchanged':True,'conditional_mean_not_used_as_label':True,'old_GAE_or_logp_used_for_learning':False,
        'source_raw_likelihood_max_abs_error':likelihood,'train_validation_correlated_single_episode':True,
        'P01_independent_validation_count':0,'invariance_real_coverage':['P03','P04','P05','P06'],
        'real_P07_P13_holdout_coverage_claimed':False,'local_qualification':{'FR_placed_tick':2268,'FL_placed_tick':3687,
            'FL_later_capture_assisted':True,'episode_later_result':'P12_INCOMPLETE_CONTROLLER_BLOCKED'},
        'source_labels_scope':'local successful FR front approach only, not whole episode or unassisted FL capture',
        'prior_AUXFR1_deterministic_recovery_proved':False,'new_PPO_credit':0,'new_AUX_credit':0,
        'same_physical_trajectory_or_MDP_claimed':False}
    receipt['receipt_content_sha256']=digest(receipt)
    return {'train_observations':expanded[train],'train_raw_targets':y[train],
        'validation_observations':expanded[val],'validation_raw_targets':y[val],
        'invariance_observations':expanded[holdout],'receipt':receipt,'per_row_semantics_proof':proofs}
