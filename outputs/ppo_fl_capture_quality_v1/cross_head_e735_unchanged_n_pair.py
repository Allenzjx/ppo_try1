"""Explicit cross-HEAD, unchanged-N offline comparison; not strict same-build pairing.

Only sealed review receipts. Never alters source receipts, strict pair rules,
production, physical simulations, or learned state. Fixed reviewed f2e -> e735 HEAD pair; prior helper remains untouched.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
spec=importlib.util.spec_from_file_location('fl_strict_event_media',OUT/'paired_event_media.py')
pair=importlib.util.module_from_spec(spec);spec.loader.exec_module(pair)
base,review,require=pair.base,pair.review_video,pair.require
B_HEAD='f2e552406ea74a5aae2803347d1c8bebe261910d'
C_HEAD='e73542cb57ad327a6246fcffb70d652a548627a3'
INTERMEDIATE_HEAD='3a50657a96c9beca6178d7f549750f21025a9fc6'
CHANGED={f'src/wlr50_clean/ppo/{name}.py' for name in (
 'semantic_history_actor','semantic_policy_distribution','semantic_training',
 'semantic_migration','semantic_cli','semantic_checkpoint_prefix_policy')}
CONFIG_NAMES={'action_schema.json','execution_profile.yaml','observation_schema.json',
 'quality_score.yaml','reward_config.yaml','stage_task_spec.yaml'}
SCENE_PATHS=('configs/environment_lock.json','configs/collision_role_map.json',
 'src/wlr50_clean/infrastructure/scene_factory.py',
 'src/wlr50_clean/infrastructure/robot_adapter.py',
 'src/wlr50_clean/infrastructure/servo_target_mapper.py')


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args])


def _version_bytes(head,path,expected):
    raw=git('show',f'{head}:{path}')
    for candidate in (raw,raw.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')):
        if hashlib.sha256(candidate).hexdigest()==expected:return candidate
    raise ValueError('Versioned byte binding differs: '+path)


def reviewed_code_delta():
    changed=set(git('diff','--name-only',B_HEAD,C_HEAD,'--','src','configs','scripts').decode().splitlines())
    require(changed==CHANGED,'Cross-head scope is not exactly the reviewed six learning files')
    legs=[]
    for start,end,label in ((B_HEAD,INTERMEDIATE_HEAD,'REQUEST_history_mean_entry'),
                            (INTERMEDIATE_HEAD,C_HEAD,'single_FR_knee_innovation_scale')):
        delta=set(git('diff','--name-only',start,end,'--','src','configs','scripts').decode().splitlines())
        require(delta==CHANGED,'Unreviewed intermediate code scope: '+label)
        legs.append({'source_HEAD':start,'target_HEAD':end,'change':label,'changed_files':sorted(delta)})
    # B shares these helpers, but never calls actor factories or checkpoint load.
    shared={}
    for path,names in {
      'src/wlr50_clean/ppo/semantic_training.py':('jsonable','write_json'),
      'src/wlr50_clean/ppo/semantic_cli.py':('runtime_contract','validate_request','version_paths','_request_paths','_validate_target_policy_request')}.items():
        trees=[ast.parse(git('show',f'{head}:{path}').decode()) for head in (B_HEAD,C_HEAD)]
        for name in names:
            nodes=[next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name) for tree in trees]
            require(ast.dump(nodes[0],include_attributes=False)==ast.dump(nodes[1],include_attributes=False),
                    'B-reachable shared helper changed: '+name)
            shared[path+':'+name]='AST identical'
    protected=('src/wlr50_clean/ppo/semantic_video_cli.py','src/wlr50_clean/ppo/semantic_video.py',
      'src/wlr50_clean/ppo/semantic_backend.py','src/wlr50_clean/ppo/semantic_env.py',
      'src/wlr50_clean/ppo/semantic_supervisor.py','src/wlr50_clean/ppo/semantic_residual_adapter.py',
      'src/wlr50_clean/ppo/semantic_nominal_geometry.py','src/wlr50_clean/ppo/rl_library_wrapper.py',*SCENE_PATHS)
    preflight=[]
    for head in (B_HEAD,C_HEAD):
        tree=ast.parse(git('show',f'{head}:src/wlr50_clean/ppo/semantic_cli.py').decode())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_preflight_checkpoint')
        prefix=[]
        for node in function.body:
            prefix.append(node)
            if isinstance(node,ast.If) and ast.unparse(node.test)=='args.checkpoint is None':break
        else:raise ValueError('B no-checkpoint early return missing')
        preflight.append(ast.dump(ast.Module(body=prefix,type_ignores=[]),include_attributes=False))
    require(preflight[0]==preflight[1],'B no-checkpoint preflight path changed')
    shared['semantic_cli._preflight_checkpoint:through_no_checkpoint_return']='AST identical'
    require(not git('diff','--name-only',B_HEAD,C_HEAD,'--',*protected).strip(),
            'B control/physics/scene path changed')
    return {'reviewed_HEADs':[B_HEAD,C_HEAD],'changed_runtime_paths':sorted(changed),
      'B_reachable_protected_paths_unchanged':list(protected),'shared_helpers':shared,
      'two_explicit_learning_boundaries':legs,
      'B_call_path':'checkpoint=None -> unchanged early return; roleB no checkpoint_loader; ZERO12 -> unchanged core.step/N/mapper/physics',
      'evidence_kind':'versioned source/config scope, not new physical replay or bitwise trajectory proof'}


def cross_head_common(b,c):
    require(b['receipt']['role']=='B' and b['receipt']['mode']=='N_plus_zero','Left must be B_control N+0')
    require(c['receipt']['role']=='C' and c['receipt']['mode']=='deterministic_conditional_mean','Right must be deterministic C')
    a,z=b['manifest'],c['manifest'];old,new=a['runtime_contract'],z['runtime_contract']
    require(old['source_git_commit']==B_HEAD and new['source_git_commit']==C_HEAD,'Unreviewed HEAD pair')
    require(a['experiment_id']==z['experiment_id']=='fl_capture_quality_v1','Wrong experiment')
    require(a['seed']==z['seed']==4001 and a['camera']==z['camera'],'Seed/camera differs')
    require(a['evaluation_configuration']==z['evaluation_configuration'],'Evaluation configuration differs')
    require(a['natural_reset_proof']['entry']==z['natural_reset_proof']['entry'],'Logical reset entry differs')
    mutable={'files','source_git_commit','runtime_content_sha256'}
    require({k:v for k,v in old.items() if k not in mutable}=={k:v for k,v in new.items() if k not in mutable},
            'Non-code runtime/physics/config/version contract differs')
    require(set(old['files'])==set(new['files']),'Runtime inventory changed')
    delta={p for p in old['files'] if old['files'][p]!=new['files'][p]}
    require(delta==CHANGED,'Receipt inventory delta is not exact reviewed six files')
    evidence=reviewed_code_delta()
    bindings={}
    for path in sorted(delta):
        _version_bytes(B_HEAD,path,old['files'][path]);_version_bytes(C_HEAD,path,new['files'][path])
        bindings[path]={'B_sha256':old['files'][path],'C_sha256':new['files'][path]}
    configs=old['selected_configuration']
    require(set(configs)==CONFIG_NAMES and configs==new['selected_configuration'],'Six selected configurations differ')
    for record in configs.values():
        path=record['path'];expected=record['sha256']
        require(old['files'].get(path)==new['files'].get(path)==expected,'Configuration inventory mismatch')
        require(_version_bytes(B_HEAD,path,expected)==_version_bytes(C_HEAD,path,expected),'Configuration bytes differ')
    scene={path:old['files'][path] for path in SCENE_PATHS}
    for path,expected in scene.items():
        require(new['files'][path]==expected,'Scene/control binding differs')
        require(_version_bytes(B_HEAD,path,expected)==_version_bytes(C_HEAD,path,expected),'Scene/control bytes differ')
    return {'B_actual_HEAD':B_HEAD,'C_actual_HEAD':C_HEAD,
      'B_runtime_content_sha256':old['runtime_content_sha256'],'C_runtime_content_sha256':new['runtime_content_sha256'],
      'same_selected_configuration':configs,'same_evaluation_configuration':a['evaluation_configuration'],
      'same_scene_construction_and_locked_inputs':scene,'same_camera':a['camera'],'same_seed':4001,
      'same_logical_reset_entry':a['natural_reset_proof']['entry'],'reviewed_code_scope':evidence,
      'changed_file_hashes':bindings,'B_rerun_at_C_HEAD':False,'strict_same_HEAD_pair':False,
      'identical_initial_measured_state_claimed':False,'bitwise_physical_trajectory_equivalence_claimed':False,
      'untracked_external_asset_identity_independently_remeasured':False,
      'disclosure':'Cross-HEAD descriptive comparison; unchanged N/control/locked scene paths, B not rerun at C HEAD'}


def render_cross_head(b,c,output,*,b_end,c_end,kind):
    require(1<b_end<=b['receipt']['frame_count'] and 1<c_end<=c['receipt']['frame_count'],'Bad event frame interval')
    count=max(b_end,c_end);require(count/15<=200,'Comparison exceeds200s')
    filters=[]
    for i,end in enumerate((b_end,c_end)):
        label=('B N+0 recorded at f2e5524' if i==0 else 'C PPO deterministic recorded at e73542c')
        lines=(label,'CROSS-HEAD | N control and locked scene/config unchanged',
          'B was NOT re-run at e73542c | not a same-build pair')
        chain=(f'[{i}:v]trim=start_frame=0:end_frame={end},setpts=PTS-STARTPTS,'
          'scale=960:652,pad=960:800:0:90:black,tpad=stop_mode=clone:stop=-1,setpts=N/(15*TB)')
        for y,line in zip((8,34,58),lines):
            chain+=(',drawtext=fontfile=\'C\\:/Windows/Fonts/arial.ttf\':'
              f"text='{line}':fontcolor=yellow:fontsize=19:x=12:y={y}")
        chain+=(',drawtext=fontfile=\'C\\:/Windows/Fonts/arial.ttf\':'
          "text='RUN WINDOW ENDED - FROZEN; NO PHYSICS / METRIC CREDIT':fontcolor=yellow:fontsize=18:x=12:y=765:"
          f"enable='gte(n,{end})'[v{i}]")
        filters.append(chain)
    filters.append('[v0][v1]hstack=inputs=2:shortest=1,format=yuv420p[out]')
    command=[str(base.find_ffmpeg()),'-hide_banner','-nostdin','-v','error','-n','-threads','2',
      '-filter_threads','1','-filter_complex_threads','1','-i',b['receipt']['output'],'-i',c['receipt']['output'],
      '-filter_complex',';'.join(filters),'-map','[out]','-an','-frames:v',str(count),'-r','15','-fps_mode','cfr',
      '-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-threads','2','-movflags','+faststart',str(output)]
    base.run(command)
    validation=review.review.validate_video(output,count,1920,800)
    return {'output':str(output),'kind':kind,'frame_count':count,'source_frame_intervals_half_open':[[0,b_end],[0,c_end]],
      'freeze_added_frames':[count-b_end,count-c_end],'freeze_is_physical_evidence':False,'freeze_included_in_metrics':False,
      'normal_speed':True,'speed_modified':False,'temporal_stitched':False,'same_elapsed_P01_origin':True,
      'phase_or_event_time_warping':False,'cross_HEAD_disclosure_visible_from_first_frame_and_every_panel':True,
      'synthetic_intro_time_added':False,'validation':validation,'command':command,
      'previews':base.previews(output,count,base.find_ffmpeg())}


def build(b_receipt,c_receipt,b_quality,c_quality,destination):
    b,c=pair.checked_review(b_receipt),pair.checked_review(c_receipt)
    common=cross_head_common(b,c)
    quality=pair._quality_pair(b_quality,c_quality,b,c)
    quality.update({'comparison_scope':'cross_HEAD_unchanged_N_descriptive_event_window',
      'same_HEAD_comparison':False,'frozen_padding_included':False,
      'better_policy_or_robustness_claim_from_one_pair':False})
    destination=pair._new_directory(destination)
    complete=b['receipt']['physical_task_success'] is True and c['receipt']['physical_task_success'] is True
    name='cross_HEAD_unchanged_N_full'+('' if complete else '_incomplete')+'.mp4'
    full=render_cross_head(b,c,destination/name,b_end=b['receipt']['frame_count'],c_end=c['receipt']['frame_count'],kind='full_episode_P01_origin')
    b_end=pair._event_frame(b['rows'],b['receipt']['physical_events']['placed']['FR'])+1
    c_end=pair._event_frame(c['rows'],c['receipt']['physical_events']['placed']['FR'])+1
    front=render_cross_head(b,c,destination/'cross_HEAD_unchanged_N_P01_to_FR_capture.mp4',b_end=b_end,c_end=c_end,kind='same_physical_FR_preparation_to_capture_event_each_side')
    result={'schema':'wlr50_clean.explicit_cross_HEAD_f2e_to_e735_unchanged_N_media_pair.v1',
      'B_review_receipt':str(b['receipt_path']),'C_review_receipt':str(c['receipt_path']),
      'cross_HEAD_control_path_evidence':common,'quality':quality,'full_episode':full,'front_event_window':front,
      'original_strict_pair_rule_used':False,'original_strict_pair_rule_modified':False,
      'source_receipts_modified':False,'B_rerun':False,'task_success_not_inferred_from_quality':True,
      'previous_cross_HEAD_helper_modified':False,'causal_policy_improvement_claimed':False,
      'outputs_helper_sha256':review.sha256(Path(__file__))}
    base.write_new_json(destination/'cross_HEAD_pair_receipt.json',result)
    return result


def review_code_only(output):
    """Source/locked-input proof only; no future C evidence is fabricated."""
    evidence=reviewed_code_delta()
    b_receipt_path=OUT/'B_control_review/B_control_Nplus0.media.json'
    b_receipt=json.loads(b_receipt_path.read_text(encoding='utf-8'))
    b_manifest=json.loads(Path(b_receipt['source_manifest']).read_text(encoding='utf-8'))
    runtime=b_manifest['runtime_contract']
    require(runtime['source_git_commit']==B_HEAD,'Retained B is not the reviewed source')
    selected=runtime['selected_configuration']
    require(set(selected)==CONFIG_NAMES,'Retained B selected config inventory differs')
    same={}
    for path in [*(row['path'] for row in selected.values()),*SCENE_PATHS]:
        expected=runtime['files'][path]
        require(_version_bytes(B_HEAD,path,expected)==_version_bytes(C_HEAD,path,expected),
                'Locked configuration or scene/control bytes differ')
        same[path]=expected
    result={'schema':'wlr50_clean.cross_HEAD_f2e_e735_source_only_review.v1',
      'reviewed_code_scope':evidence,'same_locked_configuration_scene_control_bytes':same,
      'B_actual_camera':b_manifest['camera'],'camera_production_path_unchanged':True,
      'future_C_actual_camera_not_yet_observed':True,
      'future_pair_requires_exact_sealed_receipt_camera_equality':True,
      'future_C_not_rendered_or_claimed_success':True,'new_B_run':False,
      'same_HEAD_pair':False,'initial_measured_state_identical_claimed':False,
      'physical_trajectory_equivalent_claimed':False,'causal_claim':False,
      'source_receipts_modified':False,'strict_pair_rules_modified':False,
      'helper_sha256':review.sha256(Path(__file__))}
    if output is not None:
        output=output.resolve()
        require(output.is_relative_to(OUT.resolve()) and not output.exists(),'Review proof must be new under output root')
        base.write_new_json(output,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review-code-only',action='store_true')
    p.add_argument('--review-proof-output',type=Path)
    for name in ('b-receipt','c-receipt','b-quality','c-quality','destination'):
        p.add_argument('--'+name,type=Path)
    args=p.parse_args()
    if args.review_code_only:
        require(all(getattr(args,name) is None for name in
            ('b_receipt','c_receipt','b_quality','c_quality','destination')),
            'Source-only review cannot mix rendering arguments')
        result=review_code_only(args.review_proof_output)
        print(json.dumps({'source_only_review':'PASS','HEADs':[B_HEAD,C_HEAD],
            'future_C_camera_receipt_required':True,'rendered':False,
            'output':None if args.review_proof_output is None else str(args.review_proof_output)},indent=2))
        return
    require(args.review_proof_output is None and all(getattr(args,name) is not None for name in
        ('b_receipt','c_receipt','b_quality','c_quality','destination')),
        'Rendering requires all five source/output arguments and no source-only output')
    result=build(args.b_receipt,args.c_receipt,args.b_quality,args.c_quality,args.destination)
    print(json.dumps({'receipt':str(args.destination/'cross_HEAD_pair_receipt.json'),
      'full':result['full_episode']['output'],'front':result['front_event_window']['output'],
      'same_HEAD_pair':False,'B_rerun':False},indent=2))


if __name__=='__main__':main()

