"""Read-only proof of the one authorized finite FL-row auxiliary lineage."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parents[2]
ROOT = OUT.parents[1]
HELPERS = OUT/'candidate/finite_auxiliary'
APPROVED_HELPERS = {
    'finite_auxiliary_mean.py':'f0e12560681f5427705ac76602c12faeb28005ef959984418c710a9552d2a65d',
    'reviewed_data.py':'78a9cd09ec5d0ad9ace99ffa1a82c828baaaf5fb7fe404c960c848d17950ad52',
    'auxiliary_cli.py':'bffb60e020e853ed0a8ee63e53571bb270d01858cfeeccc671e91b72c7eed7e7'}
BRANCH = 'task_conditioned_hip_wheel_branch'
LEDGER = 'auxiliary_mean_learning'
COUNTERS = ('global_policy_decisions','ppo_updates','optimizer_steps')
SCOPE = 'FL_AIR_gap_approach_not_capture'
BUDGET = {'max_steps':16,'learning_rate':.05,'maximum_train_request_shift_deg':1.,
          'maximum_holdout_request_shift_deg':.25,'maximum_per_state_conditional_kl':.1}


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def reject_unknown_auxiliary(value, path=(), *, allow_ledger=True):
    if isinstance(value,dict):
        for key,item in value.items():
            location=path+(key,);name=str(key).lower()
            if allow_ledger and location==(BRANCH,LEDGER):continue  # Fully checked below.
            require(not (name=='aux' or name.startswith('aux_') or 'auxiliary' in name),
                    'unreviewed auxiliary provenance outside the one validated ledger')
            reject_unknown_auxiliary(item,location,allow_ledger=allow_ledger)
    elif isinstance(value,(list,tuple)):
        for i,item in enumerate(value):reject_unknown_auxiliary(item,path+(i,),allow_ledger=allow_ledger)


def checked_helpers():
    require({n:sha(HELPERS/n) for n in APPROVED_HELPERS} == APPROVED_HELPERS,
            'sealed auxiliary helper bundle changed; dedicated media re-review required')
    sys.path.insert(0,str(HELPERS));sys.path.insert(0,str(ROOT/'src'))
    spec=importlib.util.spec_from_file_location('media_readonly_aux_cli',HELPERS/'auxiliary_cli.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    require(Path(module.a.__file__).resolve()==(HELPERS/'finite_auxiliary_mean.py').resolve(),
            'auxiliary helper import was shadowed')
    require(Path(sys.modules['reviewed_data'].__file__).resolve()==(HELPERS/'reviewed_data.py').resolve(),
            'reviewed-data helper import was shadowed')
    return module


def checkpoint(path):
    import torch
    from wlr50_clean.ppo import semantic_migration as m, semantic_training as t
    path=Path(path).resolve(strict=True)
    require(path.is_relative_to((OUT/'checkpoints/history').resolve()),'checkpoint outside immutable task history')
    meta=m.checkpoint_metadata(path);payload=torch.load(path,map_location='cpu',weights_only=False)
    reject_unknown_auxiliary(meta)
    require(payload['infos']=={k:v for k,v in meta.items() if k not in
        ('checkpoint_path','checkpoint_sha256','save_load_round_trip')},'checkpoint embedded infos differ from sidecar')
    for role in ('actor','critic'):
        h=hashlib.sha256()
        for name,value in sorted(payload[role+'_state_dict'].items()):
            require(torch.isfinite(value).all().item(),'non-finite actual '+role+' tensor')
            v=value.detach().cpu().contiguous();h.update(name.encode());h.update(str(v.dtype).encode())
            h.update(str(tuple(v.shape)).encode());h.update(v.numpy().tobytes())
        require(h.hexdigest()==meta[role+'_parameter_sha256'],'actual '+role+' tensor hash differs')
    require(t.state_hash(payload['optimizer_state_dict'])==meta['optimizer_state_sha256'],'actual Adam hash differs')
    return meta,payload


def validate_fit(report):
    allowed_aux={'auxiliary_optimizer','accepted_auxiliary_updates','attempted_auxiliary_optimizer_steps'}
    audit={k:v for k,v in report.items() if k not in allowed_aux}
    for side in ('before','after'):
        audit[side]=dict(audit.get(side,{}))
        if 'default_budget_linear_prediction' in audit[side]:
            prediction=dict(audit[side]['default_budget_linear_prediction'])
            require(type(prediction.get('auxiliary_updates_added')) is int and prediction['auxiliary_updates_added']==0,
                    'read-only linear prediction adds auxiliary updates')
            prediction.pop('auxiliary_updates_added');audit[side]['default_budget_linear_prediction']=prediction
    reject_unknown_auxiliary(audit,allow_ledger=False)
    require(report.get('schema')=='wlr50_clean.finite_FL_conditional_mean_aux.v1'
        and report.get('label_scope')==SCOPE and report.get('budget')==BUDGET,
        'unreviewed auxiliary report schema/scope/budget')
    require(report.get('optimized_parameters')==['actor.mlp.4.weight[0,:]','actor.mlp.4.bias[0]']
        and report.get('optimized_scalar_count')==257
        and report.get('auxiliary_optimizer')=='independent two-leaf SGD; momentum=0; weight_decay=0',
        'auxiliary optimizer or allowed FL row differs')
    for key in ('PPO_decisions_added','PPO_updates_added','PPO_optimizer_steps_added'):
        require(type(report.get(key)) is int and report[key]==0,'auxiliary report adds PPO credit')
    for key in ('MDP_or_control_changed','kernel_or_sigma_changed','teacher_deployed','physical_success_claimed'):
        require(report.get(key) is False,'auxiliary report changes control or claims physical success')
    require(report.get('training_rng_preserved') is True and report.get('fresh_PPO_rollout_required') is True,
            'auxiliary report lacks preserved RNG/fresh rollout contract')
    steps=report.get('steps',[]);accepted=report.get('accepted_auxiliary_updates');attempted=report.get('attempted_auxiliary_optimizer_steps')
    require(type(accepted) is int and type(attempted) is int and 1<=accepted<=attempted<=16
        and len(steps)==attempted and sum(x.get('accepted') is True for x in steps)==accepted,
        'auxiliary step counts do not match the actual finite step report')
    for i,row in enumerate(steps):
        values=[row.get(k) for k in ('learning_rate','loss_before','loss_after_candidate','max_train_request_shift_deg',
            'max_holdout_request_shift_deg','maximum_per_state_conditional_kl','selected_gradient_l2')]
        require(all(type(x) in (int,float) and math.isfinite(x) and x>=0 for x in values)
            and row.get('attempt')==i+1 and type(row.get('accepted')) is bool
            and math.isclose(row['learning_rate'],.05*(16-i)/16,rel_tol=0,abs_tol=1e-14),
            'auxiliary step values/order/linear learning rate differ')
        if row['accepted']:
            require(row['loss_after_candidate']<=row['loss_before'] and row['max_train_request_shift_deg']<=1.
                and row['max_holdout_request_shift_deg']<=.25 and row['maximum_per_state_conditional_kl']<=.1,
                'accepted auxiliary step violates its loss/trust bounds')
        else:require(i==len(steps)-1,'auxiliary learning continued after a rejected step')
    stop=report.get('stop_reason')
    require((stop=='finite_budget_exhausted' and attempted==accepted==16)
        or (stop=='candidate_step_rejected_by_finite_loss_or_trust_bound' and attempted==accepted+1 and steps[-1]['accepted'] is False)
        or (stop=='conditional_mean_already_matches_recorded_local_target' and attempted==accepted
            and steps[-1]['loss_after_candidate']<1e-12), 'auxiliary finite stopping reason contradicts its steps')
    return accepted,attempted


def validate_auxiliary_checkpoint(path, auxiliary_receipt):
    """No policy forward, sample, fit, optimizer step or simulator invocation."""
    import torch
    helpers=checked_helpers()
    from wlr50_clean.ppo import semantic_migration as m
    cache={}
    def cp(p):
        key=Path(p).resolve(strict=True)
        if key not in cache:cache[key]=checkpoint(key)
        return cache[key]
    current_path=Path(path).resolve(strict=True);current,_=cp(current_path)
    require(current['runtime_contract'].get('experiment_id')=='task_conditioned_hip_wheel_v1'
        and current['policy_contract'].get('version')=='task_conditioned_hip_wheel_sigma_v1'
        and current['policy_contract'].get('observation_dimension')==372
        and current['policy_contract'].get('raw_action_dimension')==12 and m.source_num_envs(current)==1,
        'auxiliary media supports only the current same372/full12/N1 task policy')
    ledger=current.get(BRANCH,{}).get(LEDGER)
    require(isinstance(ledger,dict) and ledger.get('schema')=='wlr50_clean.auxiliary_mean_learning_ledger.v1'
        and ledger.get('training_lineage_label')=='PPO_plus_explicit_finite_auxiliary_mean_supervision'
        and isinstance(ledger.get('events'),list) and len(ledger['events'])==1,
        'this adapter supports exactly one explicit reviewed auxiliary event')
    require(set(ledger)=={'schema','events','training_lineage_label','accepted_auxiliary_updates_total',
        'attempted_auxiliary_optimizer_steps_total'},'unreviewed auxiliary ledger fields')
    event=ledger['events'][0];fit=event['report'];accepted,attempted=validate_fit(fit)
    require(set(event)=={'event_index','kind','source_checkpoint','helper_sha256','data_receipt','report',
        'PPO_counters_unchanged','stage_requested_decisions_unchanged'},'unreviewed auxiliary event fields')
    require(type(event.get('event_index')) is int and event['event_index']==1
        and event.get('kind')=='limited_supervised_FL_conditional_mean_not_PPO'
        and ledger.get('accepted_auxiliary_updates_total')==accepted
        and ledger.get('attempted_auxiliary_optimizer_steps_total')==attempted
        and type(ledger.get('accepted_auxiliary_updates_total')) is int
        and type(ledger.get('attempted_auxiliary_optimizer_steps_total')) is int,'auxiliary ledger totals/kind differ')
    receipt_path=Path(auxiliary_receipt).resolve(strict=True);execution=read(receipt_path)
    require(execution.get('schema')=='wlr50_clean.finite_auxiliary_candidate_receipt.v1'
        and execution.get('mode')=='explicit_finite_auxiliary_not_PPO' and execution.get('fit')==fit
        and execution.get('auxiliary_updates')==accepted and execution.get('data_receipt')==event['data_receipt'],
        'external execution report is not the embedded auxiliary event')
    require(all(execution.get(k)==0 and type(execution.get(k)) is int for k in
        ('PPO_decisions_added','PPO_updates_added','PPO_optimizer_steps_added'))
        and all(execution.get(k) is False for k in ('teacher_deployed','capture_claimed','physical_success_claimed')),
        'auxiliary execution claims PPO credit, teacher deployment or physical success')
    source_binding=event['source_checkpoint'];source,source_payload=cp(source_binding['path'])
    reject_unknown_auxiliary(source,allow_ledger=False)
    require(source_binding==execution['binding']['source_checkpoint']
        and source_binding['sha256']==source['checkpoint_sha256']
        and source_binding['manifest_sha256']==sha(Path(source_binding['path']).with_name(Path(source_binding['path']).stem+'_manifest.json')),
        'source checkpoint/manifest hash differs')
    require(LEDGER not in source.get(BRANCH,{}) and source['runtime_contract']==current['runtime_contract']
        and source['policy_contract']==current['policy_contract'] and source['runner_config']==current['runner_config'],
        'auxiliary source is not first same-runtime/policy/runner boundary')
    require(event.get('helper_sha256')==APPROVED_HELPERS['finite_auxiliary_mean.py']
        and execution['binding']['helper_sha256']==APPROVED_HELPERS
        and event['data_receipt']['helper_bundle_sha256']==APPROVED_HELPERS,
        'auxiliary event does not bind all three sealed helpers')
    data=event['data_receipt'];_,_,_,rechecked=helpers.load_reviewed_selection(data['selection_path'])
    require(rechecked=={k:v for k,v in data.items() if k!='helper_bundle_sha256'}
        and execution['binding']['selection_sha256']==data['selection_sha256']
        and execution['binding']['runtime_contract_sha256']==m.digest(current['runtime_contract']),
        'actual reviewed data/selection/runtime binding differs')
    require(helpers.verify_data_compatibility(source,data,current['runtime_contract'])==execution['data_runtime_compatibility'],
            'diagnostic data crosses an unverified runtime/MDP boundary')
    inspection=execution['reviewed_inspection'];prior=read(inspection['path'])
    require(sha(inspection['path'])==inspection['sha256'] and prior.get('binding')==execution['binding']
        and prior.get('mode')=='read_only_current_checkpoint_on_real_saved_states' and prior.get('auxiliary_updates')==0,
        'reviewed pre-execution inspection is missing/stale or already optimized')
    aux_binding=execution['auxiliary_checkpoint'];aux_path=Path(aux_binding['path']).resolve(strict=True);aux,aux_payload=cp(aux_path)
    require(aux_path.name.startswith('checkpoint_aux_flmean_') and aux['stage']=='auxiliary_mean_pretrain_not_PPO'
        and aux_binding['sha256']==aux['checkpoint_sha256']
        and Path(aux_binding['manifest']).resolve()==aux_path.with_name(aux_path.stem+'_manifest.json')
        and aux.get(BRANCH,{}).get(LEDGER)==ledger,'execution report is not the immediate auxiliary checkpoint')
    require(aux['resume_ancestry'].get('operation')=='explicit_auxiliary_mean_supervision_not_PPO'
        and aux['resume_ancestry']['source_checkpoint']==source_binding
        and aux['resume_ancestry']['source_actor_parameter_sha256']==source['actor_parameter_sha256']
        and aux['resume_ancestry']['source_runtime_contract']==source['runtime_contract']
        and all(aux['resume_ancestry']['source_'+k]==source[k] for k in COUNTERS),'immediate auxiliary ancestry differs')
    require(aux.get('resume_source_checkpoint')=={'checkpoint':str(Path(source_binding['path']).resolve()),
        'checkpoint_sha256':source_binding['sha256'],
        'manifest':str(Path(source_binding['path']).resolve().with_name(Path(source_binding['path']).stem+'_manifest.json')),
        'manifest_sha256':source_binding['manifest_sha256']},'immediate auxiliary official-load source receipt differs')
    require(event['PPO_counters_unchanged']==execution['original_PPO_counters']=={k:source[k] for k in COUNTERS}
        and all(aux[k]==source[k] for k in COUNTERS)
        and event['stage_requested_decisions_unchanged']==aux['stage_requested_decisions']==source['stage_requested_decisions'],
        'auxiliary operation changes lifetime PPO counters or stage spending')
    for key in ('runtime_contract','policy_contract','runner_config','optimizer_state_sha256','normalizer_state_sha256',
                'critic_parameter_sha256','optimizer_learning_rate','training_rng_state'):
        require(aux[key]==source[key],'auxiliary operation changes protected source state: '+key)
    require(fit['actor_parameter_sha256_before']==source['actor_parameter_sha256']
        and fit['actor_parameter_sha256_after']==aux['actor_parameter_sha256']
        and fit['PPO_Adam_preserved_sha256']==source['optimizer_state_sha256']
        and fit['PPO_LR_preserved']==source['optimizer_learning_rate'],'fit/source/aux state hashes disagree')
    changed=False
    require(source_payload['actor_state_dict'].keys()==aux_payload['actor_state_dict'].keys()
        and tuple(source_payload['actor_state_dict']['mlp.4.weight'].shape)==(24,256)
        and tuple(source_payload['actor_state_dict']['mlp.4.bias'].shape)==(24,),
        'actual auxiliary source/target head architecture differs')
    for key,before in source_payload['actor_state_dict'].items():
        after=aux_payload['actor_state_dict'][key]
        require(torch.equal(before[1:],after[1:]) if key in ('mlp.4.weight','mlp.4.bias') else torch.equal(before,after),
                'actual auxiliary tensor changed beyond FL mean row0')
        changed=changed or not torch.equal(before,after)
    require(changed,'auxiliary report claims positive updates but actor did not change')
    lineage=[];cursor=current_path
    for _ in range(64):
        meta,_=cp(cursor)
        require(meta[BRANCH].get(LEDGER)==ledger and {k:v for k,v in meta[BRANCH].items() if k!=LEDGER}==source[BRANCH]
            and meta['runtime_contract']==source['runtime_contract'] and meta['policy_contract']==source['policy_contract']
            and meta['runner_config']==source['runner_config'],'auxiliary lineage/branch/runtime was changed or stripped')
        require(meta['task_conditioned_hip_wheel_branch_counts']=={k:meta[k]-meta[BRANCH]['counter_origin'][k] for k in COUNTERS},
                'PPO branch counters include auxiliary updates')
        lineage.append({'checkpoint':str(cursor),'sha256':meta['checkpoint_sha256']})
        if cursor==aux_path:break
        ancestry=meta.get('resume_ancestry',{});binding=ancestry['source_checkpoint']
        parent_path=Path(binding['checkpoint']).resolve(strict=True);parent,_=cp(parent_path)
        require(parent_path not in [Path(x['checkpoint']) for x in lineage]
            and binding['checkpoint_sha256']==parent['checkpoint_sha256']
            and Path(binding['manifest']).resolve()==parent_path.with_name(parent_path.stem+'_manifest.json')
            and binding['manifest_sha256']==sha(binding['manifest'])
            and ancestry['source_actor_parameter_sha256']==parent['actor_parameter_sha256']
            and ancestry['source_runtime_contract']==parent['runtime_contract']
            and all(ancestry['source_'+k]==parent[k] for k in COUNTERS), 'fresh PPO ancestry is not the actual preceding checkpoint')
        du=meta['ppo_updates']-parent['ppo_updates']
        require(du>0 and meta['global_policy_decisions']-parent['global_policy_decisions']==128*du
            and meta['optimizer_steps']-parent['optimizer_steps']==20*du
            and all(meta['stage_requested_decisions'][k]>=v for k,v in parent['stage_requested_decisions'].items()),
            'successor counters are not independent fresh N1 PPO updates')
        cursor=parent_path
    else:raise ValueError('auxiliary lineage exceeds the bounded 64-checkpoint audit')
    return {'method':'PPO_PLUS_LIMITED_AUX','display_label':'PPO + LIMITED AUX','label_scope':SCOPE,
        'accepted_auxiliary_updates_total':accepted,'attempted_auxiliary_optimizer_steps_total':attempted,
        'PPO_counters':{k:current[k] for k in COUNTERS},'auxiliary_PPO_credit':0,
        'execution_receipt':str(receipt_path),'execution_receipt_sha256':sha(receipt_path),
        'helper_bundle_sha256':APPROVED_HELPERS,'selection_sha256':data['selection_sha256'],
        'source_checkpoint':source_binding,'immediate_auxiliary_checkpoint':aux_binding,'verified_lineage':lineage,
        'teacher_deployed':False,'auxiliary_learning_does_not_prove_capture_or_task_success':True}
