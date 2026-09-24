"""Stdlib-only read-only production patch renderer; no runtime imports or writes."""
import ast
import difflib
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
CODE='src/wlr50_clean/ppo/'


def replace(text,old,new,count=1):
    if text.count(old)!=count:raise ValueError(f'context {text.count(old)} != {count}: {old[:100]!r}')
    return text.replace(old,new)


def change(name,text):
    if name=='semantic_policy_distribution.py':
        text=replace(text,'LEGACY_POLICY =','from .semantic_p02_progress_profile import (P02_PROGRESS_POLICY, P02_PROGRESS_ACTOR_CLASS,\n    P02_PROGRESS_OBSERVATION_LAYOUT, p02_progress_policy_contract)\n\nLEGACY_POLICY =')
        text=replace(text,'    if version == REAR_POLICY_TIMING_POLICY:\n        return','    if version == P02_PROGRESS_POLICY:\n        return p02_progress_policy_contract(observation_layout=observation_layout)\n    if version == REAR_POLICY_TIMING_POLICY:\n        return')
        text=replace(text,'    if version == REAR_POLICY_TIMING_POLICY:\n        config','    if version == P02_PROGRESS_POLICY:\n        config["actor"]["class_name"] = P02_PROGRESS_ACTOR_CLASS\n        config["actor"]["exploration_std_temperature"] = contract["exploration_std_temperature"]\n    if version == REAR_POLICY_TIMING_POLICY:\n        config')
        text=replace(text,'    if isinstance(contract, Mapping):\n        if _same_json','    if isinstance(contract, Mapping):\n        if _same_json(dict(contract), policy_contract(P02_PROGRESS_POLICY, observation_layout=P02_PROGRESS_OBSERVATION_LAYOUT)):\n            return P02_PROGRESS_POLICY\n        if _same_json')
        text=replace(text,'            (REAR_POLICY_TIMING_ACTOR_CLASS,','            (P02_PROGRESS_ACTOR_CLASS, "HeteroscedasticGaussianDistribution", "log"): P02_PROGRESS_POLICY,\n            (REAR_POLICY_TIMING_ACTOR_CLASS,')
        text=replace(text,'                   REAR_POLICY_TIMING_POLICY) and semantic_version','                   REAR_POLICY_TIMING_POLICY, P02_PROGRESS_POLICY) and semantic_version')
    elif name=='semantic_observation.py':
        text=replace(text,'CONFIG_ROOT =','from .semantic_p02_progress_profile import (P02_PROGRESS_OBSERVATION_LAYOUT,\n    P02_PROGRESS_OBSERVATION_DIM, P02_PROGRESS_GROUP, p02_progress_features)\n\nCONFIG_ROOT =')
        text=replace(text,'    rear_policy_timing_features_version: str | None = None','    rear_policy_timing_features_version: str | None = None\n    p02_progress_features_version: str | None = None')
        text=replace(text,'return (self.rear_policy_timing_features_version','return (self.p02_progress_features_version or self.rear_policy_timing_features_version')
        text=replace(text,'    rr_groups = groups\n    if rear_timing_layout',"""    p02_layout = data.get('p02_progress_features_version')
    rear_groups = groups
    if p02_layout is not None:
        if (p02_layout != P02_PROGRESS_OBSERVATION_LAYOUT
                or rear_timing_layout != REAR_POLICY_TIMING_OBSERVATION_LAYOUT
                or groups[-1] != {'name':P02_PROGRESS_GROUP,'size':3,'scale':1.0}
                or sum(row['size'] for row in groups) != P02_PROGRESS_OBSERVATION_DIM):
            raise SemanticObservationError('P02 progress must preserve419 and append exactly three fields')
        rear_groups = groups[:-1]
    elif any(row['name'] == P02_PROGRESS_GROUP for row in groups):
        raise SemanticObservationError('P02 progress group requires its explicit version marker')
    rr_groups = rear_groups
    if rear_timing_layout""")
        text=replace(text,"or rr_layout != RR_CAPTURE_OBSERVATION_LAYOUT or groups[-1] != expected_rear_tail\n                or sum(row['size'] for row in groups) != REAR_POLICY_TIMING_OBSERVATION_DIM","or rr_layout != RR_CAPTURE_OBSERVATION_LAYOUT or rear_groups[-1] != expected_rear_tail\n                or sum(row['size'] for row in rear_groups) != REAR_POLICY_TIMING_OBSERVATION_DIM")
        text=replace(text,'        rr_groups = groups[:-1]','        rr_groups = rear_groups[:-1]')
        text=replace(text,'role_layout, capture_layout, rr_layout, rear_timing_layout)','role_layout, capture_layout, rr_layout, rear_timing_layout, p02_layout)')
        text=replace(text,'        self.schema.encode(groups)\n        mass',"""        if self.schema.p02_progress_features_version is not None:
            try:
                groups[P02_PROGRESS_GROUP] = p02_progress_features(field(task,'p02_progress_credit'),task['stage_id'])
            except ValueError as error:
                raise SemanticObservationError(str(error)) from error
        self.schema.encode(groups)
        mass""")
    elif name=='semantic_training.py':
        text=replace(text,'from __future__ import annotations','from __future__ import annotations\nfrom .semantic_p02_progress_profile import P02_PROGRESS_POLICY, P02_PROGRESS_OBSERVATION_LAYOUT')
        text=text.replace('REAR_POLICY_TIMING_POLICY)','REAR_POLICY_TIMING_POLICY, P02_PROGRESS_POLICY)')
        text=replace(text,'REAR_POLICY_TIMING_OBSERVATION_DIM):','REAR_POLICY_TIMING_OBSERVATION_DIM, 422):',count=4)
        text=replace(text,'    if policy_version == REAR_POLICY_TIMING_POLICY and','    if policy_version == P02_PROGRESS_POLICY and observation_layout != P02_PROGRESS_OBSERVATION_LAYOUT:\n        raise ValueError("P02 progress actor requires its explicit422 layout")\n    if policy_version == REAR_POLICY_TIMING_POLICY and')
        text=replace(text,'if getattr(runner, "_semantic_policy_version", None) == REAR_POLICY_TIMING_POLICY','if getattr(runner, "_semantic_policy_version", None) in (REAR_POLICY_TIMING_POLICY, P02_PROGRESS_POLICY)')
        text=replace(text,'    if migration is not None and migration.get("rear_live_swing_same419_factor") is not None:',"""    if migration is not None and migration.get("p02_progress_append_factor") is not None:
        if warm_start is not None or policy_migration is not None:
            raise ValueError("P02 append cannot mix another migration")
        from .semantic_p02_progress_migration import load_p02_progress_migration
        return load_p02_progress_migration(runner,checkpoint,contract=contract,seed=seed,record=migration)
    if migration is not None and migration.get("rear_live_swing_same419_factor") is not None:""")
        text=replace(text,'    if type(actor) not in (SemanticQuarterTemperedHistoryMLPModel,','    from .semantic_p02_progress_actor import SemanticP02ProgressHistoryMLPModel, p02_progress_effective_log_std\n    if type(actor) not in (SemanticQuarterTemperedHistoryMLPModel,')
        text=text.replace('SemanticRearPolicyTimingHistoryMLPModel):','SemanticRearPolicyTimingHistoryMLPModel, SemanticP02ProgressHistoryMLPModel):')
        text=replace(text,'    if type(actor) is SemanticRearPolicyTimingHistoryMLPModel:\n        effective_log_std, task_sigma_evidence = rear_policy_timing_effective_log_std(','    if type(actor) in (SemanticRearPolicyTimingHistoryMLPModel, SemanticP02ProgressHistoryMLPModel):\n        sigma_kernel = p02_progress_effective_log_std if type(actor) is SemanticP02ProgressHistoryMLPModel else rear_policy_timing_effective_log_std\n        effective_log_std, task_sigma_evidence = sigma_kernel(')
        text=replace(text,'    if type(actor) is SemanticRearPolicyTimingHistoryMLPModel:\n        record.update','    if type(actor) in (SemanticRearPolicyTimingHistoryMLPModel, SemanticP02ProgressHistoryMLPModel):\n        record.update')
        text=replace(text,'    return raw, record\n',"""    if type(actor) is SemanticP02ProgressHistoryMLPModel:
        record.update(schema='wlr50_clean.actual_p02_progress_request.v1',policy_version=P02_PROGRESS_POLICY,
            p02_progress_observed_features=vector(observation['policy'][...,419:422]),
            sigma_kernel_observation_slice=[0,419])
    return raw, record
""")
        text=replace(text,'for key in ("rear_live_swing_migration",','for key in ("p02_progress_migration", "rear_live_swing_migration",')
    elif name=='semantic_migration.py':
        text=replace(text,'    if supplied.get("schema") == "wlr50_clean.rear_live_swing_same419.v3":',"""    if supplied.get("schema") == "wlr50_clean.p02_progress_append.v1":
        from .semantic_p02_progress_migration import validate_p02_progress_migration
        return validate_p02_progress_migration(checkpoint,current_contract,plan_path,project_root=project_root)
    if supplied.get("schema") == "wlr50_clean.rear_live_swing_same419.v3":""")
        text=replace(text,'        if type(observation_layout) is not str or observation_layout not in (ROLE_OBSERVATION_LAYOUT,P05_CAPTURE_OBSERVATION_LAYOUT,RR_CAPTURE_OBSERVATION_LAYOUT,REAR_POLICY_TIMING_OBSERVATION_LAYOUT) or num_envs != 1:','        from .semantic_p02_progress_profile import P02_PROGRESS_OBSERVATION_LAYOUT\n        if type(observation_layout) is not str or observation_layout not in (ROLE_OBSERVATION_LAYOUT,P05_CAPTURE_OBSERVATION_LAYOUT,RR_CAPTURE_OBSERVATION_LAYOUT,REAR_POLICY_TIMING_OBSERVATION_LAYOUT,P02_PROGRESS_OBSERVATION_LAYOUT) or num_envs != 1:')
        text=replace(text,'result.update(observation_layout=observation_layout, observation_dimension=(REAR_POLICY_TIMING_OBSERVATION_DIM','result.update(observation_layout=observation_layout, observation_dimension=(422\n            if observation_layout == P02_PROGRESS_OBSERVATION_LAYOUT else REAR_POLICY_TIMING_OBSERVATION_DIM')
    elif name=='semantic_rear_policy_timing_migration.py':
        text=replace(text,'    receipt=metadata.get(MIGRATION,{})\n    runtime_bound',"""    if metadata.get('p02_progress_migration') is not None:
        from .semantic_p02_progress_migration import validate_p02_progress_lineage
        validate_p02_progress_lineage(metadata,contract,output_root,checkpoint_output_routing=checkpoint_output_routing)
        return
    receipt=metadata.get(MIGRATION,{})
    runtime_bound""")
    elif name=='semantic_cli.py':
        text=replace(text,'from __future__ import annotations','from __future__ import annotations\nfrom .semantic_p02_progress_profile import P02_PROGRESS_POLICY, P02_PROGRESS_OBSERVATION_LAYOUT')
        text=replace(text,'            if sum((initial_append,same419,live_swing)) != 1:','            p02_append = isinstance(planned.get("p02_progress_append_factor"),dict)\n            if sum((initial_append,same419,live_swing,p02_append)) != 1:')
        text=replace(text,'            elif live_swing:\n                if',"""            elif p02_append:
                if (planned.get('schema') != 'wlr50_clean.p02_progress_append.v1'
                        or not branch_requested or source_root.resolve()!=checkpoint_output.resolve()):
                    raise ValueError('P02 append requires the existing learned branch')
            elif live_swing:
                if""")
        text=replace(text,'        if args._observation_layout != REAR_POLICY_TIMING_OBSERVATION_LAYOUT or metadata["seed"] != args.seed:',"""        if args._migration_record.get('p02_progress_append_factor') is not None:
            if args._observation_layout != P02_PROGRESS_OBSERVATION_LAYOUT or metadata['seed'] != args.seed:
                raise ValueError('P02 append requires exact422 and original seed')
            args._policy_version=P02_PROGRESS_POLICY
            return
        if args._observation_layout != REAR_POLICY_TIMING_OBSERVATION_LAYOUT or metadata["seed"] != args.seed:""")
        text=text.replace('(P05_CAPTURE_POLICY,RR_CAPTURE_POLICY,REAR_POLICY_TIMING_POLICY):','(P05_CAPTURE_POLICY,RR_CAPTURE_POLICY,REAR_POLICY_TIMING_POLICY,P02_PROGRESS_POLICY):')
    elif name=='semantic_checkpoint_prefix_policy.py':
        text=replace(text,'from .semantic_rear_policy_timing_profile import REAR_POLICY_TIMING_POLICY','from .semantic_p02_progress_profile import P02_PROGRESS_POLICY\nfrom .semantic_p02_progress_actor import SemanticP02ProgressHistoryMLPModel\nfrom .semantic_rear_policy_timing_profile import REAR_POLICY_TIMING_POLICY')
        text=replace(text,'(P05_CAPTURE_POLICY,RR_CAPTURE_POLICY,REAR_POLICY_TIMING_POLICY):','(P05_CAPTURE_POLICY,RR_CAPTURE_POLICY,REAR_POLICY_TIMING_POLICY,P02_PROGRESS_POLICY):')
        text=replace(text,'REAR_POLICY_TIMING_POLICY:SemanticRearPolicyTimingHistoryMLPModel}','REAR_POLICY_TIMING_POLICY:SemanticRearPolicyTimingHistoryMLPModel,\n            P02_PROGRESS_POLICY:SemanticP02ProgressHistoryMLPModel}')
    elif name=='semantic_checkpoint_prefix.py':
        text=replace(text,'from .semantic_rear_policy_timing_profile import','from .semantic_p02_progress_profile import P02_PROGRESS_OBSERVATION_LAYOUT\nfrom .semantic_rear_policy_timing_profile import')
        text=replace(text,'{"observation_dimension": REAR_POLICY_TIMING_OBSERVATION_DIM,"observation_layout": REAR_POLICY_TIMING_OBSERVATION_LAYOUT,"action_dimension":12}):','{"observation_dimension": REAR_POLICY_TIMING_OBSERVATION_DIM,"observation_layout": REAR_POLICY_TIMING_OBSERVATION_LAYOUT,"action_dimension":12},\n                {"observation_dimension":422,"observation_layout":P02_PROGRESS_OBSERVATION_LAYOUT,"action_dimension":12}):')
        text=replace(text,'or (dimension == REAR_POLICY_TIMING_OBSERVATION_DIM and layout == REAR_POLICY_TIMING_OBSERVATION_LAYOUT))):','or (dimension == REAR_POLICY_TIMING_OBSERVATION_DIM and layout == REAR_POLICY_TIMING_OBSERVATION_LAYOUT)\n                    or (dimension == 422 and layout == P02_PROGRESS_OBSERVATION_LAYOUT))):')
    else:raise ValueError(name)
    ast.parse(text,filename=name)
    return text


def render():
    pieces=[]
    names=('semantic_policy_distribution.py','semantic_observation.py','semantic_training.py',
        'semantic_migration.py','semantic_rear_policy_timing_migration.py','semantic_cli.py',
        'semantic_checkpoint_prefix_policy.py','semantic_checkpoint_prefix.py')
    for name in names:
        path=CODE+name;before=(ROOT/path).read_text(encoding='utf-8');after=change(name,before)
        pieces.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+path,tofile='b/'+path))
    for name in ('semantic_p02_progress_profile.py','semantic_p02_progress_actor.py','semantic_p02_progress_migration.py'):
        text=(HERE/name).read_text(encoding='utf-8');ast.parse(text,filename=name)
        pieces.extend(difflib.unified_diff([],text.splitlines(True),fromfile='/dev/null',tofile='b/'+CODE+name))
    return ''.join(pieces)


if __name__=='__main__':print(render(),end='')
