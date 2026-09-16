// Output-only, exact-source transformations; never writes the production tree.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url)),root=path.resolve(here,'../..');
const candidate=path.join(here,'temperature_loader_candidate');
const changes=[];
function replace(s,a,b){assert.equal(s.split(a).length,2,`Expected one exact source region: ${a.slice(0,80)}`);return s.replace(a,()=>b);}
function emit(rel,source){const target=path.join(candidate,rel);fs.mkdirSync(path.dirname(target),{recursive:true});fs.writeFileSync(target,source);changes.push(rel);}
let s=fs.readFileSync(path.join(root,'src/wlr50_clean/ppo/semantic_training.py'),'utf8').replaceAll('\r\n','\n');
s=replace(s,`    from .semantic_policy_distribution import HISTORY_POLICY
    if policy_version == HISTORY_POLICY and semantic_version != "v3":
        raise ValueError("history-conditioned policy requires the v3 semantic runtime")`,
`    from .semantic_policy_distribution import HISTORY_POLICY, HISTORY_TEMPERED_POLICY
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    if policy_version in (HISTORY_POLICY, HISTORY_TEMPERED_POLICY) and semantic_version != "v3":
        raise ValueError("history-conditioned policy requires the v3 semantic runtime")
    if policy_version == HISTORY_TEMPERED_POLICY and observation_layout != ROLE_OBSERVATION_LAYOUT:
        raise ValueError("tempered history policy requires the explicit role372 observation layout")`);
const helper=`def _validated_exploration_temperature_factor(metadata: Mapping[str, Any], record: Mapping[str, Any], *,
        semantic_version: str, seed: int, device: str, observation_layout: str | None) -> Mapping[str, Any] | None:
    """Validate only the reviewed same372 distribution boundary, after plan revalidation."""
    factor = record.get("exploration_temperature_factor")
    if factor is None:
        return None
    from .semantic_migration import source_num_envs
    from .semantic_policy_distribution import HISTORY_POLICY, HISTORY_TEMPERED_POLICY, policy_version_from_metadata
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    exclusive = ("execution_factor", "instrumentation_observation_contract", "video_instrumentation_factor",
        "nominal_timing_factor", "body_reward_factor", "height_recovery_factor")
    if any(record.get(key) is not None for key in exclusive):
        raise RuntimeError("exploration temperature migration cannot mix other migration factors")
    if (not isinstance(factor, Mapping) or semantic_version != "v3"
            or metadata.get("semantic_version") != "v3" or seed != metadata.get("seed")
            or observation_layout != ROLE_OBSERVATION_LAYOUT or source_num_envs(metadata) != 1
            or metadata.get("runner_config", {}).get("device") != device
            or policy_version_from_metadata(metadata) != HISTORY_POLICY):
        raise RuntimeError("exploration temperature migration requires the exact v3 role372 N1 source")
    source_policy = policy_contract(HISTORY_POLICY, observation_layout=observation_layout)
    target_policy = policy_contract(HISTORY_TEMPERED_POLICY, observation_layout=observation_layout)
    observation = {"source_policy_contract": source_policy, "target_policy_contract": target_policy,
        "observation_layout": observation_layout, "observation_dimension": 372, "action_dimension": 12,
        "num_envs": 1, "parameter_mapping": "identity_all_parameters_and_buffers"}
    if (metadata.get("policy_contract") != source_policy
            or factor.get("source_policy_contract") != source_policy
            or factor.get("target_policy_contract") != target_policy
            or factor.get("source_policy_version") != source_policy["version"]
            or factor.get("target_policy_version") != target_policy["version"]
            or factor.get("observation_contract") != observation):
        raise RuntimeError("exploration temperature migration lacks exact source/target policy and observation contracts")
    horizon = runner_return_profile(metadata["runner_config"], semantic_version="v3")["version"]
    configs = {version: semantic_runner_config(seed=seed, device=device, semantic_version="v3",
        policy_version=version, observation_layout=observation_layout, return_profile=horizon)
        for version in (HISTORY_POLICY, HISTORY_TEMPERED_POLICY)}
    if (metadata["runner_config"] != configs[HISTORY_POLICY]
            or factor.get("source_runner_config") != configs[HISTORY_POLICY]
            or factor.get("target_runner_config") != configs[HISTORY_TEMPERED_POLICY]):
        raise RuntimeError("exploration temperature migration runner configuration differs from its exact contracts")
    return factor


`;
s=replace(s,'def load_semantic_checkpoint(',helper+'def load_semantic_checkpoint(');
s=replace(s,`    from .semantic_policy_distribution import policy_version_from_metadata
    if policy_version_from_metadata(metadata) != runner._semantic_policy_version:
        raise RuntimeError("checkpoint policy distribution differs from the constructed actor")
    if runner.alg.storage.observations["policy"].shape[-1] in (324, 372):
        if metadata.get("policy_contract") != _runner_policy_contract(runner):
            raise RuntimeError("checkpoint observation layout differs; explicit append migration is required")`,
`    from .semantic_policy_distribution import policy_version_from_metadata
    verified = None
    if migration is not None and migration.get("exploration_temperature_factor") is not None:
        from .semantic_migration import validate_migration_plan
        verified = validate_migration_plan(checkpoint, contract, Path(migration["plan_path"]))
        if verified != dict(migration):
            raise RuntimeError("migration changed since pre-AppLauncher validation")
    temperature = _validated_exploration_temperature_factor(metadata, verified or {},
        semantic_version=runner._semantic_version, seed=seed, device=str(runner.device),
        observation_layout=getattr(runner, "_semantic_observation_layout", None))
    if temperature is not None:
        if (runner._semantic_policy_version != temperature["target_policy_contract"]["version"]
                or _runner_policy_contract(runner) != temperature["target_policy_contract"]
                or runner._semantic_runner_config != temperature["target_runner_config"]):
            raise RuntimeError("constructed tempered actor differs from the verified target configuration")
    else:
        if policy_version_from_metadata(metadata) != runner._semantic_policy_version:
            raise RuntimeError("checkpoint policy distribution differs from the constructed actor")
        if runner.alg.storage.observations["policy"].shape[-1] in (324, 372):
            if metadata.get("policy_contract") != _runner_policy_contract(runner):
                raise RuntimeError("checkpoint observation layout differs; explicit append migration is required")`);
s=replace(s,`    if migration is not None:
        from .semantic_migration import validate_migration_plan
        verified = validate_migration_plan(checkpoint, contract, Path(migration["plan_path"]))
        if verified != dict(migration):
            raise RuntimeError("migration changed since pre-AppLauncher validation")
        expected_contract = metadata["runtime_contract"]`,
`    if migration is not None:
        if verified is None:
            from .semantic_migration import validate_migration_plan
            verified = validate_migration_plan(checkpoint, contract, Path(migration["plan_path"]))
            if verified != dict(migration):
                raise RuntimeError("migration changed since pre-AppLauncher validation")
        expected_contract = metadata["runtime_contract"]`);
s=replace(s,`        if sum(x is not None for x in (video_factor, timing_factor, body_reward_factor, height_factor)) > 1:
            raise RuntimeError("video, nominal timing, body reward and height migration receipts must be exclusive")
        reviewed_factor = video_factor or timing_factor or body_reward_factor or height_factor`,
`        temperature_factor = None if temperature is None else temperature["observation_contract"]
        if sum(x is not None for x in (video_factor, timing_factor, body_reward_factor, height_factor, temperature_factor)) > 1:
            raise RuntimeError("video, nominal timing, body reward, height and temperature migration receipts must be exclusive")
        reviewed_factor = video_factor or timing_factor or body_reward_factor or height_factor or temperature_factor`);
s=replace(s,`        if metadata.get("runner_config") != semantic_runner_config(seed=seed, device=str(runner.device),
                semantic_version=metadata.get("semantic_version", "v2"),
                policy_version=runner._semantic_policy_version, observation_layout=layout):`,
`        if temperature is None and metadata.get("runner_config") != semantic_runner_config(seed=seed, device=str(runner.device),
                semantic_version=metadata.get("semantic_version", "v2"),
                policy_version=runner._semantic_policy_version, observation_layout=layout):`);
emit('src/wlr50_clean/ppo/semantic_training.py',s);
s=fs.readFileSync(path.join(root,'src/wlr50_clean/ppo/semantic_cli.py'),'utf8').replaceAll('\r\n','\n');
s=replace(s,`        args._migration_record = validate_migration_plan(args.checkpoint, contract, args.resume_migration)
        if metadata.get("runner_config") != semantic_runner_config(seed=int(metadata["seed"]), device=args.device,`,
`        args._migration_record = validate_migration_plan(args.checkpoint, contract, args.resume_migration)
        from .semantic_training import _validated_exploration_temperature_factor
        temperature = _validated_exploration_temperature_factor(metadata, args._migration_record,
            semantic_version=args.semantic_version, seed=int(metadata["seed"]), device=args.device,
            observation_layout=args._observation_layout)
        if temperature is not None:
            if getattr(args, "num_envs", 1) != 1:
                raise ValueError("exploration temperature migration requires N1")
            args._policy_version = temperature["target_policy_contract"]["version"]
        elif metadata.get("runner_config") != semantic_runner_config(seed=int(metadata["seed"]), device=args.device,`);
s=replace(s,`        policy_contract(resolved)  # Reject an unsupported internal selection.`,
`        policy_contract(resolved, observation_layout=_resolved_observation_layout(args))
        # Reject an unsupported internal selection, including a missing role layout.`);
emit('src/wlr50_clean/ppo/semantic_cli.py',s);
const test='tests/unit/test_semantic_temperature_loader_cli.py';
emit(test,fs.readFileSync(path.join(here,'temperature_loader_cli_test_source.py'),'utf8'));
const parts=[],applyParts=['*** Begin Patch'];
for(const rel of changes){
 const base=path.join(root,rel),target=path.join(candidate,rel);
 if(fs.existsSync(base)){
  const r=spawnSync('git',['diff','--no-index','--no-ext-diff','--',base,target],{encoding:'utf8'});
  assert([0,1].includes(r.status),r.stderr);const lines=r.stdout.split('\n');
  const firstHunk=lines.findIndex(line=>line.startsWith('@@'));
  assert(firstHunk>=0);
  applyParts.push(`*** Update File: ${base.replaceAll('\\','/')}`,
   ...lines.slice(firstHunk).filter((line,i,ls)=>i!==ls.length-1||line!=='')
    .map(line=>line.startsWith('@@')?'@@':line));
  parts.push(lines.map(line=>line.startsWith('diff --git ')?`diff --git a/${rel} b/${rel}`:
   line.startsWith('--- ')?`--- a/${rel}`:line.startsWith('+++ ')?`+++ b/${rel}`:line).join('\n'));
 }else{
  const lines=fs.readFileSync(target,'utf8').trimEnd().split('\n');
  applyParts.push(`*** Add File: ${base.replaceAll('\\','/')}`,...lines.map(l=>'+'+l));
  parts.push(`diff --git a/${rel} b/${rel}\nnew file mode 100644\n--- /dev/null\n+++ b/${rel}\n@@ -0,0 +1,${lines.length} @@\n${lines.map(l=>'+'+l).join('\n')}\n`);
 }
}
const patch=path.join(here,'temperature_loader_cli.patch');fs.writeFileSync(patch,parts.join(''));
const applyPatch=path.join(here,'temperature_loader_cli_apply.patch');
fs.writeFileSync(applyPatch,[...applyParts,'*** End Patch',''].join('\n'));
console.log(JSON.stringify({patch,applyPatch,candidate,paths:changes,production_written:false}));
