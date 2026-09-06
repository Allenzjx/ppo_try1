"""Offline entry/continuation checks; synthetic smoke fixtures are not live proof."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest
from wlr50_clean.ppo.semantic_migration import topology, source_num_envs, stage_partition
from wlr50_clean.ppo.semantic_cli import smoke_actions
from wlr50_clean.ppo import semantic_cli

def test_exact_100k_partition_and_honest_single_tail_rounding():
    assert stage_partition(100000) == {
        "N8_requested":99328,"N8_actual":99328,"N8_updates":97,
        "N1_requested":672,"N1_actual":768,"N1_updates":6,
        "total_requested":100000,"total_actual":100096,"rounding_overrun":96}

def test_sampling_label_is_not_a_phase_suffix_curriculum_claim():
    record = topology(8)
    assert record["observation_dimension"] == 324
    assert record["phase_suffix_curriculum_implemented"] is False
    assert record["reset_sampling"] == "P01_only"
    assert record["task_timeout_bootstrap"] is False

def test_vector_capable_checkpoint_cannot_omit_source_topology():
    from wlr50_clean.ppo.semantic_migration import VECTOR_FILES
    with pytest.raises(ValueError,match="lacks explicit execution topology"):
        source_num_envs({"runtime_contract":{"files":{next(iter(VECTOR_FILES)):"a"*64}}})
    assert source_num_envs({"execution_topology":topology(8)}) == 8

def test_smoke_zero_single_row_and_identifiable_every_row_are_distinct():
    assert smoke_actions(0) == [[0.0]*12 for _ in range(8)]
    probe = smoke_actions(8)
    assert probe[0][8] > .1 and all(not any(row) for row in probe[1:])
    varied = smoke_actions(32)
    assert len({tuple(row) for row in varied}) == 8
    assert all(any(row) for row in varied)

def test_n8_nonwhole_rollout_request_fails_before_launcher(tmp_path,monkeypatch):
    monkeypatch.setattr(semantic_cli,"RUNS_ROOT",tmp_path/"runs")
    monkeypatch.setattr(semantic_cli,"OUTPUT_ROOT",tmp_path/"out")
    source = tmp_path / "out/checkpoints/source.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"test checkpoint")
    source.with_name("source_manifest.json").write_text(json.dumps({"stage_requested_decisions":{"phase_suffix":0}}))
    args = semantic_cli.parser().parse_args(["train","--run-dir",str(tmp_path/"runs/train"),
        "--expected-head","a"*40,"--num-envs","8","--stage","phase_suffix","--decisions","100000",
        "--checkpoint",str(source),"--vector-smoke-evidence",str(tmp_path/"smoke.json")])
    with pytest.raises(ValueError,match="whole128x8"):
        semantic_cli.validate_request(args)


def test_current_5520_smoke_remainder_is_five_vector_updates_and_explicit_tail():
    assert stage_partition(5520) == {
        "N8_requested":5120,"N8_actual":5120,"N8_updates":5,
        "N1_requested":400,"N1_actual":512,"N1_updates":4,
        "total_requested":5520,"total_actual":5632,"rounding_overrun":112}
    assert 4480 + stage_partition(5520)["total_actual"] == 10112


def test_gpu_probe_really_constructs_and_records_unavailable_not_zero(tmp_path,monkeypatch):
    monkeypatch.setattr(semantic_cli.subprocess,"run",lambda *a,**kw: SimpleNamespace(
        returncode=0,stdout="123, N/A\n",stderr=""))
    probe = semantic_cli.GpuProbe(tmp_path,"cpu")
    try:
        row = probe.sample("CPU_contract_probe")
        assert row["pid"] > 0
        assert row["process_memory"]["stdout"] == "123, N/A"
        assert row["process_memory"]["N_A_is_unavailable_not_zero"] is True
        assert probe.summary()["sample_count"] == 1
        assert json.loads((tmp_path/"gpu_memory.jsonl").read_text())["event"] == "CPU_contract_probe"
    finally:
        probe.close()


def test_gpu_probe_initializes_torch_allocator_before_resetting_stats(tmp_path,monkeypatch):
    import torch
    events=[]
    monkeypatch.setattr(torch.cuda,"init",lambda:events.append("init"))
    def reset(device):
        assert events==["init"]
        events.append(("reset",device))
    monkeypatch.setattr(torch.cuda,"reset_peak_memory_stats",reset)
    probe=semantic_cli.GpuProbe(tmp_path,"cuda:0")
    probe.close()
    assert events==["init",("reset","cuda:0")]


def test_gpu_probe_actual_installed_torch_cold_process_without_isaac(tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("requires an actual CUDA device; CPU mocks do not prove allocator initialization")
    code = """
import json,sys
from pathlib import Path
import torch
from wlr50_clean.ppo.semantic_cli import GpuProbe
assert not torch.cuda.is_initialized(), 'test requires cold Torch allocator'
probe=GpuProbe(Path(sys.argv[1]),'cuda:0')
try:
    assert torch.cuda.is_initialized()
    row=probe.sample('cold_torch_no_isaac')
    assert row['device_total_bytes']>0
    assert row['torch_allocated_bytes']>=0
    assert row['torch_peak_allocated_bytes']>=row['torch_allocated_bytes']
    assert probe.summary()['sample_count']==1
    print(json.dumps({'event':row['event'],'total':row['device_total_bytes']}))
finally:
    probe.close()
"""
    result=subprocess.run([sys.executable,"-c",code,str(tmp_path)],capture_output=True,text=True,timeout=45)
    assert result.returncode==0,result.stdout+result.stderr
    assert json.loads(result.stdout.strip())["event"]=="cold_torch_no_isaac"
    assert json.loads((tmp_path/"gpu_memory.jsonl").read_text())["event"]=="cold_torch_no_isaac"


def test_physical_row_probe_consumes_real_authoritative_raw_dataclasses(tmp_path):
    from test_semantic_observation_reward_env import _frame
    with (tmp_path/"physical.jsonl").open("x") as stream:
        probe = semantic_cli._RowProbe(3,stream)
        probe.reset()
        probe.observe(_frame(),_frame(1),None)
        probe.reset()
        probe.observe(_frame(),_frame(1),None)
        assert len(probe.batches)==2 and probe.summary()["measured_physics_ticks"]==2
        assert len(probe.batches[0][0]["actual_full12"])==12


def test_real_vector_kernel_smoke_collects_two_resets_zero_and_identifiable_native_rows(tmp_path,monkeypatch):
    from test_semantic_vector_draft import FakeVectorBackend
    backend=FakeVectorBackend()
    def offline_isolation(control,treatment,actual_backend):
        # Only PhysX isolation is replaced here; all real kernel/callback/audit
        # code runs. This fixture never publishes a production live artifact.
        assert actual_backend is backend
        assert all(len(rows)==512 for rows in control+treatment)
        return {"passed":True,"fixture_only_not_live_evidence":True}
    monkeypatch.setattr(semantic_cli,"_isolation",offline_isolation)
    gpu=SimpleNamespace(sample=lambda event: None,summary=lambda:{"fixture_only":True})
    result=semantic_cli._smoke(backend,SimpleNamespace(run_dir=tmp_path,device="cpu"),{},gpu)
    assert result["explicit_reset_count"]==result["actual_reset_count"]==2
    assert result["optimizer_steps"]==0
    assert all(n>0 for n in result["per_row_effect_counts"])
    rows=[json.loads(line) for line in (tmp_path/"vector_native_audit.jsonl").read_text().splitlines()]
    assert len(rows)==1024
    assert all(not any(row["info"]["raw_policy_action_full12"]) for row in rows if row["segment"]==0)


def test_actual_isolation_measurements_reject_unexcited_row_physical_contamination():
    import torch
    def traces():
        return [[{"physics_tick":tick,"actual_full12":[0.]*12,"base_position_m":[0.]*3}
                 for tick in range(65,257)] for _ in range(8)]
    control,treatment=traces(),traces()
    for row in treatment[0]:
        row["actual_full12"][8]=.1
    backend=SimpleNamespace(scene=SimpleNamespace(env_origins=torch.tensor([[8.*row,0.,0.] for row in range(8)]),
        cfg=SimpleNamespace(filter_collisions=True,replicate_physics=True)))
    assert semantic_cli._isolation(control,treatment,backend)["passed"] is True
    treatment[3][0]["base_position_m"][1]=.1
    assert semantic_cli._isolation(control,treatment,backend)["passed"] is False


def fixture_contracts_and_smoke(tmp_path):
    from wlr50_clean.ppo import semantic_migration as migration
    from test_semantic_migration import contract
    old,new = contract(1),contract(2)
    for relative in migration.VECTOR_FILES:
        path = tmp_path/relative
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((migration.PROJECT_ROOT/relative).read_bytes())
        new["files"][relative] = migration.file_sha(path)
    new["runtime_content_sha256"] = migration.digest(new["files"])
    directory = tmp_path/"runs/ppo_semantic_v2/interface_smoke/offline_fixture_not_live"
    directory.mkdir(parents=True)
    data = {"schema":"wlr50_clean.semantic_vector_smoke.v1","runtime_contract":new,
        "num_envs":8,"explicit_reset_count":2,"optimizer_steps":0,
        "functional_passed":True,"all_row_native_audits_verified":True,
        "one_step_write_capture_verified":True,"physical_isolation_verified":True,
        "per_row_effect_counts":[1]*8,"fixture_only_not_live_evidence":True}
    for name in ("vector_native_audit.jsonl","vector_physical_probe.jsonl","gpu_memory.jsonl"):
        (directory/name).write_text('{"fixture_only_not_live_evidence":true}\n')
    path = directory/"vector_smoke_manifest.json"
    path.write_text(json.dumps(data))
    (directory/"run_manifest.json").write_text(json.dumps({"lifecycle":"SUCCEEDED","result":data}))
    return old,new,path


@pytest.mark.parametrize("fault",["missing_module","existing_module_changed","source8_without_modules","task_change","stale_smoke","missing_smoke_artifact"])
def test_execution_factor_is_exact_additive_and_never_a_task_change_waiver(tmp_path,fault):
    from wlr50_clean.ppo import semantic_migration as migration
    from test_semantic_migration import checkpoint
    old,new,smoke = fixture_contracts_and_smoke(tmp_path)
    if fault=="missing_module":
        del new["files"][next(iter(migration.VECTOR_FILES))]
    elif fault=="existing_module_changed":
        old["files"].update({path:"0"*64 for path in migration.VECTOR_FILES})
    elif fault=="task_change":
        old["files"][migration.SUPERVISOR]="a"*64
        new["files"][migration.SUPERVISOR]="b"*64
    elif fault=="stale_smoke":
        data=json.loads(smoke.read_text());data["runtime_contract"]["source_git_commit"]="f"*40
        smoke.write_text(json.dumps(data))
    elif fault=="missing_smoke_artifact":
        smoke.with_name("vector_native_audit.jsonl").unlink()
    for value in (old,new):
        value["runtime_content_sha256"]=migration.digest(value["files"])
    source=checkpoint(tmp_path,old)
    if fault=="source8_without_modules":
        sidecar=source.with_name(source.stem+"_manifest.json")
        data=json.loads(sidecar.read_text());data["execution_topology"]=topology(8)
        sidecar.write_text(json.dumps(data))
    changed=sorted(k for k in set(old["files"])|set(new["files"]) if old["files"].get(k)!=new["files"].get(k))
    with pytest.raises(ValueError):
        migration.build_migration_plan(source,new,allowed_changed_files=changed,reason="reviewed execution only",
            execution_evidence={"target_num_envs":1 if fault=="source8_without_modules" else 8,"vector_smoke":smoke},project_root=tmp_path)


def test_real_main_rejects_undeclared_same_head_topology_switch_before_app(tmp_path,monkeypatch):
    from wlr50_clean.ppo import semantic_migration as migration
    old,new,smoke=fixture_contracts_and_smoke(tmp_path)
    output=tmp_path/"outputs/ppo_semantic_v2"
    source=output/"checkpoints/history/source.pt"
    source.parent.mkdir(parents=True);source.write_bytes(b"fixture checkpoint")
    source.with_name("source_manifest.json").write_text(json.dumps({
        "schema":"wlr50_clean.semantic_checkpoint.v1","checkpoint_path":str(source.resolve()),
        "checkpoint_sha256":migration.file_sha(source),"save_load_round_trip":True,
        "runtime_contract":new,"seed":1001,"execution_topology":topology(1),
        "runner_config":semantic_cli.semantic_runner_config(seed=1001,device="cuda:0"),
        "stage_requested_decisions":{"smoke":4480}}))
    monkeypatch.setattr(semantic_cli,"PROJECT_ROOT",tmp_path)
    monkeypatch.setattr(semantic_cli,"RUNS_ROOT",tmp_path/"runs/ppo_semantic_v2")
    monkeypatch.setattr(semantic_cli,"OUTPUT_ROOT",output)
    monkeypatch.setattr(semantic_cli,"runtime_contract",lambda **kw:new)
    monkeypatch.setattr(semantic_cli,"dispatch_live",lambda *a:pytest.fail("must reject before any Isaac import"))
    with pytest.raises(ValueError,match="changing checkpoint execution topology"):
        semantic_cli.main(["train","--run-dir",str(tmp_path/"runs/ppo_semantic_v2/train/rejected"),
            "--expected-head",new["source_git_commit"],"--num-envs","8","--checkpoint",str(source),
            "--decisions","1024","--vector-smoke-evidence",str(smoke)])
    assert not (tmp_path/"runs/ppo_semantic_v2/train/rejected").exists()


def test_real_official_checkpoint_one_to_eight_update_to_one_preserves_learning_lineage(tmp_path,monkeypatch):
    import torch
    from wlr50_clean.ppo import semantic_migration as migration
    from wlr50_clean.ppo.semantic_training import (train_semantic,load_semantic_checkpoint,
        state_hash,parameter_hash,write_json)
    from test_semantic_training import Core,make_runner
    from test_semantic_vector_draft import FakeVectorBackend
    from wlr50_clean.ppo.semantic_vector_env import SemanticVectorRslEnv
    from wlr50_clean.ppo.semantic_vector_training import construct_semantic_vector_runner,train_semantic_vector
    class Core324(Core):
        def reset(self,*,seed=1001,options=None):
            return super().reset(seed=seed,options=options)+(0.,)*316
        def step(self,raw):
            result=super().step(raw);result.observation+=(0.,)*316;return result
    prior_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        old,new,smoke=fixture_contracts_and_smoke(tmp_path)
        output=tmp_path/"outputs"
        runner,env=make_runner(Core324())
        train_semantic(runner,env,run_dir=tmp_path/"one",output_root=output,stage="smoke",decisions=128,contract=old,seed=1001)
        first=output/"checkpoints/history/checkpoint_step_000000128.pt"
        plan=migration.build_migration_plan(first,new,
            allowed_changed_files=sorted(k for k in new["files"] if old["files"].get(k)!=new["files"][k]),
            reason="reviewed additive execution",execution_evidence={"target_num_envs":8,"vector_smoke":smoke},project_root=tmp_path)
        plan_path=tmp_path/"one_to_eight.json";write_json(plan_path,plan)
        original_validate=migration.validate_migration_plan
        monkeypatch.setattr(migration,"validate_migration_plan",lambda *a,**kw: original_validate(*a,project_root=tmp_path,**kw))
        record=migration.validate_migration_plan(first,new,plan_path)
        actor=parameter_hash(runner.alg.actor);optimizer=state_hash(runner.alg.optimizer.state_dict())
        expected_rng=torch.rand(8)
        vector=SemanticVectorRslEnv(FakeVectorBackend(terminal_tick=3),device="cpu")
        batch_runner,_=construct_semantic_vector_runner(vector,seed=1001,device="cpu")
        infos=load_semantic_checkpoint(batch_runner,first,contract=new,seed=1001,migration=record)
        assert parameter_hash(batch_runner.alg.actor)==actor
        assert state_hash(batch_runner.alg.optimizer.state_dict())==optimizer
        assert torch.equal(torch.rand(8),expected_rng)
        assert batch_runner.alg.storage.step==0 and batch_runner.alg.storage.actions.shape==(128,8,12)
        assert batch_runner.alg.transition.actions is None and vector.total_decisions==0
        result=train_semantic_vector(batch_runner,vector,run_dir=tmp_path/"eight",output_root=output,
            stage="smoke",decisions=1024,contract=new,seed=1001,resume_infos=infos)
        assert result["global_policy_decisions"]==1152 and result["optimizer_steps_this_run"]==20
        second=output/"checkpoints/history/checkpoint_step_000001152.pt"
        meta=json.loads(second.with_name(second.stem+"_manifest.json").read_text())
        assert meta["optimizer_steps"]==40 and meta["stage_requested_decisions"]["smoke"]==1152
        assert meta["execution_topology"]==topology(8) and meta["phase_suffix_curriculum_implemented"] is False
        plan=migration.build_migration_plan(second,new,allowed_changed_files=[],reason="explicit N1 tail",
            execution_evidence={"target_num_envs":1,"vector_smoke":smoke},project_root=tmp_path)
        back_path=tmp_path/"eight_to_one.json";write_json(back_path,plan)
        back_record=migration.validate_migration_plan(second,new,back_path)
        new_actor=parameter_hash(batch_runner.alg.actor)
        new_optimizer=state_hash(batch_runner.alg.optimizer.state_dict());expected_rng=torch.rand(8)
        tail_runner,tail_env=make_runner(Core324())
        previous=load_semantic_checkpoint(tail_runner,second,contract=new,seed=1001,migration=back_record)
        assert parameter_hash(tail_runner.alg.actor)==new_actor
        assert state_hash(tail_runner.alg.optimizer.state_dict())==new_optimizer
        assert torch.equal(torch.rand(8),expected_rng)
        assert tail_runner.alg.storage.step==0 and tail_env.total_decisions==0
        tail=train_semantic(tail_runner,tail_env,run_dir=tmp_path/"tail",output_root=output,
            stage="smoke",decisions=16,contract=new,seed=1001,resume_infos=previous)
        assert tail["rounding_overrun"]==112 and tail["global_policy_decisions"]==1280
        assert tail["requested_policy_decisions"]==16
    finally:
        torch.set_num_threads(prior_threads)
