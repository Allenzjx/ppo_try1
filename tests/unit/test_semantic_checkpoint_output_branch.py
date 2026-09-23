"""Ancestor output isolation; CPU synthetic PPO is not real robot evidence."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from wlr50_clean.ppo import semantic_cli as cli, semantic_training as training
from wlr50_clean.ppo.semantic_migration import digest

ORIGIN=dict(global_policy_decisions=220544,ppo_updates=1688,optimizer_steps=33760)
SELECTION={"source_role":"front_validated_ancestor_control_eval","counters":ORIGIN,
           "checkpoint_sha256":"a"*64,"source_git_commit":"b"*40}
BRANCH="ancestor220544_contact_v6"


def bind_v6(meta,contract=None):
    contract=contract or {"source_git_commit":"f"*40,"runtime_content_sha256":"e"*64}
    meta["runtime_contract"]=deepcopy(contract)
    meta["rr_contact_onset_v6_migration"]={"schema":"wlr50_clean.rr_contact_onset_same410.v6",
        "target_git_commit":contract["source_git_commit"],"target_contract_sha256":digest(contract),
        "target_runtime_content_sha256":contract["runtime_content_sha256"],"source_selection":deepcopy(SELECTION),
        "rr_contact_onset_v6_factor":{"target_feedback_revision":"progress_reserve_contact_onset_incremental_v5",
            "counter_origin":deepcopy(ORIGIN)}}
    return meta


def metadata():
    return bind_v6({**ORIGIN,"semantic_version":"v3","policy_contract":{"observation_dimension":410},
        "stage_requested_decisions":dict.fromkeys(training.STAGE_BUDGETS,0),
        "rr_progress_handoff_v5_migration":{"source_selection":deepcopy(SELECTION)},
        "rr_capture_transfer_branch_counts":dict.fromkeys(ORIGIN,0)})


def bind_v7(meta,contract=None):
    contract=contract or {"source_git_commit":"7"*40,"runtime_content_sha256":"7"*64}
    meta["runtime_contract"]=deepcopy(contract)
    meta["rr_signed_contact_v7_migration"]={"schema":"wlr50_clean.rr_signed_contact_same410.v7",
        "target_git_commit":contract["source_git_commit"],"target_contract_sha256":digest(contract),
        "target_runtime_content_sha256":contract["runtime_content_sha256"],"source_selection":deepcopy(SELECTION),
        "rr_signed_contact_v7_factor":{"target_feedback_revision":"signed_band_contact_formation_incremental_v6",
            "counter_origin":deepcopy(ORIGIN)}}
    return meta


def write_checkpoint(path,meta):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"synthetic route only")
    path.with_name(path.stem+"_manifest.json").write_text(json.dumps(meta),encoding="utf-8")
    return path


@pytest.fixture
def paths(tmp_path,monkeypatch):
    runs,base,config=tmp_path/"runs",tmp_path/"outputs",tmp_path/"config"
    monkeypatch.setattr(cli,"version_paths",lambda *a,**kw:(runs,base,config))
    source=write_checkpoint(base/"checkpoints/history/ancestor_v6.pt",metadata())
    return runs,base,source


def args(paths,*,source=None,name=BRANCH,command="train",phase="P01"):
    runs,_,default=paths
    argv=[command,"--run-dir",str(runs/"test"),"--expected-head","f"*40,"--semantic-version","v3",
        "--experiment-id","rr_capture_then_rl_transfer_v1","--checkpoint",str(source or default),
        "--stage","full_episode" if phase=="P01" else "phase_suffix","--from-phase",phase]
    if command=="eval": argv += ["--mode","semantic_residual_eval","--seed","4001"]
    else: argv += ["--decisions","128"]
    if phase!="P01": argv += ["--prefix-source","checkpoint_policy"]
    if name is not None: argv += ["--checkpoint-output-branch",name]
    return cli.parser().parse_args(argv)


def route(base):
    return {"schema":"wlr50_clean.checkpoint_output_routing.v1","branch":BRANCH,
        "output_root":str((base/"branches"/BRANCH).resolve()),"main_latest_pointer_promotion":False,
        "source_selection":deepcopy(SELECTION)}


def test_default_output_and_parent_resolution_unchanged(paths):
    request=args(paths,name=None,command="eval");cli.validate_request(request)
    assert cli._checkpoint_output_root(request)==paths[1].resolve()
    assert request.checkpoint==paths[2].resolve() and not hasattr(request,"_checkpoint_output_routing")


def test_ancestor_train_without_branch_rejected_latest_default_unchanged(paths):
    with pytest.raises(ValueError,match="explicit output branch"):
        cli.validate_request(args(paths,name=None))
    meta=metadata()
    for key in ("rr_contact_onset_v6_migration","rr_progress_handoff_v5_migration"):
        meta[key]["source_selection"]["source_role"]="latest_learned_continuation"
    write_checkpoint(paths[2],meta)
    request=args(paths,name=None);cli.validate_request(request)
    assert cli._checkpoint_output_root(request)==paths[1].resolve()


@pytest.mark.parametrize("phase",["P01","P04"])
def test_initial_parent_source_supports_natural_and_checkpoint_policy_prefix(paths,phase):
    request=args(paths,phase=phase);cli.validate_request(request)
    assert request._checkpoint_output_routing==route(paths[1])
    assert request.checkpoint==paths[2].resolve()
    assert request.teacher_offset_decisions==0 and request.from_phase==phase
    if phase=="P04": assert request.prefix_source=="checkpoint_policy"


@pytest.mark.parametrize("name",["..","../escape","a/b",r"a\b","C:/escape","CON","con","lpt9","trailing.",""])
def test_invalid_branch_names_rejected_before_side_effects(paths,name):
    with pytest.raises(ValueError): cli.validate_request(args(paths,name=name))
    assert not (paths[1]/"branches").exists()


def test_branch_reparse_escape_is_rejected(paths,tmp_path):
    outside=tmp_path/"outside";outside.mkdir()
    try: (paths[1]/"branches").symlink_to(outside,target_is_directory=True)
    except OSError: pytest.skip("symlink creation unavailable to this CPU fixture")
    with pytest.raises(ValueError,match="reparse/symlink"):
        cli.validate_request(args(paths))


def test_resolved_alias_guard_without_windows_symlink_privilege(paths,tmp_path,monkeypatch):
    original=Path.resolve;alias=paths[1]/"branches"/BRANCH
    def resolve(path,*a,**kw):
        return tmp_path/"outside" if path==alias else original(path,*a,**kw)
    monkeypatch.setattr(Path,"resolve",resolve)
    with pytest.raises(ValueError,match="reparse/symlink"):
        cli.validate_request(args(paths))


@pytest.mark.parametrize("bad",["foreign_source","mutable_parent","foreign_role","borrowed_counts","occupied","foreign_branch","missing_routing","missing_v6","wrong_v6_head","wrong_v6_factor"])
def test_reject_unsafe_source_or_output_reuse(paths,bad):
    runs,base,source=paths;meta=metadata();destination=base/"branches"/BRANCH
    if bad=="foreign_source": source=write_checkpoint(runs/"foreign.pt",meta)
    elif bad=="mutable_parent": source=write_checkpoint(base/"checkpoints/checkpoint_last.pt",meta)
    elif bad=="foreign_role":
        meta["rr_progress_handoff_v5_migration"]["source_selection"]["source_role"]="latest_learned_continuation"
        write_checkpoint(source,meta)
    elif bad=="borrowed_counts": meta["global_policy_decisions"]+=640;write_checkpoint(source,meta)
    elif bad=="missing_v6": meta.pop("rr_contact_onset_v6_migration");write_checkpoint(source,meta)
    elif bad=="wrong_v6_head": meta["rr_contact_onset_v6_migration"]["target_git_commit"]="0"*40;write_checkpoint(source,meta)
    elif bad=="wrong_v6_factor": meta["rr_contact_onset_v6_migration"]["rr_contact_onset_v6_factor"]["counter_origin"]["ppo_updates"]+=1;write_checkpoint(source,meta)
    elif bad=="occupied": write_checkpoint(destination/"checkpoints/history/existing.pt",meta)
    elif bad=="foreign_branch": source=write_checkpoint(base/"branches/another/checkpoints/history/source.pt",meta)
    else: source=write_checkpoint(destination/"checkpoints/history/source.pt",meta)
    with pytest.raises(ValueError): cli.validate_request(args(paths,source=source))


@pytest.mark.parametrize("command,phase",[("train","P01"),("train","P04"),("eval","P01")])
def test_same_branch_immutable_and_pointer_resolution(paths,command,phase):
    base=paths[1];destination=base/"branches"/BRANCH;meta=metadata()
    meta.update({k:v+d for (k,v),d in zip(ORIGIN.items(),(128,1,20))})
    meta["checkpoint_output_routing"]=route(base)
    source=write_checkpoint(destination/"checkpoints/history/checkpoint_step_000220672.pt",meta)
    manifest=source.with_name(source.stem+"_manifest.json")
    training._publish_last(source,manifest,destination)
    for path in (source,destination/"checkpoints/checkpoint_last.pt"):
        request=args(paths,source=path,command=command,phase=phase)
        cli.validate_request(request)
        assert request.checkpoint==source.resolve() and request._checkpoint_output_routing==route(base)
    pointer=destination/"checkpoints/checkpoint_last_pointer.json"
    binding=json.loads(pointer.read_text());binding["manifest_sha256"]="0"*64;pointer.write_text(json.dumps(binding))
    with pytest.raises(ValueError,match="inconsistent"):
        cli.validate_request(args(paths,source=destination/"checkpoints/checkpoint_last.pt",command=command,phase=phase))


@pytest.mark.parametrize("version",[6,7])
def test_cpu_real_update_writes_only_branch_preserves_main_and_roundtrips(tmp_path,version):
    torch=pytest.importorskip("torch");pytest.importorskip("rsl_rl")
    from test_semantic_rr_capture_training_audit import Synthetic410Core
    from wlr50_clean.ppo.semantic_rr_capture_profile import RR_CAPTURE_POLICY,RR_CAPTURE_OBSERVATION_LAYOUT
    assert not torch.cuda.is_available(),"CPU-only test: CUDA_VISIBLE_DEVICES=-1"
    base=tmp_path/"outputs";destination=base/"branches"/BRANCH;record=route(base)
    preserved=[]
    for step in (220672,220800,220928,221056,221184):
        p=base/f"checkpoints/history/checkpoint_step_{step:09d}.pt";p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(f"preserve main {step}".encode());preserved.append(p)
    for name in ("checkpoint_last.pt","checkpoint_last_pointer.json","resume_state.json"):
        p=base/"checkpoints"/name;p.write_bytes(b"main pointer sentinel");preserved.append(p)
    hashes={str(p):training.sha256_file(p) for p in preserved}
    def make():
        env=training.SemanticRslAdapter(Synthetic410Core(),seed=1001,device="cpu");env.cfg["semantic_version"]="v3"
        runner,_=training.construct_semantic_runner(env,seed=1001,device="cpu",initialize_actor=False,
            policy_version=RR_CAPTURE_POLICY,observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT)
        return runner,env
    runner,env=make();runner.alg.learning_rate=1e-5
    for group in runner.alg.optimizer.param_groups:
        group["lr"]=1e-5
        for p in group["params"]:
            runner.alg.optimizer.state[p]={"step":torch.tensor(33760.),"exp_avg":torch.zeros_like(p),"exp_avg_sq":torch.zeros_like(p)}
    contract={"experiment_id":"rr_capture_then_rl_transfer_v1","semantic_version":"v3",
        "source_git_commit":"f"*40,"runtime_content_sha256":"e"*64,
        "training_budgets":training.training_quantity_budgets("rr_capture_then_rl_transfer_v1"),"evidence":"synthetic CPU only"}
    previous={**ORIGIN,"actor_parameter_sha256":training.parameter_hash(runner.alg.actor),"runtime_contract":contract,
        "stage_requested_decisions":dict.fromkeys(training.STAGE_BUDGETS,0),
        "rr_capture_transfer_branch":{"counter_origin":deepcopy(ORIGIN)},
        "rr_progress_handoff_v5_migration":{"source_selection":deepcopy(SELECTION)},
        "rr_carry_handoff_v4_migration":{"opaque":"must preserve"}}
    bind_v6(previous,contract)
    if version==7: bind_v7(previous,contract)
    with pytest.raises(ValueError,match="explicit output routing"):
        training.train_semantic(runner,env,run_dir=tmp_path/"ancestor_no_flag",output_root=base,
            stage="full_episode",decisions=128,contract=contract,seed=1001,resume_infos=previous)
    assert {int(s["step"].item()) for s in runner.alg.optimizer.state.values()}=={33760}
    result=training.train_semantic(runner,env,run_dir=tmp_path/"run",output_root=destination,stage="full_episode",
        decisions=128,contract=contract,seed=1001,resume_infos=previous,checkpoint_interval_updates=1,
        checkpoint_output_routing=record)
    assert (result["actual_policy_decisions"],result["ppo_updates_this_run"],result["optimizer_steps_this_run"])==(128,1,20)
    assert result["global_policy_decisions"]==220672 and result["checkpoint_output_routing"]==record
    assert {str(p):training.sha256_file(p) for p in preserved}==hashes
    saved=Path(result["checkpoints"][0]["checkpoint"])
    assert saved==destination/"checkpoints/history/checkpoint_step_000220672.pt"
    fresh,_=make();loaded=training.load_semantic_checkpoint(fresh,saved,contract=contract,seed=1001)
    assert {k:loaded[k] for k in ORIGIN}==dict(global_policy_decisions=220672,ppo_updates=1689,optimizer_steps=33780)
    assert loaded["checkpoint_output_routing"]==record
    assert loaded["rr_progress_handoff_v5_migration"]==previous["rr_progress_handoff_v5_migration"]
    assert loaded["rr_carry_handoff_v4_migration"]==previous["rr_carry_handoff_v4_migration"]
    assert loaded["rr_contact_onset_v6_migration"]==previous["rr_contact_onset_v6_migration"]
    if version==7: assert loaded["rr_signed_contact_v7_migration"]==previous["rr_signed_contact_v7_migration"]
    assert loaded["rr_capture_transfer_branch_counts"]==dict(global_policy_decisions=128,ppo_updates=1,optimizer_steps=20)
    assert training.state_hash(fresh.alg.optimizer.state_dict())==training.state_hash(runner.alg.optimizer.state_dict())
    assert {int(s["step"].item()) for s in fresh.alg.optimizer.state.values()}=={33780}
    with pytest.raises(ValueError,match="explicit output routing"):
        training.train_semantic(fresh,env,run_dir=tmp_path/"bad",output_root=base,stage="full_episode",decisions=128,
            contract=contract,seed=1001,resume_infos=loaded)
    bad_record={**record,"output_root":str(base.resolve())}
    with pytest.raises(ValueError,match="routing differs"):
        training.train_semantic(fresh,env,run_dir=tmp_path/"main_override",output_root=base,stage="full_episode",decisions=128,
            contract=contract,seed=1001,resume_infos=previous,checkpoint_output_routing=bad_record)
    with pytest.raises(FileExistsError,match="no optimizer step"):
        training.train_semantic(fresh,env,run_dir=tmp_path/"collision",output_root=destination,stage="full_episode",decisions=128,
            contract=contract,seed=1001,resume_infos=previous,checkpoint_output_routing=record)


def test_train_and_video_launchers_forward_explicit_branch():
    root=Path(__file__).resolve().parents[2]
    for name in ("run_semantic_ppo.ps1","run_semantic_video.ps1"):
        text=(root/"scripts"/name).read_text()
        assert "[string]$CheckpointOutputBranch" in text
        assert "@('--checkpoint-output-branch',$CheckpointOutputBranch)" in text


@pytest.mark.parametrize("phase",["P01","P04"])
def test_v7_formal_ancestor_initial_and_same_branch_resume(paths,phase):
    meta=bind_v7(metadata());old_receipt=deepcopy(meta["rr_contact_onset_v6_migration"])
    write_checkpoint(paths[2],meta)
    with pytest.raises(ValueError,match="explicit output branch"):
        cli.validate_request(args(paths,name=None,phase=phase))
    request=args(paths,phase=phase);cli.validate_request(request)
    assert request._checkpoint_output_routing==route(paths[1])
    assert meta["rr_contact_onset_v6_migration"]==old_receipt
    destination=paths[1]/"branches"/BRANCH
    meta.update({k:v+d for (k,v),d in zip(ORIGIN.items(),(128,1,20))})
    meta["checkpoint_output_routing"]=route(paths[1])
    source=write_checkpoint(destination/"checkpoints/history/checkpoint_step_000220672.pt",meta)
    training._publish_last(source,source.with_name(source.stem+"_manifest.json"),destination)
    for command in ("train","eval"):
        request=args(paths,source=destination/"checkpoints/checkpoint_last.pt",phase=phase if command=="train" else "P01",command=command)
        cli.validate_request(request)
        assert request.checkpoint==source.resolve()


@pytest.mark.parametrize("bad",["empty","wrong_feedback","stale_head","stale_contract"])
def test_v7_receipt_never_falls_back_to_v6(paths,bad):
    meta=metadata();bind_v7(meta,meta["runtime_contract"])
    receipt=meta["rr_signed_contact_v7_migration"]
    if bad=="empty": meta["rr_signed_contact_v7_migration"]={}
    elif bad=="wrong_feedback": receipt["rr_signed_contact_v7_factor"]["target_feedback_revision"]="progress_reserve_contact_onset_incremental_v5"
    elif bad=="stale_head": receipt["target_git_commit"]="0"*40
    else: receipt["target_contract_sha256"]="0"*64
    write_checkpoint(paths[2],meta)
    with pytest.raises(ValueError,match="formally published"):
        cli.validate_request(args(paths))
