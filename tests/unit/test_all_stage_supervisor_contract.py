"""New scheduler/namespace tests; synthetic conditions are not physical success."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
from wlr50_clean.ppo.semantic_supervisor import TaskStageSupervisor, TaskEvaluator, PHASE_IDS
from wlr50_clean.ppo.semantic_cli import version_paths, parser
from test_semantic_supervisor import observation, leg_state

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "configs/ppo_all_stage_acceptance_v1/stage_task_spec.yaml"

def supervisor(phase="P02"):
    # Deliberately use the old fully verified synthetic physical fixture to
    # isolate scheduler behavior from the new contact classifier unit tests.
    return TaskStageSupervisor(SPEC, evaluator=TaskEvaluator(), initial_stage_id=phase)

def timed_supervisor(phase="P02"):
    sup=supervisor(phase)
    value=sup.evaluator.observe(observation())
    for current in value["current_legs"].values():
        current["whole_body_actuation_evidence"] = False
    sup.evaluator=SimpleNamespace(observe=lambda _:deepcopy(value), snapshot=value)
    return sup

@pytest.mark.parametrize("phase", PHASE_IDS)
def test_all_local_limits_are_finite_bounded_and_keep_global_200(phase):
    sup = timed_supervisor(phase)
    first = sup.observe_and_update(observation())
    assert first["termination_reason"] is None
    limit = sup.spec["stages"][phase]["maximum_task_duration"]
    obs = observation(round((limit + 11)*120))
    end = sup.observe_and_update(obs)
    assert end["termination_reason"] == "INCOMPLETE_CONTROLLER_BLOCKED"
    assert end["termination_source"].startswith("LOCAL_")
    assert end["local_timeout"]["effective_limit_s"] <= limit + min(10, limit*.5)
    assert end["remaining_task_time_s"] == pytest.approx(200-obs["simulation_time_s"])

def test_global_is_finite_terminal_not_external_truncation():
    sup = timed_supervisor()
    sup.observe_and_update(observation())
    end = sup.observe_and_update(observation(24000))
    assert end["termination_source"] == "GLOBAL_FINITE_TASK_DEADLINE"

@pytest.mark.parametrize("distance,accepted", [(-.010001,False),(-.009,True),(.104,True),(.106,False)])
def test_declared_measurement_margin_is_symmetric_not_fitted_to_old_run(distance, accepted):
    sup = supervisor()
    obs = observation()
    leg_state(obs,"FR",x=.5+distance)
    ev = sup.evaluator.observe(obs)
    assert (sup.predicate("approach_FR",ev)==1.) is accepted

def test_downstream_current_crossing_does_not_need_reversal_to_old_interval():
    sup = supervisor()
    obs = observation()
    ev = deepcopy(sup.evaluator.observe(obs))
    ev["history"]["front_edge_crossed"]["FR"] = True
    current = ev["current_legs"]["FR"]
    current.update(front_distance_m=.20, ground_contact=False, air=True)
    assert sup.predicate("approach_FR",ev)==1.
    current["ground_contact"] = True
    assert sup.predicate("approach_FR",ev)<1.

def test_new_namespace_has_isolated_configs_and_no_policy_architecture_flag():
    paths=version_paths("v3",experiment_id="all_stage_acceptance_v1")
    assert [p.name for p in paths]==["ppo_all_stage_acceptance_v1"]*3
    args=parser().parse_args(["preflight","--run-dir",str(paths[0]/"diagnostics/test"),
        "--expected-head","a"*40,"--semantic-version","v3","--experiment-id","all_stage_acceptance_v1"])
    assert args.num_envs==1 and args.target_policy_version is None

def test_already_captured_current_top_does_not_require_second_clearance_hop():
    sup=supervisor()
    ev=deepcopy(sup.evaluator.observe(observation()))
    current=ev["current_legs"]["FR"]
    current.update(within_top_xy=True, ground_contact=False, top_contact=True, air=False, clearance_m=0.)
    assert sup.predicate("clear_FR",ev)==0.
    ev["history"]["front_edge_crossed"]["FR"]=True
    assert sup.predicate("clear_FR",ev)==1.
    current["ground_contact"]=True
    assert sup.predicate("clear_FR",ev)==0.
