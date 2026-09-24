"""Pure metadata tests for the unpromoted cooperative-prep draft; no Torch."""
import copy
import importlib.util
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("cooperative_prep_draft", HERE / "semantic_cooperative_prep_migration.py")
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


def fixture():
    source_policy = {
        "version": m.SOURCE_POLICY,
        "actor_class": "old.Actor",
        "observation_layout": m.OBSERVATION_LAYOUT,
        "observation_dimension": 422,
        "sigma_semantics": "old",
    }
    target_policy = {
        **source_policy,
        "version": m.TARGET_POLICY,
        "actor_class": "new.Actor",
        "sigma_semantics": "observable_overlap_only",
    }
    policy_delta = {
        key: {"before": source_policy.get(key), "after": target_policy.get(key)}
        for key in ("version", "actor_class", "sigma_semantics")
    }
    old = {
        "source_git_commit": m.SOURCE_HEAD,
        "runtime_content_sha256": "a" * 64,
        "files": {"src/old.py": "1" * 64, "configs/reward.yaml": "2" * 64},
        "selected_configuration": {
            "reward.yaml": {"path": "configs/reward.yaml", "sha256": "2" * 64},
        },
    }
    new = copy.deepcopy(old)
    new.update(source_git_commit="b" * 40, runtime_content_sha256="c" * 64)
    new["files"]["src/old.py"] = "3" * 64
    new["files"]["configs/reward.yaml"] = "4" * 64
    new["selected_configuration"]["reward.yaml"]["sha256"] = "4" * 64
    runtime_delta = {
        "src/old.py": {"before": "1" * 64, "after": "3" * 64},
        "configs/reward.yaml": {"before": "2" * 64, "after": "4" * 64},
    }
    counts = dict(zip(m.COUNTERS, (1234, 12, 240)))  # Deliberately non-production synthetic values.
    binding = {
        "checkpoint": "X:/sealed/future.pt",
        "checkpoint_sha256": "5" * 64,
        "manifest_sha256": "6" * 64,
        "source_git_commit": m.SOURCE_HEAD,
        **counts,
    }
    metadata = {
        **counts,
        "runtime_contract": old,
        "policy_contract": source_policy,
        "save_load_round_trip": True,
        "seed": 4001,
        "actor_parameter_sha256": "7" * 64,
        "critic_parameter_sha256": "8" * 64,
        "optimizer_state_sha256": "9" * 64,
        "optimizer_learning_rate": 1e-5,
        "normalizer_state_sha256": "a" * 64,
        "normalization": "Identity",
        "training_rng_state": {"python": [1, 2, 3]},
        "checkpoint_output_routing": {"branch": "ancestor220544_recapture_v2"},
        "rear_policy_timing_branch": {"counter_origin": {"global_policy_decisions": 220544}},
        "rear_policy_timing_branch_counts": {"global_policy_decisions": 2688},
        "rear_live_swing_migration": {"immutable": "old"},
        "rear_recapture_migration": {"immutable": "older"},
        "p02_progress_migration": {"schema": "wlr50_clean.p02_progress_append.v1"},
    }
    return metadata, old, new, binding, runtime_delta, source_policy, target_policy, policy_delta


def build():
    metadata, old, new, binding, runtime_delta, source_policy, target_policy, policy_delta = fixture()
    calls = []

    def old_validator(candidate, contract):
        calls.append((candidate, contract))
        assert candidate["policy_contract"] == source_policy
        assert contract == old
        assert m.MIGRATION not in candidate

    receipt = m.build_draft_receipt(
        metadata,
        new,
        reason="synthetic metadata boundary only",
        source_binding=binding,
        reviewed_runtime_delta=runtime_delta,
        reviewed_policy_delta=policy_delta,
        target_policy_contract=target_policy,
        ancestry_validator=old_validator,
    )
    assert len(calls) == 1
    return metadata, new, binding, runtime_delta, target_policy, policy_delta, receipt


def test_build_is_zero_credit_identity_and_validates_old_policy_first():
    metadata, new, _, _, target_policy, _, receipt = build()
    factor = receipt[m.FACTOR_KEY]
    assert factor["source_policy_contract"] == metadata["policy_contract"]
    assert factor["target_policy_contract"] == target_policy
    assert factor["parameter_mapping"].startswith("identity_all_actor_critic")
    assert factor["same_numeric_input_Gaussian_preserved"] is False
    assert all(factor[key] == 0 for key in m.ZERO_CREDIT)
    assert m.reconstruct_source_contract(new, receipt) == metadata["runtime_contract"]


def test_lineage_reconstructs_old_contract_and_policy_before_parent_validation():
    metadata, new, _, _, target_policy, _, receipt = build()
    target = copy.deepcopy(metadata)
    target.update(runtime_contract=new, policy_contract=target_policy)
    target[m.MIGRATION] = receipt
    seen = []

    def old_validator(candidate, contract):
        seen.append(True)
        assert m.MIGRATION not in candidate
        assert candidate["policy_contract"]["version"] == m.SOURCE_POLICY
        assert contract["source_git_commit"] == m.SOURCE_HEAD

    assert m.validate_draft_lineage(target, new, ancestry_validator=old_validator) == receipt
    assert seen == [True]


def test_unsealed_source_and_unreviewed_deltas_fail_closed():
    metadata, _, new, binding, runtime_delta, _, target_policy, policy_delta = fixture()
    kwargs = dict(reason="x", source_binding=binding, reviewed_runtime_delta=runtime_delta,
        reviewed_policy_delta=policy_delta, target_policy_contract=target_policy,
        ancestry_validator=lambda *_: None)
    with pytest.raises(ValueError, match="not sealed"):
        m.build_draft_receipt(metadata, new, **{**kwargs, "source_binding": None})
    with pytest.raises(ValueError, match="runtime/config delta"):
        m.build_draft_receipt(metadata, new, **{**kwargs, "reviewed_runtime_delta": None})
    with pytest.raises(ValueError, match="policy delta"):
        m.build_draft_receipt(metadata, new, **{**kwargs, "reviewed_policy_delta": None})


def test_tampered_count_hash_extra_runtime_or_forged_source_policy_rejected():
    metadata, _, new, binding, runtime_delta, _, target_policy, policy_delta = fixture()
    kwargs = dict(reason="x", source_binding=binding, reviewed_runtime_delta=runtime_delta,
        reviewed_policy_delta=policy_delta, target_policy_contract=target_policy,
        ancestry_validator=lambda *_: None)
    bad = copy.deepcopy(metadata)
    bad["ppo_updates"] += 1
    with pytest.raises(ValueError, match="counters"):
        m.build_draft_receipt(bad, new, **kwargs)
    bad_binding = copy.deepcopy(binding)
    bad_binding["checkpoint_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="SHA256"):
        m.build_draft_receipt(metadata, new, **{**kwargs, "source_binding": bad_binding})
    expanded = copy.deepcopy(new)
    expanded["files"]["src/unreviewed.py"] = "f" * 64
    with pytest.raises(ValueError, match="outside"):
        m.build_draft_receipt(metadata, expanded, **kwargs)
    forged = copy.deepcopy(metadata)
    forged["policy_contract"] = target_policy
    with pytest.raises(ValueError, match="intact sealed P02"):
        m.build_draft_receipt(forged, new, **kwargs)


def test_lineage_rejects_changed_full_state_or_nonzero_migration_credit():
    metadata, new, _, _, target_policy, _, receipt = build()
    target = copy.deepcopy(metadata)
    target.update(runtime_contract=new, policy_contract=target_policy)
    target[m.MIGRATION] = receipt
    target["optimizer_state_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="optimizer_state_sha256"):
        m.validate_draft_lineage(target, new, ancestry_validator=lambda *_: None)
    target = copy.deepcopy(metadata)
    target.update(runtime_contract=new, policy_contract=target_policy)
    target[m.MIGRATION] = copy.deepcopy(receipt)
    target[m.MIGRATION][m.FACTOR_KEY]["added_ppo_updates"] = 1
    with pytest.raises(ValueError, match="identity boundary"):
        m.validate_draft_lineage(target, new, ancestry_validator=lambda *_: None)
