from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import semantic_cli as cli


class SmokeCore:
    def __init__(self, *, physical_terminal_every=None, missing_audit=False, quantized=False):
        self.resets = 0
        self.actions = []
        self.frame = SimpleNamespace(sim_time_s=0.0)
        self.terminal_every = physical_terminal_every
        self.missing_audit = missing_audit
        self.quantized = quantized

    def reset(self, *, seed):
        self.resets += 1
        self.tick = 0
        self.done = False
        self.frame.sim_time_s = 0.0
        return (0.0, 1.0)

    def step(self, raw):
        self.actions.append(raw)
        self.tick += 1
        self.frame.sim_time_s = self.tick / 15
        self.done = self.tick == self.terminal_every
        changed = int(any(raw) and not self.quantized)
        audit = {"schema": "wlr50_clean.actuator_target_effect_audit.v1", "verified": True,
                 "actual_mapping_matches_dispatch": True, "setter_dispatch_targets_equal": True,
                 "same_tick_counterfactual": True, "raw_policy_action_full12": raw,
                 "target_dtype": "torch.float32", "changed_target_channel_count": changed,
                 "projected_residual_full12": raw}
        info = {"raw_policy_action_full12": raw, "actuator_target_effect_audit": None if self.missing_audit else audit,
                "task_success": False, "termination_reason": "FALL" if self.done else None}
        return SimpleNamespace(observation=(self.tick / 15, 1.0), reward=0.0,
                               terminated=self.done, truncated=False, info=info)

    def telemetry_summary(self):
        return {"reset_count": self.resets}


def smoke_args(tmp_path):
    return cli.parser().parse_args(["smoke", "--run-dir", str(tmp_path), "--expected-head", "a" * 40,
                                    "--max-decisions", "128", "--device", "cpu"])


def test_smoke_two_same_backend_resets_real_scale_and_zero_control(tmp_path):
    core = SmokeCore()
    result = cli._evaluation(core, smoke_args(tmp_path), contract={})
    assert core.resets == 2
    assert len(core.actions) == 128
    assert core.actions[:4] == [(0.0,) * 12] * 4
    assert core.actions[64:68] == [(0.0,) * 12] * 4
    assert max(max(map(abs, action)) for action in core.actions) == pytest.approx(math.atanh(0.05))
    assert result["interface_smoke"]["native_effect_decisions"] == 120
    assert result["task_success"] is False
    assert result["interface_smoke"]["functional_passed"] is True


def test_smoke_valid_failed_episode_resets_and_continues(tmp_path):
    core = SmokeCore(physical_terminal_every=12)
    result = cli._evaluation(core, smoke_args(tmp_path), contract={})
    assert core.resets > 2
    assert result["policy_decisions"] == 128
    assert result["interface_smoke"]["functional_passed"] is True


@pytest.mark.parametrize("options", [{"missing_audit": True}, {"quantized": True}])
def test_smoke_rejects_missing_audit_or_floatquant_zero(tmp_path, options):
    with pytest.raises(RuntimeError, match="native target"):
        cli._evaluation(SmokeCore(**options), smoke_args(tmp_path), contract={})


def test_initial_checkpoint_prevents_silent_reinitialization_before_app(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(cli, "OUTPUT_ROOT", tmp_path / "outputs")
    initial = tmp_path / "outputs/checkpoints/history/checkpoint_initial_semantic.pt"
    initial.parent.mkdir(parents=True)
    initial.write_bytes(b"immutable-initial")
    args = cli.parser().parse_args(["train", "--run-dir", str(tmp_path / "runs/train/first"),
                                   "--expected-head", "a" * 40, "--decisions", "128"])
    with pytest.raises(ValueError, match="reinitializing"):
        cli.validate_request(args)


def test_preflight_records_installed_versions_without_importing_app(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RUNS_ROOT", tmp_path / "runs")
    versions = cli.local_versions()
    assert versions["distributions"]["rsl-rl-lib"] == "5.0.1"
    monkeypatch.setattr(cli, "runtime_contract", lambda **kwargs: {"versions": versions})
    monkeypatch.setattr(cli, "dispatch_live", lambda *args: pytest.fail("preflight started Isaac"))
    directory = tmp_path / "runs/preflight"
    assert cli.main(["preflight", "--run-dir", str(directory), "--expected-head", "a" * 40]) == 0
    result = json.loads((directory / "run_manifest.json").read_text())
    assert result["result"]["isaac_started"] is False


def test_failed_live_interface_preserves_failed_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(cli, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(cli, "runtime_contract", lambda **kwargs: {})
    def fail(*args):
        raise RuntimeError("sensor mismatch")
    monkeypatch.setattr(cli, "dispatch_live", fail)
    directory = tmp_path / "runs/train/failure"
    with pytest.raises(RuntimeError, match="sensor mismatch"):
        cli.main(["train", "--run-dir", str(directory), "--expected-head", "a" * 40, "--decisions", "128"])
    assert json.loads((directory / "run_manifest.json").read_text())["lifecycle"] == "FAILED"
