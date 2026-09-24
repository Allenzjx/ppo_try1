"""NOTEXECUTED in Isaac: isolated-process, explicitly labelled diagnostic wrapper.

Only the supplied module's pure headroom function is temporarily wrapped.
It still calls the captured original projector, never touches a mapper, tensor,
actuator, policy model or source file. Root must review wiring before any run.
"""
from contextlib import contextmanager
from copy import deepcopy
import threading

from target_plan import OneShotRRDirection, SELECTED, qualified_carry


class IsolatedHeadroomProbe:
    def __init__(self, original_projector, *, read_committed_adapter_tick,
                 diagnostic_only, training_allowed=False, planner=None):
        if diagnostic_only is not True or training_allowed is not False:
            raise ValueError("explicit zero-PPO-credit diagnostic process required")
        self.original = original_projector
        self.read_tick = read_committed_adapter_tick
        self.planner = planner or OneShotRRDirection()
        self.pending = None
        self.plan_calls = 0
        self.replay_calls = 0
        self.thread_id = threading.get_ident()

    def arm_from_observer(self, context):
        """Freeze t-1 current evaluator/ACK/source owner state for dispatch t.

        Caller uses one backend tick observer (plus initial seed after reset),
        not a separately replayed nominal. Native N is NOT supplied here: the
        real headroom call will provide it after the unique mapper advance.
        """
        if threading.get_ident() != self.thread_id:
            raise RuntimeError("diagnostic wrapper is single-thread only")
        ctx = deepcopy(context)
        tick = ctx["dispatch_tick"]
        if ctx["previous_ack_tick"] != tick - 1 or self.read_tick() != tick - 1:
            raise ValueError("observer context is not the adjacent committed ACK")
        if self.pending is not None and tick != self.pending["context"]["dispatch_tick"] + 1:
            raise ValueError("one observer context per contiguous physical dispatch required")
        self.pending = dict(context=ctx, prepared=False, native=None, controller=None,
                            actual_request=None, desired=None, plan_receipt=None)

    def bind_decision_proposal(self, raw_full12, log_probability):
        """Bind the NEW15Hz policy receipt before its first physical dispatch.

        A previous tick observer may have armed geometry before this decision's
        actor call. Binding only provenance avoids logging that older proposal.
        The next seven observer contexts must carry the same decision receipt.
        """
        if self.pending is None or self.pending["prepared"]:
            raise ValueError("bind proposal only before the armed tick executes")
        if len(raw_full12) != 12:
            raise ValueError("Full12 decision proposal required")
        self.pending["context"]["policy_raw_full12"] = tuple(raw_full12)
        self.pending["context"]["policy_log_probability"] = log_probability

    def project(self, native_full12, controller_bias_full12, projected_residual_full12):
        if threading.get_ident() != self.thread_id:
            raise RuntimeError("diagnostic headroom used from another thread")
        native, controller, incoming = map(tuple,
            (native_full12, controller_bias_full12, projected_residual_full12))
        original = self.original(native_full12=native, controller_bias_full12=controller,
                                 projected_residual_full12=incoming)
        pending = self.pending
        if pending is None:  # Settling/prefix setup is exactly original.
            return original
        ctx = pending["context"]
        # Exact original prefix/settle behavior, including the adapter's possible
        # already-written zero-residual fast path. An ACTIVE probe must still
        # visit planner once to release when its current permission disappears.
        if self.planner.state == "RELEASED" or (self.planner.state == "WAIT" and
                not qualified_carry(ctx["evaluation"], ctx["phase"], ctx["minimum_other_supports"],
                                    ctx["force_noise_floor_n"])):
            return original
        tick = ctx["dispatch_tick"]
        committed = self.read_tick()
        if committed not in (tick - 1, tick):
            raise ValueError("headroom call is outside the armed same-tick window")
        if not pending["prepared"]:
            if committed != tick - 1:
                raise ValueError("audit replay cannot create or advance a diagnostic plan")
            real_context = deepcopy(ctx)
            real_context.update(same_tick_mapped_tick=tick,
                same_tick_mapped_nominal_full12=native,
                controller_bias_full12=controller,
                policy_projected_residual_full12=incoming,
                headroom_residual_intervals_servo_deg=original["policy_residual_intervals_servo_deg"],
                servo_hard_limits_deg=original["servo_hard_limits_deg"])
            injected, receipt = self.planner.plan(real_context)
            self.plan_calls += 1
            pending.update(prepared=True, native=native, controller=controller,
                actual_request=incoming, plan_receipt=deepcopy(receipt),
                desired=(None if injected is None else
                         deepcopy(receipt["desired_final_servo_deg"])))
        else:
            self.replay_calls += 1
            if native != pending["native"] or controller != pending["controller"]:
                raise ValueError("same-tick mapped native/controller changed during audit replay")
            if committed == tick - 1 and incoming != pending["actual_request"]:
                raise ValueError("second distinct pre-write policy request for one physical dispatch")
            if committed == tick and incoming not in (pending["actual_request"], (0.,) * 12):
                raise ValueError("unexpected post-write branch; only actual/zero-policy audit is supported")
        if pending["desired"] is None:
            return original
        # Recompute from the immutable planned FINAL, not from returned ACK,
        # cached headroom results, or a second mutable planner/mapper advance.
        # Crucially the audit's zero-policy branch retains the SAME exogenous
        # RR intervention. Thus a zero baseline action has no ambiguous routing.
        injected = list(incoming)
        for i in SELECTED:
            injected[i] = pending["desired"][i] - (native[i] + controller[i])
        result = self.original(native_full12=native, controller_bias_full12=controller,
                               projected_residual_full12=tuple(injected))
        if any(result["effective_policy_residual_full12"][i] != injected[i] for i in SELECTED):
            raise ValueError("original headroom cannot realize the exact diagnostic target")
        if any(result["effective_combined_post_mapper_bias_full12"][i]
               != original["effective_combined_post_mapper_bias_full12"][i]
               for i in range(12) if i not in SELECTED):
            raise ValueError("diagnostic changed an unselected channel")
        result["explicit_RR_direction_diagnostic"] = {
            "schema": "post_mapper_RR_INTERVENTION.v1", "dispatch_tick": tick,
            "selected_indices": list(SELECTED),
            "desired_final_servo_deg": deepcopy(pending["desired"]),
            "original_policy_raw_full12": list(ctx["policy_raw_full12"]),
            "original_policy_log_probability": ctx.get("policy_log_probability"),
            "actual_original_filtered_policy_request_full12": list(pending["actual_request"]),
            "incoming_branch_request_full12": list(incoming),
            "diagnostic_injected_request_full12": list(injected),
            "requested_policy_field_now_describes": "explicit_nonpolicy_diagnostic_injected_request",
            "raw_policy_sample_was_executed_unmodified": False,
            "unselected_ten_match_original_projector": True,
            "counterfactual_semantics": "same_exogenous_RR_targets_remove_only_current_policy_request",
            "new_PPO_decisions": 0, "new_PPO_updates": 0,
            "original_projector_headroom_and_downstream_clamp_slew_preserved": True}
        return result

    @contextmanager
    def installed_on(self, module):
        """Install only in a dedicated diagnostic process, restore even on error.

        Tests use a private namespace, never mutate the real production module.
        Runtime metadata must record this in-memory override despite unchanged
        disk hashes. Loading compatible frozen weights is not runtime equivalence.
        """
        if module.project_semantic_servo_headroom is not self.original:
            raise ValueError("headroom already replaced; nested overrides forbidden")
        bound = self.project
        module.project_semantic_servo_headroom = bound
        try:
            yield self
        finally:
            changed = module.project_semantic_servo_headroom is not bound
            module.project_semantic_servo_headroom = self.original
            if changed:
                raise RuntimeError("another code path replaced the scoped diagnostic wrapper")
