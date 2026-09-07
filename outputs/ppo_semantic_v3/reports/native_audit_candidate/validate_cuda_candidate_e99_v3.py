"""UNRUN e99 v3: device-arithmetic candidate, original bounded 680-case harness.

Only after the root explicitly clears the live barrier, use --run-cuda
--ack-exclusive-device --device cuda:0 --warmup 5 --samples 40 --max-seconds 120
--output <fresh .json directly in this report directory>.

This thin revision reuses the SHA-pinned original e99 harness unchanged except
for the candidate path/hash, report revision, and a precisely enumerated
cross-denormal rejection expectation. The failed e99 files/receipt stay intact.
No Torch is imported by this wrapper or while processing --help. All original
exclusive-process, source/dependency, mutation, memory and time checks remain.
No execution, equivalence success, speedup, or production adoption is claimed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS_BASE_SHA = "589ecc15517f9e63f94c07155a9bf4423095c592c72d8eda01734c3ee262b0eb"
V3_CANDIDATE_SHA = "284f7ee524ddc28d296b73310bcc3dbca42e9e5fd4018442b4acebdb80ec5919"
MAPPING_REJECTION = {
    "ok": False,
    "error_type": "ActuatorTargetEffectError",
    "error": "frozen mapping reconstruction differs from actual dispatch",
}

# Exact fixture dictionaries from the failed actual e99 receipt. There is no
# catch-all for subnormals, headroom errors, normal-neighbor cases, or arbitrary
# reference failures. Fixtures are dispatched with host flush=False; only
# their later flush=True audit is expected to reject this mapping mismatch.
CROSS_DENORMAL_CASES = (
    {"name": "subnormal_wheel_1.401298464324817e-45", "channel": 9,
     "residual": 1.401298464324817e-45},
    {"name": "subnormal_wheel_-1.401298464324817e-45", "channel": 9,
     "residual": -1.401298464324817e-45},
    {"name": "subnormal_wheel_1e-40", "channel": 9, "residual": 1e-40},
    {"name": "subnormal_wheel_-1e-40", "channel": 9, "residual": -1e-40},
    {"name": "headroom_subnormal_1.401298464324817e-45", "channel": 9,
     "residual": 1.401298464324817e-45, "headroom": True},
    {"name": "headroom_subnormal_-1.401298464324817e-45", "channel": 9,
     "residual": -1.401298464324817e-45, "headroom": True},
)


def classify_case_v3(case, flush, left, right, json_equal, unchanged, attempted_mutation):
    """Preserve positive/fault rules; add only the named exact rejection rule."""
    # Dictionary equality alone treats bool==1; the original canonical JSON
    # encoder is used here too so a changed fixture type cannot broaden scope.
    import json
    encoded = json.dumps(case, sort_keys=True, allow_nan=False)
    crossed = flush is True and any(
        encoded == json.dumps(expected, sort_keys=True, allow_nan=False)
        for expected in CROSS_DENORMAL_CASES
    )
    expected_rejection = dict(MAPPING_REJECTION) if crossed else None
    if crossed:
        # Both accepting is a failure here, even when their outputs match.
        # Both rejecting a different error is also a failure.
        passed = left == MAPPING_REJECTION and right == MAPPING_REJECTION and json_equal
        expectation = "exact_cross_denormal_mapping_rejection"
    elif case.get("fault"):
        # Original single-fault policy, including the explicitly documented
        # stronger mixed-device rejection (neither side may accept).
        passed = (not left["ok"] and not right["ok"]
                  and (json_equal or case["fault"] == "mixed_device"))
        expectation = "original_strict_fault_rejection"
    else:
        passed = left["ok"] and right["ok"] and json_equal
        expectation = "both_success_with_exact_full_json"
    return {
        "passed": bool(passed and unchanged and not attempted_mutation),
        "expectation": expectation,
        "expected_rejection": expected_rejection,
    }


def _replace_exact(source, old, new, *, count=1):
    if source.count(old) != count:
        raise RuntimeError("Pinned harness adaptation anchor changed; do not repin blindly")
    return source.replace(old, new)


def load_harness_v3():
    """Apply bounded in-memory edits to a pinned report script, never its file."""
    path = HERE / "validate_cuda_candidate_e99.py"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != HARNESS_BASE_SHA:
        raise RuntimeError("Original e99 harness changed; preserve and re-review its receipt")
    source = raw.decode("utf-8").replace("\r\n", "\n")
    source = __doc__.__repr__() + "\n" + source[source.index("from __future__ import annotations"):]
    source = _replace_exact(source,
        'CANDIDATE_SHA = "3b0ad2fd40cc5e9951b326bc23784448c3601f86c1378d5d6487ffe052606601"',
        'CANDIDATE_SHA = ' + repr(V3_CANDIDATE_SHA))
    source = _replace_exact(source, '"actuator_target_effect_candidate_e99.py"',
        '"actuator_target_effect_candidate_e99_v3.py"', count=2)
    old_rule = '''                if case.get("fault"):
                    # Mixed-device rejection is deliberately stricter in the
                    # candidate. All other selected single-fault messages must
                    # match; mutating then raising is never accepted.
                    passed = (not left["ok"] and not right["ok"] and unchanged
                              and (json_equal or case["fault"] == "mixed_device"))
                else:
                    passed = left["ok"] and right["ok"] and json_equal and unchanged
                passed = passed and not attempted_mutation
'''
    new_rule = '''                classification = classify_case_v3(
                    case, flush, left, right, json_equal, unchanged, attempted_mutation)
                passed = classification["passed"]
'''
    source = _replace_exact(source, old_rule, new_rule)
    source = _replace_exact(source,
        '"input_state_unchanged": unchanged, "passed": passed}',
        '"input_state_unchanged": unchanged, "passed": passed,\n'
        '                    "expectation": classification["expectation"],\n'
        '                    "expected_rejection": classification["expected_rejection"]}')
    source = _replace_exact(source, '"unwired_native_audit_cuda_comparison.e99.v2"',
        '"unwired_native_audit_cuda_comparison.e99.v3"')
    source = _replace_exact(source,
        '"reviewed_revision": "e99fde1b3e8366f0ff1d484140b82877df745f0c",',
        '"reviewed_revision": "e99fde1b3e8366f0ff1d484140b82877df745f0c",\n'
        '            "candidate_revision": "e99_v3_device_arithmetic_one_cpu_snapshot",\n'
        '            "harness_base_sha256": ' + repr(HARNESS_BASE_SHA) + ',\n'
        '            "cross_denormal_expected_rejections": CROSS_DENORMAL_CASES,\n'
        '            "cross_denormal_expected_rejection_only_when_host_flush_true": True,')
    namespace = {
        "__name__": "unwired_native_audit_e99_v3_harness",
        # The base harness records the actual wrapper hash, not the old file's
        # hash; it records the separately pinned base hash above as well.
        "__file__": str(Path(__file__).resolve()),
        "classify_case_v3": classify_case_v3,
        "CROSS_DENORMAL_CASES": CROSS_DENORMAL_CASES,
    }
    exec(compile(source, str(path) + "[v3-reviewed-adaptation]", "exec"), namespace)
    return namespace


if __name__ == "__main__":
    raise SystemExit(load_harness_v3()["main"]())
