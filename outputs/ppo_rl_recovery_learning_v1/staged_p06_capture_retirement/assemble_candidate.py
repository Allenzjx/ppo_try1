"""Outputs-only mechanical candidate assembly. Does not import the robot runtime."""
from pathlib import Path
import ast
import difflib
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNTIME = "src/wlr50_clean/ppo/semantic_capture_assist.py"
TEST = "tests/unit/test_semantic_p06_capture_retirement.py"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"source anchor drift: {old!r}")
    return text.replace(old, new, 1)


def main():
    original = (ROOT / RUNTIME).read_text(encoding="utf-8")
    candidate = replace_once(original,
        'CAPTURE_ASSIST_FEEDBACK_REVISION = "hold_to_air_progress_window_v2"\n',
        'CAPTURE_ASSIST_FEEDBACK_REVISION = "hold_to_air_progress_window_v2"\n'
        'CAPTURE_ASSIST_RETIREMENT_REVISION = "p06_placed_air_exhausted_release_v1"\n')
    candidate = replace_once(candidate,
        '    A blocked search retains its target while shared wheel/body actions can\n'
        '    improve geometry; it never loops a nominal-relative decrement.\n',
        '    A blocked initial capture retains its target while shared wheel/body\n'
        '    actions can improve geometry. A previously placed, currently AIR FL\n'
        '    in P06 retires an exhausted search through the existing release blend.\n'
        '    Neither path loops a nominal-relative decrement.\n')
    candidate = replace_once(candidate,
        '                "feedback_revision": CAPTURE_ASSIST_FEEDBACK_REVISION,\n',
        '                "feedback_revision": CAPTURE_ASSIST_FEEDBACK_REVISION,\n'
        '                "retirement_revision": CAPTURE_ASSIST_RETIREMENT_REVISION,\n')
    candidate = replace_once(candidate,
        '        if not window and state["mode"] != 4:\n',
        '        # The placement event does not assert current support. Retire only\n'
        '        # an already exhausted post-placement search, with explicit real\n'
        '        # AIR/no-bearing evidence; never release an initial/pending capture\n'
        '        # or a current TOP/HOLD target. The guard uses existing observed\n'
        '        # phase/history/pair-contact/helper state, not a new hidden timer.\n'
        '        retire_exhausted = bool(\n'
        '            stage == "P06" and safe and placed and not contact\n'
        '            and state["mode"] == 3. and state["blocked_reason"] == 5.\n'
        '            and context.get("current_FL_air") is True\n'
        '            and context.get("current_FL_support") is False)\n'
        '        if retire_exhausted:\n'
        '            previous = tuple(_finite(x, "previous final") for x in previous_final_full12)\n'
        '            if len(previous) != 12:\n'
        '                raise ValueError("capture assist must release from the actual previous Full12 final target")\n'
        '            # Re-anchor once to the actual final dispatch, not a nominal\n'
        '            # recomputation or a pre-clamp assist target. Keep all search,\n'
        '            # contact and nominal/mapper history; mode RELEASED prevents\n'
        '            # later rearming until the normal episode reset.\n'
        '            state.update(mode=4., blocked_reason=0., release_fraction=0.,\n'
        '                         hip_target_deg=previous[0], knee_hold_deg=previous[1])\n'
        '        if not window and state["mode"] != 4:\n')
    candidate = replace_once(candidate,
        '            "placed_FL": bool(history.get("placed", {}).get("FL")),\n',
        '            "placed_FL": bool(history.get("placed", {}).get("FL")),\n'
        '            # Preserve missing/invalid evidence as unknown, never AIR.\n'
        '            "current_FL_air": fl.get("air"),\n'
        '            "current_FL_support": fl.get("support"),\n')
    ast.parse(candidate)
    test_text = (HERE / "test_semantic_p06_capture_retirement.py").read_text(encoding="utf-8")
    ast.parse(test_text)
    output = HERE / "candidate" / RUNTIME
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(candidate, encoding="utf-8", newline="\n")
    difference = list(difflib.unified_diff(original.splitlines(True), candidate.splitlines(True),
                                          fromfile="a/" + RUNTIME, tofile="b/" + RUNTIME))
    test_diff = list(difflib.unified_diff([], test_text.splitlines(True),
                                        fromfile="/dev/null", tofile="b/" + TEST))
    (HERE / "candidate.patch").write_text("".join(difference + test_diff), encoding="utf-8", newline="\n")
    apply_lines = ["*** Begin Patch\n", "*** Update File: " + RUNTIME + "\n"]
    apply_lines += ["@@\n" if line.startswith("@@") else line for line in difference[2:]]
    apply_lines += ["*** Add File: " + TEST + "\n"]
    apply_lines += ["+" + line + "\n" for line in test_text.splitlines()]
    apply_lines += ["*** End Patch\n"]
    (HERE / "candidate.apply_patch").write_text("".join(apply_lines), encoding="utf-8", newline="\n")
    sha = lambda data: hashlib.sha256(data).hexdigest()
    manifest = {
        "status": "isolated_not_applied_not_physically_validated",
        "source_head": "72e63592bdf412d375abfda29b03cf456fbf4f7e",
        "runtime_paths": [RUNTIME], "test_paths": [TEST],
        "source_file_sha256": sha((ROOT / RUNTIME).read_bytes()),
        "source_normalized_text_sha256": sha(original.encode()),
        "candidate_file_sha256": sha(output.read_bytes()),
        "candidate_apply_patch_sha256": sha((HERE / "candidate.apply_patch").read_bytes()),
        "test_sha256": sha(test_text.encode()),
        "numeric_capture_feature_count": 12,
        "observation_dimension_change": 0,
        "policy_action_dimension_change": 0,
        "reward_change": False, "model_loaded": False,
        "controller_contribution": "Retire exhausted post-placement P06 FL ownership using existing 0.75 s final-target blend",
    }
    (HERE / "candidate_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
