"""Pinned return-estimator identities; no tensors, mutable history or simulation.

The current run reads gamma from its versioned reward configuration. Constants
here identify the only supported historical parameter combinations; they are
not a second independently tunable runtime discount.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

LEGACY_RETURN_PROFILE = "legacy_gamma_0995_lambda_095_v1"
RETURN_PROFILE = "v3_gamma_09985_lambda_099_v1"
RUNNER_PROFILE_KEY = "semantic_return_profile"
COLLECTION_PROFILE_KEY = "semantic_collection_profile"
COLLECTION_512 = "n1_rear_owner439_collection512_v1"


def runner_collection_length(config: Mapping[str, Any]) -> int:
    """Validate collection independently; absent marker is historical128 only."""
    marker = config.get(COLLECTION_PROFILE_KEY)
    length = config.get("num_steps_per_env")
    expected = 128 if marker is None else 512
    if (type(length) is not int or length != expected
            or (COLLECTION_PROFILE_KEY in config and marker != COLLECTION_512)):
        raise ValueError("runner collection length differs from its explicit marker")
    return length


def same_return_estimator(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Collection storage is not a PBRS parameter; still require all math keys."""
    return all(key in left and key in right and left[key] == right[key]
               for key in ("version", "gamma", "lambda"))


def profile_parameters(version: str, *, semantic_version: str | None = None) -> dict[str, Any]:
    if semantic_version not in (None, "v2", "v3"):
        raise ValueError("unsupported semantic return-profile runtime")
    if version == LEGACY_RETURN_PROFILE:
        gamma, lam = .995, .95
    elif version == RETURN_PROFILE and semantic_version != "v2":
        gamma, lam = .9985, .99
    else:
        raise ValueError("unsupported semantic return profile")
    return {"version": version, "gamma": gamma, "lambda": lam, "rollout_length": 128}


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite non-boolean number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite non-boolean number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite non-boolean number")
    return result


def reward_return_profile(values: Mapping[str, Any], *, semantic_version: str | None = None) -> dict[str, Any]:
    """Resolve only a complete known reward profile, never infer new from gamma."""
    if not isinstance(values, Mapping):
        raise ValueError("semantic reward return profile requires a mapping")
    version = values.get("return_profile", LEGACY_RETURN_PROFILE)
    if "return_profile" in values and version != RETURN_PROFILE:
        raise ValueError("explicit reward return profile must identify the new v3 profile")
    result = profile_parameters(version, semantic_version=semantic_version)
    actual = _number(values.get("gamma"), "reward gamma")
    if actual != result["gamma"]:
        raise ValueError("reward gamma differs from its explicitly identified return profile")
    # Retain the value read from the reward configuration as the runtime source.
    return {**result, "gamma": actual}


def runner_return_profile(config: Mapping[str, Any], *, semantic_version: str) -> dict[str, Any]:
    """Validate marker plus gamma/lambda/rollout; callers validate other fields."""
    if not isinstance(config, Mapping) or not isinstance(config.get("algorithm"), Mapping):
        raise ValueError("semantic runner return profile requires a complete algorithm mapping")
    version = config.get(RUNNER_PROFILE_KEY, LEGACY_RETURN_PROFILE)
    if RUNNER_PROFILE_KEY in config and version != RETURN_PROFILE:
        raise ValueError("explicit runner return profile must identify the new v3 profile")
    result = profile_parameters(version, semantic_version=semantic_version)
    algorithm = config["algorithm"]
    if (_number(algorithm.get("gamma"), "PPO gamma") != result["gamma"]
            or _number(algorithm.get("lam"), "GAE lambda") != result["lambda"]):
        raise ValueError("runner gamma/lambda differs from its return-profile marker")
    length = runner_collection_length(config)
    if length == 512 and (semantic_version != "v3" or version != RETURN_PROFILE):
        raise ValueError("collection512 requires the unchanged current v3 return estimator")
    return {**result, "rollout_length": length}
