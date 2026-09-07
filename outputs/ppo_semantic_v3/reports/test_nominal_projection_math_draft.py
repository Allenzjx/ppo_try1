"""Unexecuted draft tests; no runtime imports and no Isaac dependency.

Only run after the parent explicitly clears the live barrier. These are algebraic
counterexamples, not evidence that the physical robot obeys the first-order model.
"""

import importlib.util
import math
from pathlib import Path
import sys

import pytest


_PATH = Path(__file__).with_name("nominal_projection_math_draft.py")
_SPEC = importlib.util.spec_from_file_location("nominal_projection_math_draft", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
project = _MODULE.project_nominal_downward


def request(**overrides):
    values = dict(
        nominal_delta_rad=(0.2, -0.3),
        jacobian_x_m_per_rad=(1.0, 0.0),
        jacobian_z_m_per_rad=(0.0, 1.0),
        delta_lower_rad=(-1.0, -1.0),
        delta_upper_rad=(1.0, 1.0),
        available_descent_m=0.1,
        place_xy=False,
    )
    values.update(overrides)
    return project(**values)


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def assert_verified(result):
    assert result.feasible
    assert result.corrected_delta_rad is not None
    assert result.proof["mathematical_contract_verified"] is True
    assert result.proof["physical_motion_guaranteed"] is False
    assert result.proof["residual_constraint_applied"] is False
    assert result.proof["box_satisfied"] is True
    assert result.proof["requested_x_direction_not_reversed"] is True


def test_negative_knee_high_clearance_forward_is_identity_not_sign_gated():
    # Negative second-joint motion is useful forward carry and is not forbidden.
    result = request(
        nominal_delta_rad=(0.0, -0.3),
        jacobian_x_m_per_rad=(0.0, -0.4),
        jacobian_z_m_per_rad=(0.0, 0.05),
        available_descent_m=max(0.0, 0.1 - 0.015),
    )
    assert_verified(result)
    assert result.status == "identity_within_descent_allowance"
    assert result.corrected_delta_rad == (0.0, -0.3)
    assert result.proof["corrected_linear_x_m"] == pytest.approx(0.12)


def test_already_place_xy_allows_nominal_descent_without_projection():
    result = request(place_xy=True, available_descent_m=0.0)
    assert_verified(result)
    assert result.status == "identity_place_xy"
    assert result.corrected_delta_rad == (0.2, -0.3)
    assert result.proof["downward_constraint_enforced"] is False
    assert result.proof["downward_halfspace_satisfied"] is False


def test_vertical_projection_preserves_forward_and_minimizes_joint_change():
    result = request()
    assert_verified(result)
    assert result.corrected_delta_rad == pytest.approx((0.2, -0.1))
    assert result.nominal_correction_rad == pytest.approx((0.0, 0.2))
    assert result.status == "projected_exact_forward"
    assert result.proof["joint_correction_norm_rad"] == pytest.approx(0.2)


def test_exhausted_descent_allows_forward_and_other_joint_not_whole_leg_hold():
    result = request(available_descent_m=0.0)
    assert_verified(result)
    assert result.corrected_delta_rad == pytest.approx((0.2, 0.0))
    assert result.corrected_delta_rad != (0.0, 0.0)


def test_preserving_forward_is_lexicographic_not_unconstrained_nearest_point():
    result = request(
        nominal_delta_rad=(0.4, -0.4),
        jacobian_x_m_per_rad=(1.0, 1.0),
        available_descent_m=0.0,
    )
    # Euclidean projection on z alone would be (.4, 0), changing original x=0.
    # Keeping exact x is feasible and therefore preferred, even at greater norm.
    assert_verified(result)
    assert result.corrected_delta_rad == pytest.approx((0.0, 0.0))
    assert result.proof["forward_priority"] == "exact_original_x"


def test_collinear_opposing_rows_reduce_forward_without_pseudoinverse():
    result = request(
        nominal_delta_rad=(0.8, 0.25),
        jacobian_x_m_per_rad=(1.0, 0.0),
        jacobian_z_m_per_rad=(-1.0, 0.0),
        available_descent_m=0.2,
    )
    assert_verified(result)
    assert result.status == "projected_relaxed_forward"
    assert result.corrected_delta_rad == pytest.approx((0.2, 0.25))
    assert result.proof["corrected_linear_x_m"] >= 0.0


def test_zero_allowance_collinear_conflict_can_reduce_forward_to_zero():
    result = request(
        nominal_delta_rad=(0.8, 0.25),
        jacobian_z_m_per_rad=(-1.0, 0.0),
        available_descent_m=0.0,
    )
    assert_verified(result)
    assert result.corrected_delta_rad == pytest.approx((0.0, 0.25))


def test_box_can_require_more_forward_and_chooses_the_smallest_feasible_increase():
    result = request(
        nominal_delta_rad=(0.1, -0.5),
        jacobian_z_m_per_rad=(1.0, 1.0),
        delta_lower_rad=(0.0, -0.5),
        delta_upper_rad=(1.0, -0.5),
        available_descent_m=0.0,
    )
    assert_verified(result)
    assert result.status == "projected_relaxed_forward"
    assert result.corrected_delta_rad == pytest.approx((0.5, -0.5))
    assert result.proof["requested_x_magnitude_increase_m"] == pytest.approx(0.4)
    assert result.proof["no_amplification_guarantee"] is False


def test_negative_requested_x_is_not_reversed_when_every_safe_solution_is_positive():
    result = request(
        nominal_delta_rad=(0.2, -0.3),
        jacobian_x_m_per_rad=(1.0, 1.0),
        delta_lower_rad=(0.2, -0.3),
        delta_upper_rad=(0.2, 0.2),
        available_descent_m=0.0,
    )
    # Original x is negative; every downward-feasible point has positive x.
    assert not result.feasible
    assert result.status == "infeasible_nonreversal"
    assert result.corrected_delta_rad is None
    assert result.nominal_correction_rad is None


def test_out_of_box_nominal_is_not_silently_box_projected_or_sent_as_identity():
    with pytest.raises(ValueError, match="nominal must already obey"):
        request(
            nominal_delta_rad=(0.2, -0.3),
            delta_lower_rad=(-0.5, 0.1),
            delta_upper_rad=(0.5, 0.1),
            place_xy=True,
        )


def test_all_safe_box_points_require_reverse_and_none_is_silently_sent():
    result = request(
        nominal_delta_rad=(0.2, 0.1),
        jacobian_z_m_per_rad=(-1.0, -1.0),
        delta_lower_rad=(-0.5, 0.1),
        delta_upper_rad=(0.5, 0.1),
        available_descent_m=0.0,
    )
    assert not result.feasible
    assert result.status == "infeasible_nonreversal"
    assert result.corrected_delta_rad is None


def test_box_downward_conflict_is_explicit_not_zero_or_original_fallback():
    result = request(
        nominal_delta_rad=(0.2, -0.3),
        delta_lower_rad=(0.1, -0.4),
        delta_upper_rad=(0.3, -0.2),
        available_descent_m=0.1,
    )
    assert not result.feasible
    assert result.status == "infeasible_box_downward"
    assert result.corrected_delta_rad is None
    assert result.nominal_correction_rad is None
    assert result.proof["mathematical_contract_verified"] is False


def test_zero_horizontal_jacobian_uses_exact_euclidean_vertical_projection():
    result = request(jacobian_x_m_per_rad=(0.0, 0.0))
    assert_verified(result)
    assert result.corrected_delta_rad == pytest.approx((0.2, -0.1))


def test_zero_vertical_jacobian_is_identity_even_with_zero_allowance():
    result = request(jacobian_z_m_per_rad=(0.0, 0.0), available_descent_m=0.0)
    assert_verified(result)
    assert result.corrected_delta_rad == (0.2, -0.3)


def test_swapping_and_sign_changing_joint_coordinates_keeps_same_world_solution():
    original = request()
    transformed = request(
        nominal_delta_rad=(0.3, 0.2),
        jacobian_x_m_per_rad=(0.0, 1.0),
        jacobian_z_m_per_rad=(-1.0, 0.0),
    )
    assert_verified(transformed)
    q = original.corrected_delta_rad
    assert transformed.corrected_delta_rad == pytest.approx((-q[1], q[0]))
    assert transformed.proof["corrected_linear_x_m"] == pytest.approx(original.proof["corrected_linear_x_m"])
    assert transformed.proof["corrected_linear_z_m"] == pytest.approx(original.proof["corrected_linear_z_m"])


def test_nearly_parallel_rows_produce_bounded_verified_target():
    result = request(
        nominal_delta_rad=(0.8, 0.25),
        jacobian_x_m_per_rad=(1.0, 1e-14),
        jacobian_z_m_per_rad=(-1.0, 0.0),
        available_descent_m=0.2,
    )
    assert_verified(result)
    assert result.corrected_delta_rad[0] == pytest.approx(0.2)
    assert all(-1.0 <= q <= 1.0 for q in result.corrected_delta_rad)


@pytest.mark.parametrize("allowance", [0.0, 0.1, 0.3])
def test_exact_boundary_is_identity(allowance):
    result = request(nominal_delta_rad=(0.2, -allowance), available_descent_m=allowance)
    assert_verified(result)
    assert result.status == "identity_within_descent_allowance"
    assert result.nominal_correction_rad == (0.0, 0.0)


def test_positive_vertical_motion_is_identity_without_place_xy():
    result = request(nominal_delta_rad=(0.2, 0.3), available_descent_m=0.0)
    assert_verified(result)
    assert result.corrected_delta_rad == (0.2, 0.3)


def test_degenerate_line_box_and_single_point_identity():
    result = request(delta_lower_rad=(0.2, -0.3), delta_upper_rad=(0.2, 0.0))
    assert_verified(result)
    assert result.corrected_delta_rad == pytest.approx((0.2, -0.1))
    point = request(
        nominal_delta_rad=(0.2, 0.0),
        delta_lower_rad=(0.2, 0.0), delta_upper_rad=(0.2, 0.0),
        available_descent_m=0.0,
    )
    assert_verified(point)
    assert point.corrected_delta_rad == (0.2, 0.0)


def test_second_call_is_identity_no_history_or_repeated_projection_credit():
    first = request()
    second = request(nominal_delta_rad=first.corrected_delta_rad)
    assert_verified(second)
    assert second.corrected_delta_rad == first.corrected_delta_rad
    assert second.nominal_correction_rad == (0.0, 0.0)


@pytest.mark.parametrize("override", [
    {"nominal_delta_rad": (float("nan"), 0.0)},
    {"jacobian_x_m_per_rad": (float("inf"), 0.0)},
    {"jacobian_z_m_per_rad": (0.0, float("nan"))},
    {"delta_lower_rad": (float("-inf"), -1.0)},
    {"delta_upper_rad": (1.0, float("inf"))},
    {"available_descent_m": float("nan")},
    {"available_descent_m": -0.1},
    {"available_descent_m": True},
    {"place_xy": 1},
    {"nominal_delta_rad": (False, 0.0)},
    {"nominal_delta_rad": (0.0,)},
    {"jacobian_x_m_per_rad": (0.0, 0.0, 0.0)},
    {"jacobian_z_m_per_rad": "01"},
    {"delta_lower_rad": (1.0, -1.0), "delta_upper_rad": (-1.0, 1.0)},
    {"delta_lower_rad": (0.3, -1.0)},
])
def test_bad_inputs_fail_before_identity_or_projection(override):
    with pytest.raises(ValueError):
        request(**override)
    # Place XY is not a route around finite/box validation.
    if "place_xy" not in override:
        with pytest.raises(ValueError):
            request(**{**override, "place_xy": True})


def test_finite_but_unrepresentable_intermediate_never_emits_target():
    with pytest.raises(_MODULE.ProjectionNumericalError):
        request(
            nominal_delta_rad=(1e308, 1e308),
            jacobian_x_m_per_rad=(2.0, 0.0),
            delta_lower_rad=(-1e308, -1e308),
            delta_upper_rad=(1e308, 1e308),
            place_xy=True,
        )


def test_no_extra_inputs_can_restrict_residual_or_use_phase_or_pose():
    for extra in ("residual_delta_rad", "phase_id", "historical_peak_pose"):
        with pytest.raises(TypeError):
            request(**{extra: 0})


@pytest.mark.parametrize("vertical_row", [(0.0, 1.0), (1.0, 1.0), (-1.0, 0.2)])
def test_output_box_and_vertical_halfspace_against_dense_feasible_candidates(vertical_row):
    # Jx=0 makes the second objective an ordinary Euclidean projection. A grid
    # is only an independent lower-resolution optimality check, not the solver.
    nominal = (0.2, -0.3)
    result = request(jacobian_x_m_per_rad=(0.0, 0.0), jacobian_z_m_per_rad=vertical_row)
    assert_verified(result)
    output = result.corrected_delta_rad
    assert dot(vertical_row, output) >= -0.1 - 1e-12
    distance = math.dist(output, nominal)
    for i in range(-20, 21):
        for j in range(-20, 21):
            point = (i / 20.0, j / 20.0)
            if dot(vertical_row, point) >= -0.1:
                assert distance <= math.dist(point, nominal) + 1e-12
