"""Exact point-cloud geometry tests and bounded CPU-only timing, not Isaac FPS."""
import cProfile
import json
import math
from statistics import median
from time import perf_counter

import numpy as np
import pytest

from wlr50_clean.ppo.semantic_physical_sensing import (
    SemanticColliderGeometry, SemanticSensorReader,
    _body_local_point_array, _world_bounds_from_body_local_array,
)
from wlr50_clean.sensing.contact_classifier import BASE_BODY, WHEEL_BODIES
from wlr50_clean.sensing.geometry import (
    ColliderGeometryCache, _world_bounds_from_body_local_points,
)


BODIES = (BASE_BODY, *WHEEL_BODIES)


def scalar(points, position, quaternion):
    return _world_bounds_from_body_local_points(points,
        position_w_m=position, orientation_wxyz=quaternion)


def vector(array, position, quaternion):
    return _world_bounds_from_body_local_array(array,
        position_w_m=position, orientation_wxyz=quaternion)


def assert_bounds_equal(actual, expected):
    if expected is None:
        assert actual is None
        return
    assert actual is not None
    a = np.array((*actual.minimum_m, *actual.maximum_m))
    b = np.array((*expected.minimum_m, *expected.maximum_m))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(np.signbit(a), np.signbit(b))


@pytest.mark.parametrize('count', [1, 8, 257, 8192])
@pytest.mark.parametrize('seed', [0, 1, 42, 99])
def test_random_current_pose_is_exact_original_float64_arithmetic(count, seed):
    rng = np.random.default_rng(seed)
    points = tuple(map(tuple, rng.uniform(-.3, .3, (count, 3))))
    position = rng.uniform(-2., 2., 3)
    quaternion = rng.normal(size=4) * 5.  # Both helpers normalize nonunit input.
    array = _body_local_point_array(points)
    assert array.dtype == np.float64 and not array.flags.writeable
    assert_bounds_equal(vector(array, position, quaternion), scalar(points, position, quaternion))


@pytest.mark.parametrize('quaternion', [
    (1., 0., 0., 0.), (-1., 0., 0., 0.),
    (math.sqrt(.5), 0., math.sqrt(.5), 0.), (0., 0., 0., 8.),
    (1.e-11, 0., 0., 0.), (1.e200, 0., 0., 0.),
])
def test_axis_rotation_nonunit_and_original_large_quaternion_semantics(quaternion):
    points = ((-.2, .01, -.02), (.2, -.01, .02), (.011, .04, -.05))
    assert_bounds_equal(vector(_body_local_point_array(points), (0., 0., .2), quaternion),
                        scalar(points, (0., 0., .2), quaternion))


@pytest.mark.parametrize('points', [
    (), ((float('nan'), 0., 0.),), ((0., float('inf'), 0.),),
    ((0., 0., float('-inf')),), ((0., 0.),), ((0., 0., 0., 0.),),
])
def test_empty_nonfinite_wrong_width_remain_unverified(points):
    array = _body_local_point_array(points)
    assert array is None
    assert_bounds_equal(vector(array, (0., 0., 0.), (1., 0., 0., 0.)),
                        scalar(points, (0., 0., 0.), (1., 0., 0., 0.)))


@pytest.mark.parametrize('points', [None, 1., [0., 0., 0.], [[[0., 0., 0.]]],
    [(0., 0., 0.), (1., 2.)], [('not-a-number', 0., 0.)]])
def test_malformed_assets_fail_closed_without_geometry(points):
    assert _body_local_point_array(points) is None


@pytest.mark.parametrize('position,quaternion', [
    (None, (1., 0., 0., 0.)), ((0., 0.), (1., 0., 0., 0.)),
    ((float('nan'), 0., 0.), (1., 0., 0., 0.)),
    ((0., 0., 0.), None), ((0., 0., 0.), (0., 0., 0., 0.)),
    ((0., 0., 0.), (1.e-12, 0., 0., 0.)),
    ((0., 0., 0.), (float('inf'), 0., 0., 0.)),
    ((0., 0., 0.), (1., 0., 0.)),
])
def test_invalid_pose_has_original_failure_semantics(position, quaternion):
    points = ((.1, .2, .3),)
    assert_bounds_equal(vector(_body_local_point_array(points), position, quaternion),
                        scalar(points, position, quaternion))


@pytest.mark.parametrize('points,position,valid', [
    (((999.999, 0., 0.),), (0., 0., 0.), True),
    (((1000., 0., 0.),), (0., 0., 0.), False),
    (((-1000., 0., 0.),), (0., 0., 0.), False),
    (((1001., 0., 0.),), (-2., 0., 0.), True),  # Bound is world, not local magnitude.
    (((999., 0., 0.),), (1., 0., 0.), False),
    (((-999., 0., 0.),), (-1., 0., 0.), False),
    (((1.e308, 1.e308, 1.e308),), (0., 0., 0.), False),
])
def test_exact_world_magnitude_boundary_and_overflow(points, position, valid):
    q = (.5, .5, .5, .5) if points[0][0] > 1.e100 else (1., 0., 0., 0.)
    actual = vector(_body_local_point_array(points), position, q)
    assert (actual is not None) == valid
    assert_bounds_equal(actual, scalar(points, position, q))


@pytest.mark.parametrize('sign', [1., -1.])
def test_signed_zero_reduction_ties_preserve_first_point(sign):
    points = ((math.copysign(0., sign), 0., -0.), (math.copysign(0., -sign), -0., 0.))
    for q in [(1., 0., 0., 0.), (-1., -0., 0., -0.)]:
        assert_bounds_equal(vector(_body_local_point_array(points), (-0., -0., -0.), q),
                            scalar(points, (-0., -0., -0.), q))


class PointProvider:
    def __init__(self, points):
        self._body_local_points = {body: points for body in BODIES}
        self._collider_paths = {body: (body + '/actual_mesh',) for body in BODIES}

    def collision_bounds(self, *args, **kwargs):
        raise AssertionError('no second traversal or scalar transform on warmed assets')


def poses():
    return {body: (0., 0., .3) for body in BODIES}, {body: (1., 0., 0., 0.) for body in BODIES}


def test_cache_only_asset_arrays_each_current_pose_changes_real_bounds_and_count():
    points = ((-.2, -.01, -.02), (.2, .01, .02))
    provider = PointProvider(points)
    legacy = ColliderGeometryCache(provider)
    geometry = SemanticColliderGeometry(legacy)
    p, q = poses()
    first = geometry.sample(p, q)
    cached = {body: record[1] for body, record in geometry._point_arrays.items()}
    q = {body: (math.sqrt(.5), 0., math.sqrt(.5), 0.) for body in BODIES}
    p[BASE_BODY] = (.1, .2, .4)
    second = geometry.sample(p, q)
    for body in BODIES:
        # sample normalizes before the helper, just as the original sample did.
        from wlr50_clean.sensing.geometry import _optional_quat
        assert_bounds_equal(second.body_bounds_w_m[body], scalar(points, p[body], _optional_quat(q[body])))
        assert geometry._point_arrays[body][1] is cached[body]
    assert first.body_bounds_w_m != second.body_bounds_w_m
    assert geometry.last_point_counts == dict.fromkeys(BODIES, 2)
    assert math.isfinite(geometry.last_sample_wall_s) and geometry.last_sample_wall_s >= 0.
    assert not legacy._shapes
    assert all(value is points for value in provider._body_local_points.values())


def test_replaced_asset_refreshes_cache_and_mutable_assets_revalidate_nonfinite():
    provider = PointProvider(((0., 0., 0.), (.1, .1, .1)))
    geometry = SemanticColliderGeometry(ColliderGeometryCache(provider))
    p, q = poses()
    geometry.sample(p, q)
    old = geometry._point_arrays[BASE_BODY][1]
    provider._body_local_points[BASE_BODY] = ((-.2, 0., 0.), (.2, 0., 0.))
    assert geometry.sample(p, q).body_bounds_w_m[BASE_BODY].minimum_m[0] == -.2
    assert geometry._point_arrays[BASE_BODY][1] is not old
    mutable = np.array([[0., 0., 0.], [.3, .1, .1]])
    provider._body_local_points[BASE_BODY] = mutable
    assert BASE_BODY in geometry.sample(p, q).body_bounds_w_m
    assert BASE_BODY not in geometry._point_arrays
    mutable[0, 0] = np.nan
    bad = geometry.sample(p, q)
    assert BASE_BODY not in bad.body_bounds_w_m
    assert geometry.last_point_counts[BASE_BODY] is None and bad.quality
    provider._body_local_points[BASE_BODY] = ()
    assert BASE_BODY not in geometry.sample(p, q).body_bounds_w_m


def test_arrays_do_not_alias_mutable_provider_or_permit_writes():
    original = np.array([[0., 0., 0.], [1., 2., 3.]])
    array = _body_local_point_array(original)
    assert not np.shares_memory(array, original)
    original[0] = np.nan
    assert np.isfinite(array).all()
    with pytest.raises(ValueError):
        array[0, 0] = 2.


def test_new_diagnostics_describe_same_atomic_read_and_do_not_alias_counts():
    from test_sensing_stack import _fake_adapter, _FakeContactBackend
    reader = SemanticSensorReader(_fake_adapter(), contact_backend=_FakeContactBackend(),
        geometry_backend=ColliderGeometryCache(PointProvider(((0., 0., 0.), (.1, .1, .1)))))
    raw = reader.read(physics_tick=10, simulation_time_s=1., commanded_full12=(2.,)*8+(.4,)*4)
    assert raw.geometry_point_counts == dict.fromkeys(BODIES, 2)
    assert raw.geometry_sample_wall_s == reader.geometry_backend.last_sample_wall_s
    assert raw.geometry_sample_wall_s >= 0.
    reader.geometry_backend.last_point_counts[BASE_BODY] = 999
    assert raw.geometry_point_counts[BASE_BODY] == 2


def test_bounded_cpu_microbenchmark_reports_not_enforces_speed(record_property):
    """Synthetic isolated hot path, not an end-to-end simulator forecast."""
    rng = np.random.default_rng(7301)
    cases = []
    for count in (32, 4096, 65536):
        points = tuple(map(tuple, rng.uniform(-.3, .3, (count, 3))))
        poses_to_test = [(rng.uniform(-1., 1., 3), rng.normal(size=4)) for _ in range(3)]
        start = perf_counter()
        array = _body_local_point_array(points)
        conversion_s = perf_counter() - start
        scalar_times, vector_times = [], []
        for p, q in poses_to_test:
            start = perf_counter(); expected = scalar(points, p, q)
            scalar_times.append(perf_counter()-start)
            start = perf_counter(); actual = vector(array, p, q)
            vector_times.append(perf_counter()-start)
            assert_bounds_equal(actual, expected)
        row = dict(points=count, poses=3, cache_conversion_s=conversion_s,
            scalar_median_s=median(scalar_times), vector_median_s=median(vector_times),
            vector_over_scalar=median(vector_times)/median(scalar_times))
        cases.append(row)
    profile_counts = {}
    for name, call in [('scalar', lambda: scalar(points, *poses_to_test[0])),
                       ('vector', lambda: vector(array, *poses_to_test[0]))]:
        profiler = cProfile.Profile(); profiler.runcall(call)
        stats = profiler.getstats()
        profile_counts[name] = dict(total_calls=sum(v.callcount for v in stats),
            quat_rotate_calls=sum(v.callcount for v in stats
                if hasattr(v.code, 'co_name') and v.code.co_name == '_quat_rotate'))
    assert profile_counts['scalar']['quat_rotate_calls'] == 65536
    assert profile_counts['vector']['quat_rotate_calls'] == 0
    result = dict(scope='single_body_synthetic_cached_point_cloud_CPU_only',
        exact_float64_aabb_equal=True, cases=cases, profile_counts=profile_counts,
        excludes='USD discovery, contacts, policy, physics, GPU, disk and end-to-end throughput')
    record_property('geometry_cpu_microbenchmark', json.dumps(result, separators=(',', ':')))
    print('\nGEOMETRY_CPU_MICROBENCHMARK=' + json.dumps(result, separators=(',', ':')))
