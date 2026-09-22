"""Run protected control/task regressions against the complete candidate overlay."""
import candidate_bootstrap as runtime
import pytest

FILES = [
    'test_semantic_nominal_command_history.py',
    'test_semantic_front_wheel_authority.py',
    'test_semantic_p06_wheel_tail.py',
    'test_semantic_p05_capture_handoff_v2.py',
    'test_semantic_p13_final_stop_owner.py',
    'test_p09_free_air_lift_v3.py',
    'test_p09_free_air_carry_source_v1.py',
    'test_semantic_request_history_actor.py',
]
if __name__ == '__main__':
    assert all(row['candidate'] for row in runtime.provenance().values())
    raise SystemExit(pytest.main([*(str(runtime.ROOT/'tests/unit'/name) for name in FILES),
        '-q', '--disable-warnings', '--tb=short',
        '--junitxml=' + str(runtime.HERE/'protected_control_regressions.xml')]))
