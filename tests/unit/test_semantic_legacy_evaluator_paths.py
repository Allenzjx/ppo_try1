"""Version-selectable common evaluation never changes the legacy controller."""
from pathlib import Path

from wlr50_clean.ppo.semantic_legacy_evaluation import PhysicalEvaluationRecorder
from wlr50_clean.ppo.semantic_metrics import SemanticMetricsAccumulator
from wlr50_clean.ppo.semantic_supervisor import TaskEvaluator
from test_semantic_legacy_evaluation import audited_frame, PROJECTION


ROOT = Path(__file__).resolve().parents[2]


def test_explicit_v3_common_paths_construct_real_evaluator_and_metrics(tmp_path):
    config = ROOT / "configs/ppo_semantic_v3"
    recorder = PhysicalEvaluationRecorder(tmp_path,
        task_spec_path=config / "stage_task_spec.yaml",
        quality_score_path=config / "quality_score.yaml")
    try:
        assert recorder.evaluator.spec == TaskEvaluator(config / "stage_task_spec.yaml").spec
        reference_metrics = SemanticMetricsAccumulator(config / "quality_score.yaml")
        assert recorder.metrics.config == reference_metrics.config
        first, second = audited_frame(0), audited_frame(1, controller_success=True)
        recorder.start(first)
        recorder.observe(first, second, PROJECTION)
        assert recorder.summary()["task_success"] is False
    finally:
        recorder.close()


def test_omitted_common_paths_keep_original_defaults(tmp_path):
    recorder = PhysicalEvaluationRecorder(tmp_path)
    try:
        assert recorder.evaluator.spec == TaskEvaluator().spec
        assert recorder.metrics.config == SemanticMetricsAccumulator().config
    finally:
        recorder.close()
