"""Read and verify the candidate artifact; never admits it to training."""
from pathlib import Path
import hashlib
import json

import numpy as np

EXPECTED_SOURCE = 'a4eb243ce07ad4a9ed1cf2f7f9a22e951181bdb6e0716361b8eaeb173acfd5b6'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, label):
    if not bool(condition):
        raise ValueError(label)


def load_candidate(manifest_path):
    """Return (numpy arrays, receipt), preserving original source-only labels.

    No actor/optimizer is loaded and no files are written. The returned receipt
    keeps field-supported reconstruction distinct from directly saved inputs.
    """
    manifest_path = Path(manifest_path).resolve()
    receipt = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(receipt['schema'] == 'wlr50_clean.det_front_reconstruction_candidate.v1', 'unsupported candidate schema')
    require(receipt['status'] == 'CANDIDATE_ONLY_NOT_ADMITTED_TO_LEARNING', 'candidate status changed')
    require(receipt['source_checkpoint']['sha256'] == EXPECTED_SOURCE, 'wrong source checkpoint')
    require(receipt['observation_provenance']['direct389_tensor_was_saved_by_source'] is False
            and receipt['observation_provenance']['reconstructed_from_explicit_synchronized_fields'] is True
            and receipt['observation_provenance']['missing_fields_guessed_or_filled'] is False,
            'reconstruction provenance mislabelled')
    for key in ('dataset', 'checks', 'helper', 'source_video_manifest', 'source_checkpoint'):
        binding = receipt[key]
        require(sha(binding['path']) == binding['sha256'], key + ' hash differs')
    require(receipt['learning']['AUX_updates'] == receipt['learning']['PPO_decisions_added']
            == receipt['learning']['PPO_updates_added'] == 0, 'candidate was mislabelled learning credit')
    require(receipt['checks']['numeric_atol_fixed_before_replay'] == 1e-6, 'numerical tolerance changed')
    require(all(0 <= value <= 1e-6 for value in receipt['checks']['maximum_absolute_errors'].values()),
            'source model replay outside declared tolerance')
    with np.load(receipt['dataset']['path'], allow_pickle=False) as loaded:
        arrays = {key: loaded[key].copy() for key in loaded.files}
    require({key: list(value.shape) for key, value in arrays.items()} == receipt['dataset']['shapes'], 'array shape receipt differs')
    x, raw = arrays['X389_reconstructed'], arrays['raw12_actual_deterministic']
    require(x.shape == (254, 389) and raw.shape == (254, 12), 'not reviewed P02 window')
    require(x.dtype == raw.dtype == np.float32 and all(np.isfinite(a).all() for a in arrays.values()), 'invalid tensor dtype/value')
    require(np.array_equal(x[:, :13], np.tile([0., 1.] + [0.] * 11, (254, 1))), 'non-P02 input')
    require(np.array_equal(raw, arrays['source_conditional_mean12']), 'raw labels changed from actual deterministic means')
    require(np.array_equal(x[:, 195:207], arrays['source_history_center12']), 'recorded P02 HISTORY differs')
    require(np.all(x[:, 372:389] == 0), 'unexpected capture assistance/pending state')
    require(np.array_equal(arrays['source_decision'], np.arange(3, 257))
            and np.array_equal(arrays['input_tick'], np.arange(16, 2041, 8)), 'source row identity differs')
    for name, start in (('suggested_train_indices', 0), ('suggested_validation_indices', 1), ('source_only_indices', 2)):
        require(np.array_equal(arrays[name], np.arange(start, 254, 3)), 'fixed proposed split changed')
    checks = [json.loads(line) for line in Path(receipt['checks']['path']).read_text(encoding='utf-8').splitlines()]
    require(len(checks) == 254, 'row verification is incomplete')
    for index, row in enumerate(checks):
        require(row['candidate_index'] == index and row['decision'] == int(arrays['source_decision'][index])
                and row['start_tick'] == int(arrays['input_tick'][index]), 'row check identity differs')
        require(hashlib.sha256(x[index].tobytes()).hexdigest() == row['reconstructed_float32_observation_sha256'], 'row input hash differs')
        require(row['history_and_assist_exact'] is True and row['all12_permission_no_assist_each_physics_tick'] is True,
                'row does not satisfy original action evidence')
        require(all(0 <= value <= 1e-6 for value in row['errors_max_abs'].values()), 'row replay exceeds tolerance')
    # This is an additional read receipt, not a mutation to the construction one.
    return arrays, {**receipt, 'read_validation': {'manifest_path': str(manifest_path),
        'manifest_sha256': sha(manifest_path), 'validator_path': str(Path(__file__).resolve()),
        'validator_sha256': sha(Path(__file__)), 'verified_rows': len(checks), 'fit_performed': False}}
