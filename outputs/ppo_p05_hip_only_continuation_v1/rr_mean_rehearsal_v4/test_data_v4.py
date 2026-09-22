"""Small fail-closed data tests; no actor fitting or real checkpoint writes."""
from copy import deepcopy
import numpy as np
import pytest

import data_v4 as data


def arrays():
    phase = np.array([9] * 345 + [10] + [11] * 167)
    x = np.zeros((513, 389), dtype=np.float32)
    x[np.arange(513), phase - 1] = 1
    current = x.copy(); current[:237, 17] = np.float32(.001)
    protection = np.zeros((305, 389), dtype=np.float32)
    pp = list(sorted(data.PROTECTION_PHASES)) + [2] * 296
    protection[np.arange(305), np.asarray(pp) - 1] = 1
    probe = np.zeros((2, 389), dtype=np.float32); probe[:, 0] = 1
    train, validation = data.fixed_split(phase)
    return dict(source_observations389=x, observations389_current_X17=current,
        raw12_actual=np.ones((513, 12), dtype=np.float32), source_mean12=np.zeros((513, 12), dtype=np.float32),
        source_sigma12=np.ones((513, 12), dtype=np.float32), source_logp=np.zeros(513, dtype=np.float32),
        protection_observations=protection, p01_observations=probe,
        source_indices=np.arange(747, 1260), input_ticks=np.arange(5976, 10073, 8),
        global_decisions=np.arange(204524, 205037), train_indices=train, validation_indices=validation)


def test_fixed_split_keeps_all_rows_and_P10_train_only():
    a = arrays(); data.validate_arrays(a)
    tx = a['observations389_current_X17'][a['train_indices']]
    vx = a['observations389_current_X17'][a['validation_indices']]
    assert {k: len(v) for k, v in data.phase_groups(tx).items()} == {'P09': 230, 'P10': 1, 'P11': 111}
    assert {k: len(v) for k, v in data.phase_groups(vx).items()} == {'P09': 115, 'P11': 56}
    assert sorted([*a['train_indices'], *a['validation_indices']]) == list(range(513))
    assert a['train_indices'][:4].tolist() == [0, 2, 3, 5]


@pytest.mark.parametrize('kind', ['other_column', 'P12_positive', 'P09_protection', 'mean_labels', 'split', 'nonfinite', 'wrong_window', 'wrong_X17_count'])
def test_bad_data_fails_closed(kind):
    a = arrays()
    if kind == 'other_column': a['observations389_current_X17'][0, 18] = .1
    if kind == 'P12_positive':
        for key in ('source_observations389', 'observations389_current_X17'):
            a[key][0, :13] = 0; a[key][0, 11] = 1
    if kind == 'P09_protection':
        a['protection_observations'][0, :13] = 0; a['protection_observations'][0, 8] = 1
    if kind == 'mean_labels': a['raw12_actual'][0] = a['source_mean12'][0]
    if kind == 'split': a['validation_indices'][0] = 0
    if kind == 'nonfinite': a['raw12_actual'][0, 0] = np.nan
    if kind == 'wrong_window': a['source_indices'][0] = 746
    if kind == 'wrong_X17_count': a['observations389_current_X17'][237, 17] = .1
    with pytest.raises(ValueError): data.validate_arrays(a)


def test_uniform_recent_selection_has_no_action_or_outcome_input():
    assert data.uniform_indices([10, 80]).tolist() == [10, 80]
    chosen = data.uniform_indices(np.arange(232))
    assert len(chosen) == len(set(chosen.tolist())) == 16
    assert (chosen[0], chosen[-1]) == (0, 231)


@pytest.fixture
def metadata():
    m = data.read(data.SOURCE.with_name(data.SOURCE.stem + '_manifest.json'))
    m['checkpoint_path'] = str(data.SOURCE)
    return m


@pytest.mark.parametrize('kind', ['runtime', 'two_events', 'modified_old_AUX', 'counter', 'source_sha'])
def test_actual_metadata_rejects_wrong_semantics_or_lineage(metadata, kind):
    altered = deepcopy(metadata); contract = deepcopy(metadata['runtime_contract'])
    if kind == 'runtime':
        contract['source_git_commit'] = 'future'; altered['runtime_contract'] = contract
    if kind == 'two_events':
        altered['rr_postcross_workspace_branch']['front_rehearsal_auxiliary']['events'].pop()
    if kind == 'modified_old_AUX':
        altered['task_conditioned_hip_wheel_branch']['auxiliary_mean_learning']['tampered'] = True
    if kind == 'counter': altered['global_policy_decisions'] -= 128
    if kind == 'source_sha': altered['checkpoint_sha256'] = '0' * 64
    with pytest.raises(ValueError): data.validate_current_metadata(altered, contract, metadata)


def test_sealed_actual_data_read_only(metadata):
    if not data.MANIFEST.exists(): pytest.skip('build not completed yet')
    loaded = data.load_reviewed_data(metadata, metadata['runtime_contract'])
    assert loaded['train_observations'].shape == (342, 389)
    assert loaded['validation_observations'].shape == (171, 389)
    assert loaded['protection_observations'].shape == (305, 389)
    assert 'protection_raw_targets' not in loaded
    assert not loaded['receipt']['optimization_authorized_by_loader']
    assert loaded['receipt']['source_checkpoint']['entire_existing_front_AUX_ledger']['accepted_auxiliary_updates_total'] == 96


def test_loader_rejects_bound_dataset_hash_change(metadata, monkeypatch):
    if not data.MANIFEST.exists(): pytest.skip('build not completed yet')
    original = data.sha
    monkeypatch.setattr(data, 'sha', lambda path: '0' * 64 if str(path) == str(data.DATA) else original(path))
    with pytest.raises(ValueError, match='binding changed'):
        data.load_reviewed_data(metadata, metadata['runtime_contract'])
