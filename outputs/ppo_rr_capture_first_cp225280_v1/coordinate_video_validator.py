"""Optional output-side coordinate binding, stdlib only; not installed.

Existing source/checkpoint/decision SHA and schema checks remain mandatory.
This is a supplementary unit-consistency check, not proof of physical success.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re

GAIN_KEY = 'local_mean_coordinate_gain_full12'
LEDGER_KEY = 'local_mean_coordinate_migrations'
IDENTITY = (1.,) * 12
RR10 = (1.,) * 6 + (10., 10.) + (1.,) * 4
SOURCE_HEAD = '0ff03eafeeb75ba8505a99478cea2a6243cf93f5'
# Exact existing local-task source families only. Never infer identity merely
# because an unknown future checkpoint omitted its gain field.
LEGACY_IDENTITY_HEADS = {
    'ecf205e80094693938057589fc91f7f31082ef89': 'v1',
    '5f8487b76f27eb165a329a0b6f3096f54ebb4d48': 'v2',
    '1e10d39c9a80c017cb7e4a0035d00c40a5b4dd81': 'v2',
    SOURCE_HEAD: 'v2',
}
ZERO_CREDIT = ('new_policy_decisions', 'new_PPO_updates', 'new_PPO_Adam_steps', 'new_AUX_steps')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def checked_gain(value):
    require(isinstance(value, (list, tuple)) and len(value) == 12 and
            all(type(x) in (int, float) and math.isfinite(x) for x in value) and
            tuple(value) in (IDENTITY, RR10), 'invalid mean coordinate gain')
    return tuple(float(x) for x in value)


def checked_hash(value, length=64):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{'+str(length)+'}', value),
            'invalid coordinate provenance hash')
    return value


def _gain(mapping, *, legacy, label):
    if GAIN_KEY not in mapping:
        require(legacy, label+' omitted new coordinate gain')
        return IDENTITY
    # Explicit null never means missing/identity.
    return checked_gain(mapping[GAIN_KEY])


def _receipt(event):
    binding = event.get('receipt') or {}
    require(isinstance(binding.get('path'), str) and Path(binding['path']).is_absolute(),
            'coordinate receipt requires its explicit absolute path')
    expected = checked_hash(binding.get('sha256'))
    path = Path(binding['path']).resolve(strict=True)
    with path.open('rb') as stream:
        require(hashlib.file_digest(stream, 'sha256').hexdigest() == expected,
                'coordinate receipt bytes changed')
    document = json.loads(path.read_text(encoding='utf-8-sig'))
    require(isinstance(document, dict) and
            document.get('schema') == 'candidate.rr_mean_coordinate_migration.v1' and
            document.get('status') == 'PASS' and
            isinstance(document.get('flags'), dict) and document['flags'] and
            all(value is True for value in document['flags'].values()),
            'coordinate receipt lacks passed actual migration checks')
    require(document.get('source_gain') == list(IDENTITY) and
            document.get('target_gain') == list(RR10) and
            all(type(document.get(k)) is int and document[k] == 0 for k in ZERO_CREDIT),
            'coordinate receipt gain or zero-learning credit differs')
    return expected


def validate_coordinate_video_binding(source, checkpoint):
    """Call after existing schema/hash validation; return per-row expected units."""
    runtime = checkpoint.get('runtime_contract') or {}
    require(source.get('runtime_contract') == runtime,
            'coordinate source/checkpoint runtime differs')
    head = checked_hash(runtime.get('source_git_commit'), 40)
    version = LEGACY_IDENTITY_HEADS.get(head)
    legacy = version is not None
    profile = runtime.get('local_contract') or {}
    actor = (checkpoint.get('runner_config') or {}).get('actor') or {}
    control = source.get('control_contributions') or {}
    gain = _gain(profile, legacy=legacy, label='runtime')
    require(_gain(actor, legacy=legacy, label='saved actor') == gain and
            _gain(control, legacy=legacy, label='video control') == gain,
            'runtime/actor/control coordinate gain mismatch')
    if legacy:
        require(gain == IDENTITY and
                source.get('schema') == 'wlr50_clean.frozen_prior_rr_capture_video.'+version and
                checkpoint.get('schema') == 'wlr50_clean.frozen_prior_rr_capture_checkpoint.'+version,
                'old immutable source cannot advertise changed coordinates/schema')
    else:
        require(gain == RR10 and profile.get('observation_dimension') == 448 and
                source.get('schema') == 'wlr50_clean.frozen_prior_rr_capture_video.v2' and
                checkpoint.get('schema') == 'wlr50_clean.frozen_prior_rr_capture_checkpoint.v2' and
                actor.get('observation_layout') == 'role439_rr_capture_local_v2' and
                actor.get('legacy447_migration_only', False) is False,
                'new coordinate video requires explicit gain10 formal448 actor')
    events = checkpoint.get(LEDGER_KEY, [])
    source_events = control.get(LEDGER_KEY, [])
    require(isinstance(events, list) and isinstance(source_events, list) and
            json.dumps(events, sort_keys=True, allow_nan=False) ==
            json.dumps(source_events, sort_keys=True, allow_nan=False),
            'source/checkpoint coordinate migration receipts differ')
    receipt_sha = None
    if legacy:
        require(not events, 'old identity source cannot contain gain10 migration')
    else:
        require(len(events) == 1 and isinstance(events[0], dict),
                'gain10 requires exactly one coordinate migration')
        event = events[0]
        require(event.get('schema') == 'wlr50_clean.rr_mean_coordinate_publication.v1' and
                event.get('source_head') == SOURCE_HEAD and event.get('destination_head') == head and
                event.get('source_gain') == list(IDENTITY) and event.get('target_gain') == list(RR10) and
                event.get('destination_runtime_sha256') == runtime.get('runtime_content_sha256') and
                all(type(event.get(k)) is int and event[k] == 0 for k in ZERO_CREDIT),
                'coordinate migration does not bind this runtime/gain or has learning credit')
        for key in ('source_checkpoint_sha256', 'source_manifest_sha256',
                    'source_runtime_sha256', 'destination_runtime_sha256'):
            checked_hash(event.get(key))
        require(isinstance(event.get('source_counts'), dict) and event['source_counts'] and
                event['source_counts'] == event.get('destination_counts'),
                'coordinate migration changed counters')
        counts = checkpoint.get('counts') or {}
        require(all(type(value) is int and value >= 0 and type(counts.get(key)) is int and
                    counts[key] >= value for key, value in event['destination_counts'].items()),
                'checkpoint counters precede coordinate migration')
        receipt_sha = _receipt(event)
    return dict(gain_full12=list(gain), legacy_identity=legacy, source_head=head,
                coordinate_receipt_sha256=receipt_sha, migration_count=len(events))


def validate_coordinate_video_request(request, binding):
    """Every sealed decision, including inactive policy rows, uses these units."""
    require(isinstance(binding, dict) and type(binding.get('legacy_identity')) is bool,
            'request needs validated checkpoint coordinate binding')
    expected = checked_gain(binding.get('gain_full12'))
    actual = _gain(request, legacy=binding['legacy_identity'], label='policy request')
    require(actual == expected, 'policy request gain differs from runtime/actor/control')
    return list(actual)
