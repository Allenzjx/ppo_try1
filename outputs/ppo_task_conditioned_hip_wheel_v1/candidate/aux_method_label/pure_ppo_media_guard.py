"""Candidate-only pure-PPO media gate; auxiliary provenance requires its own adapter.

No auxiliary ledger schema or learning claim is inferred from the key's presence.
Old review/quantity pair modules remain unchanged. No encoding occurs on rejection.
"""
from __future__ import annotations
import argparse
from collections.abc import Mapping
import importlib.util
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


r = module('pure_guard_existing_review', OUT/'review_video.py')


def auxiliary_paths(value, path='$'):
    """Presence is fail-closed, including empty/unknown nested or inherited ledgers."""
    found = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f'{path}.{key}'
            name = str(key).lower()
            if name == 'aux' or name.startswith('aux_') or 'auxiliary' in name:
                found.append(child)
            found.extend(auxiliary_paths(item, child))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            found.extend(auxiliary_paths(item, f'{path}[{i}]'))
    return found


def require_pure_infos(infos):
    found = auxiliary_paths(infos)
    r.require(not found,
        'Pure PPO media refused: nested/unknown auxiliary provenance at '+', '.join(found)+
        '. A dedicated validated PPO + LIMITED AUX adapter is required; presence alone is not a valid aux ledger.')
    return {'method':'pure_PPO', 'auxiliary_learning_claimed':False,
            'nested_auxiliary_provenance_checked':True}


def checkpoint_method(context):
    if context['role'] == 'B':
        return {'method':'N_plus_zero', 'auxiliary_learning_claimed':False}
    r.require(context['role'] == 'C', 'Only original B or officially loaded C is eligible')
    identity = context['checkpoint']
    metadata = r.base.read_json(identity['manifest'])
    # The existing sealed/checked review has already verified the real CP hash.
    import torch
    infos = torch.load(identity['checkpoint'], map_location='cpu', weights_only=False)['infos']
    r.require(infos == {k:v for k,v in metadata.items() if k not in
        ('checkpoint_path','checkpoint_sha256','save_load_round_trip')},
        'Actual checkpoint infos and verified sidecar disagree on method provenance')
    return require_pure_infos(infos)


def export_pure(source, destination):
    context = r.sealed_source(source)
    checkpoint_method(context)  # Before directory creation, native export or encoding.
    return r.export(source, destination)


def quantity_pair_pure(b_receipt, c_receipt, destination):
    b, c = r.checked_review(b_receipt), r.checked_review(c_receipt)
    checkpoint_method(b['context']); checkpoint_method(c['context'])
    pair = module('pure_guard_existing_quantity_pair', OUT/'pair_quantity_only.py')
    return pair.build_pair(b_receipt, c_receipt, destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export-pure')
    export.add_argument('--source', type=Path, required=True)
    export.add_argument('--destination', type=Path, required=True)
    pair = sub.add_parser('quantity-pair-pure')
    pair.add_argument('--b-receipt', type=Path, required=True)
    pair.add_argument('--deterministic-receipt', type=Path, required=True)
    pair.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    result = (export_pure(args.source, args.destination) if args.command == 'export-pure' else
        quantity_pair_pure(args.b_receipt, args.deterministic_receipt, args.destination))
    print(json.dumps(result, indent=2))
