"""One-time parent-authorized compaction of NOT-YET-FROZEN v4 metadata.

Numeric NPZ and all historical sources remain unchanged. No model/fit/budget.
"""
import json
import torch
import data_v4 as data


def main():
    assert not torch.cuda.is_available()
    manifest = data.read(data.MANIFEST)
    assert manifest['status'] == 'FIXED_OFFPOLICY_DATA_NOT_OPTIMIZATION_AUTHORIZATION'
    assert not manifest['fit_executed'] and not manifest['budget_selected']
    assert manifest['dataset']['sha256'] == data.sha(data.DATA) == '607c057d209aeb0278115632507f67042f10a59bc1ee142884d10a42b0265144'
    reference = data.read(manifest['source_manifest']['path'])
    reference['checkpoint_path'] = str(data.SOURCE)
    manifest['source_checkpoint'] = data.validate_current_metadata(reference, reference['runtime_contract'], reference)
    manifest['loader'] = data.binding(data.__file__)
    data.MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    loaded = data.load_reviewed_data(reference, reference['runtime_contract'])
    result = {'schema': data.SCHEMA + '.readonly_test', 'result': 'PASS',
        'manifest': data.binding(data.MANIFEST), 'dataset': data.binding(data.DATA),
        'data_receipt_sha256': data.digest(loaded['receipt']), 'source_checkpoint_sha256': data.SOURCE_SHA,
        'numeric_data_unchanged': True,
        'shapes': {k: list(v.shape) for k, v in loaded.items() if torch.is_tensor(v)},
        'fit_executed': False, 'checkpoint_written': False}
    (data.HERE / 'data_read_test.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
