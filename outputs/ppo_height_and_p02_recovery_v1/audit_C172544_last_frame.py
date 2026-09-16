"""Independent decode and data-panel pixel check; output-only, no simulation."""
import json
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
STEM = HERE / 'videos/PPO_checkpoint172544_full12_wheel_qd'
FFMPEG = Path('C:/Users/kskzz/miniconda3/envs/env_isaaclab/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe')
full = STEM.with_name(STEM.name + '_independent_last.png')
crop = STEM.with_name(STEM.name + '_independent_last_panel.png')
receipt = HERE / 'C172544_overlay_independent_pixel_audit.json'
for output in (full, crop, receipt):
    if output.exists():
        raise FileExistsError(output)
subprocess.run([str(FFMPEG), '-hide_banner', '-loglevel', 'error', '-threads', '2',
                '-i', str(STEM.with_suffix('.mp4')), '-vf', 'select=eq(n\\,90)',
                '-frames:v', '1', '-threads', '2', str(full)], check=True)
decoded = Image.open(full).convert('RGB')
assert decoded.size == (1280, 960)
decoded_panel = decoded.crop((0, 720, 1280, 960))
decoded_panel.save(crop)
original = STEM.with_name(STEM.name + '.panels') / 'panel_000090.png'
expected = np.asarray(Image.open(original).convert('RGB'), dtype=np.int16)
actual = np.asarray(decoded_panel, dtype=np.int16)
assert expected.shape == actual.shape == (240, 1280, 3)
bright = expected.min(axis=2) >= 180
lost = bright & (actual.max(axis=2) < 80)
columns = []
for i, leg in enumerate(('FL', 'FR', 'RL', 'RR')):
    sl = slice(i * 320, (i + 1) * 320)
    columns.append({'leg': leg, 'mean_absolute_pixel_error_0_255': float(np.abs(expected[:, sl] - actual[:, sl]).mean()),
                    'original_bright_pixel_count': int(bright[:, sl].sum()),
                    'original_bright_to_decoded_dark_pixel_count': int(lost[:, sl].sum())})
result = {'schema': 'wlr50_clean.overlay_independent_decode_pixel_audit.v1', 'frame_index': 90,
          'actual_physics_tick': 723, 'full_decoded_frame': str(full), 'decoded_panel': str(crop),
          'original_data_panel': str(original), 'pixel_count': int(bright.size),
          'mean_absolute_pixel_error_0_255': float(np.abs(expected - actual).mean()),
          'original_bright_pixel_count': int(bright.sum()), 'original_bright_to_decoded_dark_pixel_count': int(lost.sum()),
          'bright_threshold_all_RGB': 180, 'dark_threshold_max_RGB': 80, 'columns': columns,
          'conclusion': 'Independent MP4 decode preserves all originally bright data glyph pixels; no removed-letter evidence in the encoded video. Full-frame preview appearance alone is not pixel evidence.'}
assert lost.sum() == 0
receipt.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
