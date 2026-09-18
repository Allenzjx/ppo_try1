"""Pixel QA of real decoded captions, not a re-render or OCR inference."""
from pathlib import Path
import json
import hashlib
import numpy as np
from PIL import Image

FOLDER=Path(__file__).resolve().parent/'cp178432_run051034'


def compare(decoded,panel,box):
    with Image.open(decoded) as image:a=np.asarray(image.convert('RGB').crop(box),dtype=np.int16)
    with Image.open(panel) as image:b=np.asarray(image.convert('RGB'),dtype=np.int16)
    assert a.shape==b.shape
    bright=b.max(axis=2)>180
    lost=bright & (a.max(axis=2)<70)
    return {'decoded_image':str(decoded),'expected_Pillow_panel':str(panel),'decoded_crop_box':box,
        'decoded_image_sha256':hashlib.sha256(decoded.read_bytes()).hexdigest(),
        'expected_panel_sha256':hashlib.sha256(panel.read_bytes()).hexdigest(),
        'source_bright_text_pixel_count':int(bright.sum()),'bright_to_dark_pixel_count':int(lost.sum()),
        'bright_to_dark_fraction':float(lost.sum()/bright.sum()),
        'mean_abs_rgb_difference':float(np.abs(a-b).mean()),
        'difference_scope':'Lossy YUV420/H264 colors/edges need not be byte-identical; this detects missing bright glyph regions.'}


def main():
    media=json.loads((FOLDER/'PPO_full_actual_wheels_RR.media.json').read_text())
    rows=[]; seen=set()
    for key in media['camera_keyframes']:
        index=key['frame_index']
        if index in seen:continue
        seen.add(index)
        rows.append(compare(Path(key['path']),FOLDER/'PPO_full_actual_wheels_RR.panels'/f'panel_{index:06d}.png',(0,720,1280,880)))
    rows.append(compare(FOLDER/'caption_QA_terminal_crop.png',FOLDER/'PPO_full_actual_wheels_RR.panels/panel_000738.png',(0,0,1280,160)))
    for index,suffix in [(0,'01'),(449,'02')]:
        rows.append(compare(FOLDER/f'PPO_P05_FL_capture_failure_actual_30s_decoded_{suffix}.png',
            FOLDER/'PPO_P05_FL_capture_failure_actual_30s.panels'/f'panel_{index:06d}.png',(0,720,1280,880)))
    result={'schema':'wlr50_clean.decoded_caption_pixel_QA.v1','rows':rows,
        'all_checked_original_bright_glyph_pixels_remain_nonblack':all(r['bright_to_dark_pixel_count']==0 for r in rows),
        'independent_single_frame_crop':str(FOLDER/'caption_QA_terminal_crop.png'),
        'initial_visual_preview_concern':'Full-frame tool preview appeared to omit characters, while actual PNG/video panel pixels and independently decoded crop preserve them.',
        'encoding_damage_established':False,'existing_full_and_comparison_reencoded':False,
        'scope':'Listed real decoded caption frames only. Not an all-frame OCR/visual guarantee; full video decode/PTS checks are separate.'}
    with (FOLDER/'caption_pixel_QA.json').open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({'checked_frames':len(rows),'all_bright_glyphs_nonblack':result['all_checked_original_bright_glyph_pixels_remain_nonblack'],
        'mae_range':[min(r['mean_abs_rgb_difference'] for r in rows),max(r['mean_abs_rgb_difference'] for r in rows)]}))


if __name__=='__main__':main()
