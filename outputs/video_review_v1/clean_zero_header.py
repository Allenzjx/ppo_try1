"""Data-only static title PNG avoids intermittent FFmpeg drawtext glyph loss."""
import argparse
import importlib.util
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw, ImageFont

spec=importlib.util.spec_from_file_location('clean_header_base',Path(__file__).with_name('sealed_zero_review.py'))
review=importlib.util.module_from_spec(spec); spec.loader.exec_module(review)
base=review.base


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--folder',type=Path,required=True)
    args=parser.parse_args(); folder=args.folder.resolve(strict=True)
    base.require(folder.is_relative_to(review.OUT),'Output-only review folder required')
    native=folder/'zero_Nplus0_full_native.media.json'; original=base.read_json(native)
    old=base.read_json(folder/'zero_Nplus0_full_review.media.json')
    base.require(review.sha(original['output'])==original['output_validation']['sha256'],'Native input changed')
    count=original['frame_count']; header=folder/'zero_clean_header.png'; output=folder/'zero_Nplus0_full_review_clean.mp4'
    base.require(not header.exists() and not output.exists(),'Never overwrite')
    title=f"N + 0 | NON-RESIDUAL | recorded {original['physical_result']} | normal speed - 15 fps"
    panel=Image.new('RGB',(1280,60),'black'); draw=ImageDraw.Draw(panel)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',23)
    base.require(draw.textlength(title,font=font)<=1248,'Title exceeds original margin')
    draw.text((16,16),title,font=font,fill='white'); panel.save(header)
    ffmpeg=base.find_ffmpeg()
    command=[str(ffmpeg),'-hide_banner','-nostdin','-v','error','-n','-loop','1','-framerate','15','-i',str(header),
        '-i',original['output'],'-filter_complex','[0:v]format=yuv420p,setpts=PTS-STARTPTS[h];[1:v]setpts=PTS-STARTPTS[v];[h][v]vstack=inputs=2:shortest=1[out]',
        '-map','[out]','-an','-frames:v',str(count),'-r','15','-fps_mode','cfr','-c:v','libx264','-crf','18',
        '-preset','veryfast','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    base.run(command)
    validation=base.validate_mp4(output,expected_frame_count=count,expected_fps=15,expected_width=1280,expected_height=780,
        maximum_duration_s=200,require_sane_container_duration=True)
    base.require(validation['valid'],'Clean-title video failed full decode')
    frames=base.decode_frame_timeline(output)
    base.require(len(frames)==count and all(abs(f.pts_s-i/15)<1e-5 for i,f in enumerate(frames)),'Clean title changed PTS')
    selected=old['camera_keyframes']; indices=sorted({r['source_frame'] for r in selected})
    expression='+'.join(f'eq(n\\,{i})' for i in indices)
    base.run([str(ffmpeg),'-hide_banner','-nostdin','-v','error','-n','-i',str(output),'-vf','select='+expression,
        '-fps_mode','passthrough','-frames:v',str(len(indices)),'-threads','1',str(folder/'clean_camera_key_%02d.png')])
    paths={frame:folder/f'clean_camera_key_{i+1:02d}.png' for i,frame in enumerate(indices)}
    for row in selected:
        row['path']=str(paths[row['source_frame']]); base.require(paths[row['source_frame']].is_file(),'Missing clean keyframe')
    result={'schema':'wlr50_clean.static_title_zero_review.v1','output':str(output),'native_receipt':str(native),
        'native_receipt_sha256':review.sha(native),'source_manifest':original['source_manifest'],
        'source_manifest_sha256':original['source_manifest_sha256'],'physical_result':original['physical_result'],
        'caption':title,'static_title_png':str(header),'static_title_png_sha256':review.sha(header),
        'reason':'Manual QA observed intermittent missing glyphs in drawtext frame2; clean static-PNG header replaces only the derivative title renderer.',
        'old_derivative_preserved':str(folder/'zero_Nplus0_full_review.mp4'),'original_native_preserved':True,
        'original_view_uncropped':True,'frame_count':count,'fps':15,'normal_speed':True,'new_physics_ticks':0,
        'physical_duration_s':original['physical_duration_s'],'media_duration_s':count/15,'stitched':False,'interpolated':False,
        'validation':base.compact_validation(validation),'camera_keyframes':selected,'manual_camera_QA_pending':True,'command':command}
    base.write_new_json(output.with_suffix('.media.json'),result)
    print(str(output))


if __name__=='__main__':
    original_run=subprocess.run
    def bounded(command,*args,**kw):
        if isinstance(command,(list,tuple)) and 'ffmpeg' in Path(command[0]).name.lower():
            command=[command[0],'-threads','2','-filter_threads','1','-filter_complex_threads','1',*command[1:]]
        return original_run(command,*args,**kw)
    subprocess.run=bounded
    try:main()
    finally:subprocess.run=original_run
