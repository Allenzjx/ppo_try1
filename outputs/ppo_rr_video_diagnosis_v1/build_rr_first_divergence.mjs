// One flat scientific CSV. Frozen actual records; no simulator or policy calls.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import assert from 'node:assert/strict';
import {Workbook} from '@oai/artifact-tool';
const root=path.dirname(fileURLToPath(import.meta.url));
const records=JSON.parse(await fs.readFile(path.join(root,'rr_bounded_before.records.json'),'utf8'));
const summary=JSON.parse(await fs.readFile(path.join(root,'rr_bounded_before.summary.json'),'utf8'));
const output=path.join(root,'rr_first_divergence.csv');
try {await fs.access(output); throw new Error('Preserve existing CSV; do not overwrite');}
catch(e){if(e.code!=='ENOENT')throw e;}
const channels=['FL_hip','FL_knee','FR_hip','FR_knee','RL_hip','RL_knee','RR_hip','RR_knee','FL_wheel','FR_wheel','RL_wheel','RR_wheel'];
const vectorSuffix={
 raw_residual_full12:channels.map(x=>x+'_latent'),
 projected_residual_full12:channels.map((x,i)=>x+(i<8?'_deg':'_rad_s')),
 phase_mask_full12:channels,
 base_chassis_rpy_derived_rad:['roll','pitch','yaw'],
 calibrated_body_rpy_rad:['roll','pitch','yaw'],
 body_angular_velocity_world_rad_s:['x','y','z'],
 imu_angular_velocity_base_frame_rad_s:['x','y','z'],
 body_linear_velocity_world_m_s:['x','y','z'],
 com_position_world_m:['x','y','z'],com_velocity_world_m_s:['x','y','z'],
};
for(const leg of ['FL','FR','RL','RR'])for(const surface of ['ground','obstacle'])vectorSuffix[leg+'_'+surface+'_force_w_n']=['x','y','z'];
const scopes=t=>t<=120?'initial_and_first_tracking_0_to_1s':t>=166&&t<=189?'five_degree_tracking_onset_window':t>=976&&t<=983?'C0_terminal_same_elapsed_window':null;
const selected=records.filter(r=>scopes(r.episode_physics_tick)!==null);
assert.equal(selected.length,306);
const flat=selected.map(r=>{
 const out={run_label:r.run_label,record_scope:scopes(r.episode_physics_tick),episode_physics_tick:r.episode_physics_tick,simulation_time_s:r.simulation_time_s,source_phase_id:r.source_phase_id};
 out.source_run=summary[r.run_label==='B0_zero'?'zero':'ppo'].source;
 for(const [k,v] of Object.entries(r)){
   if(k in vectorSuffix){for(let i=0;i<vectorSuffix[k].length;i++)out[k+'_'+vectorSuffix[k][i]]=Array.isArray(v)?v[i]??null:null;}
   else if(k==='headroom_clipped_servo_indices'){for(let i=0;i<8;i++)out['headroom_clip_'+channels[i]]=Array.isArray(v)?v.includes(i):null;}
   else if(k==='body_collision'){out.body_collision_detected=v?.detected??null;out.body_collision_real_pair_active=v?.real_pair_active??null;}
   else if(k==='data_quality'){out.data_quality_flags=Array.isArray(v)?JSON.stringify(v):null;}
   else if(k==='com_included_bodies'){out.com_included_body_count=Array.isArray(v)?v.length:typeof v==='number'?v:null;}
   else if(k==='task_transition_at_exact_tick'){out.exact_tick_transition_record_present=Array.isArray(v)?v.length>0:null;}
   else if(v===null||['string','number','boolean'].includes(typeof v))out[k]=v;
   else throw new Error('Unmapped non-scalar: '+k);
 }
 return out;
});
const first=['run_label','record_scope','episode_physics_tick','simulation_time_s','source_phase_id','RR_knee_nominal_request_deg','RR_knee_mapped_N_deg','RR_knee_final_target_deg','RR_knee_actual_deg','RR_knee_e_tracking_actual_minus_final_deg','RR_knee_e_residual_final_minus_mapped_N_deg','RR_knee_direct_same_tick_policy_delta_deg'];
const keys=[...new Set(flat.flatMap(Object.keys))];
const header=[...first,...keys.filter(k=>!first.includes(k)).sort()];
const rows=flat.map(r=>header.map(k=>r[k]??null));
assert(rows.every(r=>r.length===header.length&&r.every(v=>v===null||typeof v!=='object')));
const wb=Workbook.create(), sheet=wb.worksheets.add('RR first divergence');
sheet.getRangeByIndexes(0,0,1,header.length).values=[header];
for(let i=0;i<rows.length;i+=128){const block=rows.slice(i,i+128);sheet.getRangeByIndexes(i+1,0,block.length,header.length).values=block;}
sheet.showGridLines=false;sheet.freezePanes.freezeRows(1);
sheet.getRangeByIndexes(0,0,rows.length+1,header.length).format={font:{name:'Arial',size:10},columnWidth:24,rowHeight:23,verticalAlignment:'center'};
sheet.getRangeByIndexes(0,0,1,header.length).format={font:{name:'Arial',size:10,bold:true},fill:'#E8EEF4',wrapText:true,rowHeight:90,horizontalAlignment:'center'};
sheet.getRangeByIndexes(1,3,rows.length,1).setNumberFormat('0.000000');
sheet.getRangeByIndexes(1,5,rows.length,7).setNumberFormat('0.000000');
wb.recalculate();
const inspection=await wb.inspect({kind:'region',sheetId:sheet.name,range:'F35:L39',maxChars:3000,tableMaxRows:5,tableMaxCols:7});
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:10},maxChars:1500});
const png=await wb.render({sheetName:sheet.name,range:'F1:L8',scale:1,format:'png'});
await fs.writeFile(path.join(root,'rr_first_divergence_preview.png'),new Uint8Array(await png.arrayBuffer()));
const quote=v=>v===null||v===undefined?'':typeof v==='number'||typeof v==='boolean'?String(v):'"'+String(v).replaceAll('"','""')+'"';
const csv=[header.map(quote).join(',')];
for(let i=0;i<rows.length;i+=128){const size=Math.min(128,rows.length-i);const authored=sheet.getRangeByIndexes(i+1,0,size,header.length).values;
 for(let j=0;j<size;j++){assert.deepEqual(authored[j].map(v=>v??null),rows[i+j]);csv.push(authored[j].map(quote).join(','));}}
await fs.writeFile(output,csv.join('\r\n')+'\r\n',{flag:'wx',encoding:'utf8'});
await fs.writeFile(path.join(root,'rr_first_divergence_authoring_receipt.json'),JSON.stringify({output,rows:rows.length,columns:header.length,scope_ticks:[[0,120],[166,189],[976,983]],per_run_rows:153,missing_values:'blank_unknown_not_zero',units:'canonical servo deg, wheel rad/s, world CoM m and m/s; quaternion-calibrated body Euler rad',source_records:summary.record_file,scalar_grid_verified:true,recalculated:true,inspection,errors,not_on_policy:true,optimizer_updates:0,preview:'rr_first_divergence_preview.png'},null,2),{flag:'wx'});
console.log(JSON.stringify({output,rows:rows.length,columns:header.length,preview:path.join(root,'rr_first_divergence_preview.png')}));
