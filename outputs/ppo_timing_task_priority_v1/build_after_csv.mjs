// Scientific flat CSVs authored in typed artifact-tool workbooks.
// No simulator/model calls. Inputs must already be completed, source-bound extracts.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import assert from 'node:assert/strict';
import {Workbook} from '@oai/artifact-tool';
const root=path.dirname(fileURLToPath(import.meta.url));
const [sequenceInput,...prefixes]=process.argv.slice(2);
assert(sequenceInput&&prefixes.length>=2,'sequence JSON and at least zero/PPO completed extract prefixes required');
const read=async p=>JSON.parse(await fs.readFile(path.resolve(root,p),'utf8'));
const scalar=v=>v==null?null:typeof v==='object'?JSON.stringify(v):v;
const sequence=await read(sequenceInput);
assert(Array.isArray(sequence.rows)&&sequence.rows.length>0);
assert(typeof sequence.command_time_definition==='string'&&sequence.command_time_definition.length>0,'Sequence clock definition missing');
const sequenceRows=sequence.rows.map(r=>{
 const {command_time_s,...rest}=r;
 const preTime=r.command_pre_step_time_s??command_time_s??null;
 if(r.command_pre_step_time_s!=null&&command_time_s!=null)assert.equal(r.command_pre_step_time_s,command_time_s,'Conflicting pre-step time aliases');
 if(r.command_pre_step_tick==null)assert(preTime===null&&r.command_post_step_tick==null,'Unknown dispatch cannot acquire a time');
 else{
  assert(Number.isInteger(r.command_pre_step_tick)&&r.command_pre_step_tick>=0&&r.command_post_step_tick===r.command_pre_step_tick+1);
  assert(Number.isFinite(preTime)&&Math.abs(preTime-r.command_pre_step_tick/120)<1e-10,'Command time must be pre-step');
 }
 return {...Object.fromEntries(Object.entries(rest).map(([k,v])=>[k,scalar(v)])),command_pre_step_time_s:preTime,
  selected_joint_array_order:JSON.stringify(sequence.selected_joint_order),leg_and_wheel_array_order:JSON.stringify(sequence.leg_and_wheel_order)};
});
const channels=['FL_hip','FL_knee','FR_hip','FR_knee','RL_hip','RL_knee','RR_hip','RR_knee','FL_wheel','FR_wheel','RL_wheel','RR_wheel'];
const legs=['FL','FR','RL','RR'];
const suffix={raw_residual_full12:channels.map(k=>k+'_latent'),
 projected_residual_full12:channels.map((k,i)=>k+(i<8?'_deg':'_rad_s')),phase_mask_full12:channels,
 calibrated_body_rpy_rad:['roll','pitch','yaw'],body_angular_velocity_world_rad_s:['x','y','z'],
 com_position_world_m:['x','y','z'],com_velocity_world_m_s:['x','y','z'],
 source_lines:['physical','native','decision']};
for(const k of ['contact_surface','ground_contact','top_contact','support','bearing_force_n','bearing_verified','load_fraction','load_fraction_valid'])suffix['legs_'+k]=legs;
for(const k of ['wheels_nominal_canonical_rad_s','wheels_mapped_N_canonical_rad_s','wheels_final_target_canonical_rad_s',
 'wheels_actual_canonical_rad_s','wheels_actual_native_target_rad_s','wheels_zero_current_policy_native_target_rad_s',
 'wheels_direct_current_policy_delta_native_rad_s'])suffix[k]=legs;
const servoFields=['nominal_request_deg','mapped_N_deg','final_target_deg','actual_deg','velocity_deg_s',
 'e_tracking_actual_minus_final_deg','e_residual_final_minus_mapped_N_deg'];
const rrFields=['mapper_N_before_geometry_deg','nominal_geometry_adjustment_deg','geometry_corrected_N_deg',
 'controller_feedback_plus_normal_bias_deg','normal_bias_separately_recorded_deg','bounded_controller_bias_deg',
 'effective_current_policy_residual_deg','candidate_before_final_slew_deg','previous_final_servo_deg',
 'direct_current_policy_delta_deg','geometry_only_native_delta_rad'];
const requiredFields=[...Object.keys(suffix),...channels.slice(0,8).flatMap(j=>servoFields.map(f=>j+'_'+f)),
 ...['RR_hip','RR_knee'].flatMap(j=>rrFields.map(f=>j+'_'+f)),
 'run_label','episode_physics_tick','simulation_time_s','source_phase_id','native_internal_tick','native_verified',
 'semantic_endpoint_observation_tick','RR_current_lift_valid','RR_C_front_edge_crossed','RR_P_placed'];
function validateTraceContract(summary,records){
 assert(summary.record_schema==='wlr50_clean.all_servo_wheel_decomposition_records.v2'&&summary.record_fields===128,
  'Complete v2 all-servo/wheel records required; old sparse-field extracts cannot be silently accepted');
 assert(Number.isInteger(summary.end_tick)&&summary.end_tick>0&&summary.end_tick<=24000&&summary.native_ticks===summary.end_tick);
 assert(summary.fixture_old_not_after===false&&summary.completed_at_utc&&summary.native_ticks===summary.native_verified_ticks);
 assert(Array.isArray(records)&&records.length>0&&summary.exact_full_stream_join_complete===true&&records.length===summary.sparse_record_count);
 assert.equal(records.length,Math.floor(summary.end_tick/8)+1+(summary.end_tick%8===0?0:1),'Sparse grid count differs');
 assert.deepEqual(summary.array_leg_order,legs);
 assert.deepEqual(summary.source_line_order,['physical','native','semantic_decision']);
 const wheel=summary.wheel_array_contract;
 assert.deepEqual(wheel.order,legs);assert.deepEqual(wheel.native_equals_canonical_times_sign,[-1,1,-1,1]);
 assert.deepEqual(wheel.physical_joint_order,['front_left_ankle','front_right_ankle','rear_left_ankle','rear_right_ankle']);
 assert.deepEqual([...wheel.canonical_keys,...wheel.native_keys].sort(),Object.keys(suffix).filter(k=>k.startsWith('wheels_')).sort());
 const keys=Object.keys(records[0]).sort();assert.equal(keys.length,128);
 for(let index=0;index<records.length;index++){
  const r=records[index],tick=index===records.length-1?summary.end_tick:index*8;
  assert.deepEqual(Object.keys(r).sort(),keys,'Record keys differ within the run');
  assert(requiredFields.every(k=>Object.hasOwn(r,k)),'Missing required servo/wheel/decomposition field');
  assert.equal(r.run_label,summary.run_label);assert.equal(r.episode_physics_tick,tick,'Sparse tick missing, repeated or reordered');
  assert(Number.isFinite(r.simulation_time_s)&&Math.abs(r.simulation_time_s-tick/120)<1e-10,'Trace time differs from physical tick');
  if(tick>0)assert.equal(r.native_verified,true);
  for(const [k,names] of Object.entries(suffix)){
   const value=r[k];assert(value===null||Array.isArray(value)&&value.length===names.length,'Wrong array shape: '+k);
   if(Array.isArray(value))assert(value.every(v=>v===null||typeof v==='string'||typeof v==='boolean'||typeof v==='number'&&Number.isFinite(v)),'Invalid array value: '+k);
  }
  assert(r.semantic_endpoint_observation_tick===null||r.semantic_endpoint_observation_tick===tick,'Semantic evidence clock differs');
  if(r.semantic_endpoint_observation_tick===null)for(const k of Object.keys(suffix).filter(k=>k.startsWith('legs_')))assert.equal(r[k],null,'No forward-filled semantic load/contact');
 }
}
const trace=[],bindings=[];
for(const prefix of prefixes){
 const summary=await read(prefix+'.summary.json'),records=await read(prefix+'.records.json');
 validateTraceContract(summary,records);
 bindings.push({prefix,source:summary.source,head:summary.source_git_commit,runtime_sha:summary.runtime_content_sha256,
   checkpoint_decisions:summary.checkpoint_decisions,rows:records.length,end_tick:summary.end_tick});
 for(const r of records){
  const out={run_label:r.run_label,episode_physics_tick:r.episode_physics_tick,simulation_time_s:r.simulation_time_s,
   source_phase_id:r.source_phase_id,source_run:summary.source,checkpoint_decisions:summary.checkpoint_decisions};
  for(const [k,v] of Object.entries(r)){
   if(k in suffix){for(let i=0;i<suffix[k].length;i++)out[k+'_'+suffix[k][i]]=Array.isArray(v)?v[i]??null:null;}
   else if(k.endsWith('_e_residual_final_minus_mapped_N_deg'))
    out[k.replace('_e_residual_final_minus_mapped_N_deg','_final_minus_mapper_N_before_geometry_total_deg')]=v;
   else out[k]=scalar(v);
  }
  trace.push(out);
 }
}
const quote=v=>v==null?'':typeof v==='number'||typeof v==='boolean'?String(v):'"'+String(v).replaceAll('"','""')+'"';
function parseCSV(text){
 const rows=[];let row=[],value='',quoted=false;
 for(let i=0;i<text.length;i++){const c=text[i];if(c==='"'){if(quoted&&text[i+1]==='"'){value+='"';i++;}else quoted=!quoted;}
  else if(!quoted&&c===','){row.push(value);value='';}
  else if(!quoted&&c==='\n'){row.push(value.replace(/\r$/,''));rows.push(row);row=[];value='';}
  else value+=c;
 }assert(!quoted&&value===''&&row.length===0);return rows;
}
async function author(name,data,first,previewColumns){
 const output=path.join(root,name+'.csv');
 try{await fs.access(output);throw new Error('Preserve existing output: '+output);}catch(e){if(e.code!=='ENOENT')throw e;}
 const keys=[...new Set(data.flatMap(Object.keys))];
 const header=[...first.filter(k=>keys.includes(k)),...keys.filter(k=>!first.includes(k))];
 const rows=data.map(r=>header.map(k=>r[k]??null));
 assert(rows.every(r=>r.every(v=>v===null||['string','boolean'].includes(typeof v)||typeof v==='number'&&Number.isFinite(v))));
 const wb=Workbook.create(),sheet=wb.worksheets.add('Evidence');
 sheet.getRangeByIndexes(0,0,1,header.length).values=[header];
 for(let i=0;i<rows.length;i+=128)sheet.getRangeByIndexes(i+1,0,Math.min(128,rows.length-i),header.length).values=rows.slice(i,i+128);
 sheet.showGridLines=false;sheet.freezePanes.freezeRows(1);
 sheet.getRangeByIndexes(0,0,rows.length+1,header.length).format={font:{name:'Arial',size:10},columnWidth:25,rowHeight:24,verticalAlignment:'center'};
 sheet.getRangeByIndexes(0,0,1,header.length).format={font:{name:'Arial',size:10,bold:true},fill:'#E8EEF4',wrapText:true,rowHeight:90};
 wb.recalculate();
 const inspection=await wb.inspect({kind:'region',sheetId:sheet.name,range:previewColumns+'8',maxChars:2500,tableMaxRows:5,tableMaxCols:7});
 const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:10},maxChars:1500});
 const png=await wb.render({sheetName:sheet.name,range:previewColumns+'8',scale:1,format:'png'});
 await fs.writeFile(path.join(root,name+'_preview.png'),new Uint8Array(await png.arrayBuffer()));
 const csv=[header.map(quote).join(',')];
 for(let i=0;i<rows.length;i+=128){const values=sheet.getRangeByIndexes(i+1,0,Math.min(128,rows.length-i),header.length).values;
  for(let j=0;j<values.length;j++){assert.deepEqual(values[j].map(v=>v??null),rows[i+j]);csv.push(values[j].map(quote).join(','));}}
 await fs.writeFile(output,csv.join('\r\n')+'\r\n',{flag:'wx'});
 const reloaded=parseCSV(await fs.readFile(output,'utf8'));
 assert.equal(reloaded.length,rows.length+1);assert.deepEqual(reloaded[0],header);assert(reloaded.every(r=>r.length===header.length));
 for(let i=0;i<rows.length;i++)assert.deepEqual(reloaded[i+1],rows[i].map(v=>v==null?'':String(v)));
 return {output,rows:rows.length,columns:header.length,scalar_grid_verified:true,recalculated:true,roundtrip_verified:true,inspection,errors};
}
const outputs=[];
outputs.push(await author('sequence_before_after',sequenceRows,['run_label','action_identity','source_event','command_pre_step_tick','command_post_step_tick','command_pre_step_time_s','measured_response_tick'],'A1:G'));
outputs.push(await author('rr_target_actual_and_task_trace',trace,['run_label','episode_physics_tick','simulation_time_s','source_phase_id','RR_knee_nominal_request_deg','RR_knee_final_target_deg','RR_knee_actual_deg','RR_knee_e_tracking_actual_minus_final_deg'],'A1:H'));
await fs.writeFile(path.join(root,'after_csv_authoring_receipt.json'),JSON.stringify({outputs,bindings,sequence_input:sequenceInput,
 missing:'blank means unknown/not-sampled, never invented zero',units:'canonical joint deg, wheel rad/s, world m and m/s; body rad and rad/s',
 trace_scope:'tick0, every8 physics ticks and true terminal; full-rate extrema remain in each summary',
 clock_contract:{sequence_source_definition:sequence.command_time_definition,
  sequence_export_definition:'command_pre_step_time_s explicitly renames legacy command_time_s; pre-step tick t precedes the effect in post-step physical tick t+1, at 120Hz.',
  trace_definition:'episode_physics_tick and simulation_time_s are measured post-step coordinates (except true reset tick0); native_internal_tick is a different reset-offset clock and is never substituted.',
  owner_definition:'Owner observation_tick and decision_end_tick remain separately recorded inside JSON cells; neither is silently aligned to current trace measurement.',
  response_definition:'available_response_sample_tick is a retained measurement, not an exact response onset; measured_response_tick can be null or a separately labeled evaluator event.',
  sequence_source_limits:sequence.limits},
 attribution:'final minus mapper N before geometry is a total difference that can include nominal geometry, controller bias, slew and shared history; N is this run\'s actual unique mapper output, not an independent fresh FSM trajectory. Use explicit RR decomposition and exact same-prestate policy deltas. Native wheel signs are canonical times [-1,+1,-1,+1].',
 physics_or_learning_modified:false},null,2),{flag:'wx'});
console.log(JSON.stringify({outputs:outputs.map(({output,rows,columns})=>({output,rows,columns}))}));
