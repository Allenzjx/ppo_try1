// Output-only, one streaming pass of this explicitly completed training audit.
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url)),root=path.resolve(here,'../..');
const read=p=>JSON.parse(fs.readFileSync(p,'utf8').replace(/^\uFEFF/,''));
const receipt=read(path.join(here,'training_P01_2048_actual.json')),run=receipt.runs[0];
assert.equal(run.lifecycle,'SUCCEEDED');assert.equal(run.increments.policy_decisions,2048);
const lifecycle=read(path.join(run.run,'run_manifest.json'));
assert.equal(lifecycle.lifecycle,'SUCCEEDED');assert(lifecycle.completed_at_utc);
const audit=path.join(run.run,'residual_and_projection_audit.jsonl');
const output=path.join(here,'training_2048_action_distribution.json');
const markdown=path.join(here,'training_2048_action_distribution.md');
for(const p of [output,markdown])assert(!fs.existsSync(p),'Do not overwrite a completed result');
const names=['front_left_hip','front_left_knee','front_right_hip','front_right_knee',
 'rear_left_hip','rear_left_knee','rear_right_hip','rear_right_knee',
 'front_left_ankle','front_right_ankle','rear_left_ankle','rear_right_ankle'];
const selected=['P01','P02','P03'];
const vector=v=>Array.isArray(v)&&v.length===12&&v.every(Number.isFinite);
function stats(xs){if(!xs.length)return null;const sorted=[...xs].sort((a,b)=>a-b),n=xs.length;
 const mean=xs.reduce((s,x)=>s+x,0)/n,rms=Math.sqrt(xs.reduce((s,x)=>s+x*x,0)/n);
 const q=p=>{const i=(n-1)*p,l=Math.floor(i),h=Math.ceil(i);return sorted[l]+(sorted[h]-sorted[l])*(i-l);};
 const count=t=>xs.filter(x=>Math.abs(x)>=t).length;
 return{n,min:sorted[0],max:sorted.at(-1),mean,population_std:Math.sqrt(xs.reduce((s,x)=>s+(x-mean)**2,0)/n),
  rms,p01:q(.01),p05:q(.05),p50:q(.5),p95:q(.95),p99:q(.99),
  abs_ge_2_count:count(2),abs_ge_2_fraction:count(2)/n,abs_ge_3_count:count(3),abs_ge_3_fraction:count(3)/n};}
const channel=()=>({mean:[],sigma:[],raw:[],innovation:[],z:[],projected:[],native:[],sign_flip:0,
 max_abs_raw_event:null,max_abs_z_event:null});
const groups=Object.fromEntries(selected.map(p=>[p,{decisions:0,physics_ticks:0,verified_ticks:0,
 actual_native_effect_ticks:0,channels:names.map(channel)}]));
let rows=0,previous=run.source.global_policy_decisions,first=null;
const phaseCounts={},updatePhaseCounts={};
for await(const line of readline.createInterface({input:fs.createReadStream(audit),crlfDelay:Infinity})){
 if(!line.trim())continue;const r=JSON.parse(line),a=r.applied_audit;
 assert.equal(r.global_policy_decision,++previous);rows++;
 const mu=r.old_distribution_mean_full12,sigma=r.old_distribution_std_full12,raw=r.raw_policy_action_full12;
 assert(vector(mu)&&vector(sigma)&&vector(raw)&&sigma.every(s=>s>0));
 assert.deepEqual(a.raw_policy_action_full12,raw);
 if(!first)first={global_policy_decision:previous,phase:a.phase_id,old_mean:mu,conditional_sigma:sigma,raw};
 phaseCounts[a.phase_id]=(phaseCounts[a.phase_id]??0)+1;
 const update=run.source.ppo_updates+Math.ceil(rows/run.configuration.rollout_policy_decisions);
 const uc=updatePhaseCounts[update]??={};uc[a.phase_id]=(uc[a.phase_id]??0)+1;
 if(!(a.phase_id in groups))continue;
 const g=groups[a.phase_id],native=a.actuator_target_effect_audit,summary=a.actuator_target_effect_audit_summary;
 assert(native?.verified===true&&summary?.all_ticks_verified===true);
 assert.deepEqual(native.raw_policy_action_full12,raw);assert.deepEqual(native.phase_mask_full12,Array(12).fill(1));
 assert.equal(summary.physics_ticks,a.physics_ticks);assert.equal(summary.verified_tick_count,a.physics_ticks);
 const delta=[...native.native_target_delta.servo_position_rad,...native.native_target_delta.wheel_velocity_rad_s];
 assert(vector(delta)&&vector(a.projected_residual_full12));
 g.decisions++;g.physics_ticks+=a.physics_ticks;g.verified_ticks+=summary.verified_tick_count;
 g.actual_native_effect_ticks+=summary.actual_native_effect_tick_count;
 g.channels.forEach((c,i)=>{const e=raw[i]-mu[i],z=e/sigma[i];assert(Number.isFinite(z));
  c.mean.push(mu[i]);c.sigma.push(sigma[i]);c.raw.push(raw[i]);c.innovation.push(e);c.z.push(z);
  c.projected.push(a.projected_residual_full12[i]);c.native.push(delta[i]);
  if(mu[i]!==0&&raw[i]*mu[i]<0)c.sign_flip++;
  const event={global_policy_decision:previous,ppo_update_intended:update,episode_decision:a.decision_count,
   endpoint_tick:a.physics_tick,raw:raw[i],conditional_mean:mu[i],conditional_sigma:sigma[i],innovation_z:z};
  if(!c.max_abs_raw_event||Math.abs(raw[i])>Math.abs(c.max_abs_raw_event.raw))c.max_abs_raw_event=event;
  if(!c.max_abs_z_event||Math.abs(z)>Math.abs(c.max_abs_z_event.innovation_z))c.max_abs_z_event=event;
 });
}
assert.equal(rows,run.increments.policy_decisions);assert.equal(previous,run.saved.global_policy_decisions);
for(const [p,n]of Object.entries(receipt.aggregate.phase_policy_decisions))assert.equal(phaseCounts[p]??0,n);
const completed=Object.fromEntries(Object.entries(groups).map(([p,g])=>[p,{
 decisions:g.decisions,physics_ticks:g.physics_ticks,verified_ticks:g.verified_ticks,
 actual_native_effect_ticks:g.actual_native_effect_ticks,
 pooled_z_descriptive_only:stats(g.channels.flatMap(c=>c.z)),
 channels:g.channels.map((c,i)=>{const m=stats(c.mean),e=stats(c.innovation);return{
  index:i,channel:names[i],conditional_mean: m,conditional_sigma:stats(c.sigma),raw:stats(c.raw),
  innovation: e,innovation_z:stats(c.z),
  innovation_rms_divided_by_conditional_mean_rms:m.rms>0?e.rms/m.rms:null,
  sampled_raw_opposite_conditional_mean_fraction:c.sign_flip/g.decisions,
  projected_residual_at_decision_endpoint:stats(c.projected),native_target_effect_at_decision_endpoint:stats(c.native),
  max_abs_raw_event:c.max_abs_raw_event,max_abs_z_event:c.max_abs_z_event};})}]));
const cFile=path.join(here,'p02_policy_recovery_checkpoint170240.json'),c=read(cFile);
assert.equal(c.purpose,'formal');assert.equal(c.checkpoint.saved_global_policy_decisions,170240);
assert.equal(c.runtime.source_git_commit,run.saved.source_git_commit);
assert(c.policy_execution.natural_P01_no_snapshot_verified&&c.policy_execution.full12_raw_and_native_mask_verified);
const comparison={source:cFile,checkpoint:170240,deterministic_conditional_mean:true,
 request_phase_counts:c.policy_execution.request_phase_counts,endpoint:c.endpoint.physics_tick,
 termination_source:c.endpoint.termination_source,task_success:c.endpoint.physical_task_success,
 channels:c.policy_execution.raw_latent_per_channel.map((m,i)=>({index:i,channel:names[i],min:m.min,max:m.max,
  max_abs:m.max_abs,mean:null,abs_ge_2_fraction:m.max_abs<2?0:null,abs_ge_3_fraction:m.max_abs<3?0:null})),
 scope:'All 94 issued C decisions are P01/P02. Reused 752-tick raw extrema, not re-read raw source. The extract stores no sum/mean or per-phase extrema; mean remains unavailable. Zero fraction is inferred only when max_abs is strictly below threshold.',
 same_state_or_frozen_training_distribution_comparison:false};
const result={schema:'wlr50_clean.completed_training_action_distribution.v1',created_at_utc:new Date().toISOString(),
 source:{run:run.run,audit,completed_training_receipt:path.join(here,'training_P01_2048_actual.json'),
  source_checkpoint:168192,saved_checkpoint:170240,head:run.saved.source_git_commit,
  audit_bytes:fs.statSync(audit).size,audit_streaming_passes_this_analysis:1,rows,first_decision:first},
 phase_counts_all:phaseCounts,phase_counts_by_intended_update:updatePhaseCounts,phases:completed,C170240:comparison,
 definitions:{samples:'One sample per credited policy decision and channel; not 120Hz repetitions.',
  conditional_sigma:'Recorded actual pre-step distribution standard deviation at that observation/history; not init_std, marginal raw std, stationary variance, or assumed 1.',
  innovation_z:'(sampled raw - recorded conditional mean) / recorded conditional sigma, using the same issued decision.',
  empirical_population_std:'Descriptive variability of these recorded samples across evolving states/history and 16 updates; not a stationary model.',
  saturation:'|raw|>=2 and >=3 imply |tanh(raw)|>=0.96402758 and >=0.99505475 before subsequent scale/slew/headroom projections. Not proof of native actuator hard-limit saturation.',
  native_effect:'Recorded last physics tick of each decision, final target minus same-tick zero-residual target; servo rad and wheel rad/s. Actual target effect is not measured torque or body response.',
  sign_flip:'Fraction raw*conditional_mean<0; descriptive exploration coverage, not a claim of beneficial/destructive learning.',
  pooled_z:'Pooled channel description only; correlated state selection and small phase counts preclude blanket sampling-quality/safety conclusions.'},
 changes:{production:false,checkpoint_loaded:false,simulator_launched:false,rho_or_sigma_changed:false},
 recommendation:'Prioritize additional natural-P01/early-task sampling with the current preserved distribution; 12 P01 and20 P03 samples are sparse. These conditional statistics and one deterministic failure do not identify a justified single rho/sigma change. Separately examine mean/native whole-body control with matched physical evidence.'};
fs.writeFileSync(output,JSON.stringify(result,null,2)+'\n',{flag:'wx'});
const f=x=>Number(x).toFixed(4),pct=x=>(100*x).toFixed(2)+'%';
const lines=['# Completed 2048 training: conditional action distribution', '',
 `One completed audit streaming pass; ${rows} decisions, source168192 → saved170240. No model load, simulation or production change.`, '',
 '| Phase | Decisions | Conditional sigma min–max (all channels) | Pooled z mean / population std | abs(raw)≥2 / ≥3 |',
 '|---|---:|---:|---:|---:|'];
for(const[p,g]of Object.entries(completed)){const ch=g.channels,z=g.pooled_z_descriptive_only,n=g.decisions*12;
 lines.push(`| ${p} | ${g.decisions} | ${f(Math.min(...ch.map(c=>c.conditional_sigma.min)))}–${f(Math.max(...ch.map(c=>c.conditional_sigma.max)))} | ${f(z.mean)} / ${f(z.population_std)} | ${pct(ch.reduce((s,c)=>s+c.raw.abs_ge_2_count,0)/n)} / ${pct(ch.reduce((s,c)=>s+c.raw.abs_ge_3_count,0)/n)} |`);}
lines.push('', '## Per-channel observed statistics', '',
 '| Phase/channel | sigma min / mean / max | conditional mean average | raw min / mean / max | abs(raw)≥2 / ≥3 | z mean / std / min / max |',
 '|---|---|---:|---|---|---|');
for(const[p,g]of Object.entries(completed))for(const x of g.channels)lines.push(`| ${p}/${x.channel} | ${f(x.conditional_sigma.min)} / ${f(x.conditional_sigma.mean)} / ${f(x.conditional_sigma.max)} | ${f(x.conditional_mean.mean)} | ${f(x.raw.min)} / ${f(x.raw.mean)} / ${f(x.raw.max)} | ${pct(x.raw.abs_ge_2_fraction)} / ${pct(x.raw.abs_ge_3_fraction)} | ${f(x.innovation_z.mean)} / ${f(x.innovation_z.population_std)} / ${f(x.innovation_z.min)} / ${f(x.innovation_z.max)} |`);
lines.push('', '## Interpretation and limits', '',
 '- Sigma is the recorded conditional innovation sigma, not marginal raw variability or a stationary variance. The actor leaves learned log-sigma unchanged when forming the HISTORY-conditioned mean (semantic_history_actor.py:29–63).',
 '- The tanh thresholds describe latent compression, not native saturation. Same-decision raw/native identity and all12 masks were verified; the JSON includes actual endpoint projected residual and native-target-effect statistics.',
 `- C170240 ended at tick${comparison.endpoint}/${comparison.termination_source}; all ${c.policy_execution.issued_decisions} issued P01/P02 conditional-mean actions have |raw|<2 (largest ${f(Math.max(...comparison.channels.map(x=>x.max_abs)))}). Its deterministic failure therefore does not require an extreme sampled innovation or tanh-tail saturation explanation.`,
 '- The existing C extract has raw ranges but no means; no C source rescan or invented mean was used. Its trajectory/checkpoint differs from individual training states and cannot isolate a learned motive or sigma effect.',
 '- First three phase sample counts are12/516/20; much of the block was P05 (1258). Prefer more natural-P01/early-task credit before a distribution change, while preserving Adam/normalizer/HISTORY and verifying real endpoints. Do not alter rho/sigma from this one failure.',
 '- Full per-channel quantiles, innovation ratios, exact tail counts and extreme-event decision bindings are in training_2048_action_distribution.json. Finite-run values are descriptive, not a formal normality or safety test.',
 '', `Source audit: ${audit}`, `C source receipt: ${cFile}`, '');
fs.writeFileSync(markdown,lines.join('\n'),{flag:'wx'});
console.log(JSON.stringify({output,markdown,rows,phases:Object.fromEntries(Object.entries(completed).map(([p,g])=>[p,{n:g.decisions,z:g.pooled_z_descriptive_only,channels:g.channels.map(x=>({channel:x.channel,sigma:[x.conditional_sigma.min,x.conditional_sigma.mean,x.conditional_sigma.max],raw_tail2:x.raw.abs_ge_2_fraction,raw_tail3:x.raw.abs_ge_3_fraction,noise_mean_rms:x.innovation_rms_divided_by_conditional_mean_rms}))}]))}));
