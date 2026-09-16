// Small output-only test mirror, no run/checkpoint/history copy.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url)),root=path.resolve(here,'../..');
const target=path.join(here,'temperature_combined_candidate');
function copy(from,rel){const dest=path.join(target,rel);fs.mkdirSync(path.dirname(dest),{recursive:true});fs.copyFileSync(path.join(from,rel),dest);}
for(const n of ['semantic_training','semantic_cli']) copy(path.join(here,'temperature_loader_candidate'),`src/wlr50_clean/ppo/${n}.py`);
copy(path.join(here,'temperature_loader_candidate'),'tests/unit/test_semantic_temperature_loader_cli.py');
for(const n of ['semantic_history_actor','semantic_policy_distribution']) copy(path.join(here,'temperature_actor_candidate'),`src/wlr50_clean/ppo/${n}.py`);
for(const n of ['semantic_migration','semantic_reward']) copy(root,`src/wlr50_clean/ppo/${n}.py`);
for(const n of ['stage_task_spec.yaml','execution_profile.yaml','reward_config.yaml','observation_schema.json','action_schema.json','quality_score.yaml'])
 copy(root,`configs/ppo_fsm_reference_p09_stable_v2/${n}`);
for(const v of ['v2','v3']) copy(root,`configs/ppo_semantic_${v}/reward_config.yaml`);
copy(root,'scripts/run_semantic_ppo.ps1');
console.log(JSON.stringify({target,production_written:false}));
