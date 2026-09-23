[CmdletBinding(PositionalBinding = $false)]
param(
    [ValidateSet('eval')][string]$Command = 'eval',
    [ValidateSet('v2','v3')][string]$SemanticVersion = 'v2',
    [ValidateSet('transfer_roles_v1','all_stage_acceptance_v1','fsm_reference_p09_stable_v2','task_first_recovery_v1','non_residual_refine_v1','residual_rr_fix_v1','fl_capture_quality_v1','task_conditioned_hip_wheel_v1','p05_hip_only_continuation_v1','rr_capture_then_rl_transfer_v1')][string]$ExperimentId,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{40}$')][string]$ExpectedHead,
    [ValidateSet('smoke','phase_suffix','full_episode')][string]$Stage = 'smoke',
    [ValidateRange(1,100000)][int]$Decisions,
    [ValidateRange(1,3000)][int]$MaxDecisions = 3000,
    [ValidateSet(4001)][int]$Seed = 4001,
    [string]$Checkpoint,
    [ValidatePattern('^[a-z0-9][a-z0-9_-]{0,63}$')][string]$CheckpointOutputBranch,
    [string]$ResumeMigration,
    [switch]$StochasticPolicy,
    [ValidateRange(0,2147483647)][int]$PolicySeed,
    [ValidateSet('legacy_fsm_eval','semantic_prior_eval','semantic_residual_eval')][string]$Mode = 'semantic_prior_eval',
    [ValidateSet('cpu','cuda:0')][string]$Device = 'cuda:0',
    [ValidateRange(1,100)][int]$CheckpointIntervalUpdates = 10
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not [string]::IsNullOrWhiteSpace($ExperimentId) -and $SemanticVersion -cne 'v3') {
    throw 'ExperimentId requires SemanticVersion v3'
}
$project = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$python = 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe'
$busy = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^(kit|isaac-sim)(\.exe)?$' -or
    ($_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match '(isaacsim|isaaclab|omni\.kit|wlr50_clean\.ppo\.(cli|semantic_cli|semantic_video_cli))')
})
if ($busy.Count) { throw 'Another Isaac/PPO process is active; semantic runs are sequential' }
$head = @(& git -C $project rev-parse HEAD)
if ($LASTEXITCODE -ne 0 -or $head.Count -ne 1 -or ([string]$head[0]).Trim() -cne $ExpectedHead) { throw 'Pinned semantic HEAD mismatch' }
$dirty = @(& git -C $project status --porcelain=v1 --untracked-files=all -- src/wlr50_clean scripts configs artifacts/ppo_phase_v1_start pyproject.toml)
if ($LASTEXITCODE -ne 0 -or $dirty.Count) { throw 'Semantic runtime is not clean/committed' }
$kind = switch ($Command) { 'train' { 'train' }; 'smoke' { 'interface_smoke' }; 'preflight' { 'interface_smoke' }; 'eval' {
    if ($Mode -eq 'legacy_fsm_eval') { 'baseline_A' } elseif ($Mode -eq 'semantic_prior_eval') { 'prior_B' } elseif ($Seed -ge 3001 -and $Seed -le 3005) { 'locked_test' } else { 'validation' }
} }
$runId = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ') + '_g' + $ExpectedHead.Substring(0,12) + '_' + [Guid]::NewGuid().ToString('N')
$namespace = if ([string]::IsNullOrWhiteSpace($ExperimentId)) { "ppo_semantic_$SemanticVersion" } else { "ppo_$ExperimentId" }
$runDir = Join-Path $project ("runs\$namespace\video_eval\$kind\$runId")
$logDir = Join-Path $project ("runs\$namespace\video_eval\$kind\${runId}_launcher")
[void](New-Item -ItemType Directory -Path $logDir)
$lockPath = Join-Path $project 'runs\ppo_semantic_v2\.single_process.lock'
$lock = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
$previousPath = $env:PYTHONPATH
$previousNoUser = $env:PYTHONNOUSERSITE
$previousHashSeed = $env:PYTHONHASHSEED
try {
    $env:PYTHONPATH = Join-Path $project 'src'
    $env:PYTHONNOUSERSITE = '1'
    $env:PYTHONHASHSEED = [string]$Seed
    $arguments = @('-P','-m','wlr50_clean.ppo.semantic_video_cli',$Command,'--run-dir',$runDir,
        '--semantic-version',$SemanticVersion,
        '--expected-head',$ExpectedHead,'--stage',$Stage,'--seed',[string]$Seed,
        '--max-decisions',[string]$MaxDecisions,'--mode',$Mode,'--device',$Device,
        '--checkpoint-interval-updates',[string]$CheckpointIntervalUpdates,'--no-headless')
    if (-not [string]::IsNullOrWhiteSpace($ExperimentId)) { $arguments += @('--experiment-id',$ExperimentId) }
    if ($StochasticPolicy) { $arguments += '--stochastic-policy' }
    if ($PSBoundParameters.ContainsKey('PolicySeed')) { $arguments += @('--policy-seed',[string]$PolicySeed) }
    if ($PSBoundParameters.ContainsKey('Decisions')) { $arguments += @('--decisions',[string]$Decisions) }
    if (-not [string]::IsNullOrWhiteSpace($CheckpointOutputBranch)) { $arguments += @('--checkpoint-output-branch',$CheckpointOutputBranch) }
    if (-not [string]::IsNullOrWhiteSpace($Checkpoint)) {
        $checkpointPath = if ([IO.Path]::IsPathRooted($Checkpoint)) { $Checkpoint } else { Join-Path $project $Checkpoint }
        $arguments += @('--checkpoint',[IO.Path]::GetFullPath($checkpointPath))
    }
    if (-not [string]::IsNullOrWhiteSpace($ResumeMigration)) {
        $migrationPath = if ([IO.Path]::IsPathRooted($ResumeMigration)) { $ResumeMigration } else { Join-Path $project $ResumeMigration }
        $arguments += @('--resume-migration',[IO.Path]::GetFullPath($migrationPath))
    }
    $arguments | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logDir 'arguments.json') -Encoding utf8
    & $python @arguments 1> (Join-Path $logDir 'stdout.log') 2> (Join-Path $logDir 'stderr.log')
    if ($LASTEXITCODE -ne 0) { throw "Semantic command failed; preserved run $runDir and logs $logDir" }
    $manifest = Get-Content -LiteralPath (Join-Path $runDir 'run_manifest.json') -Raw | ConvertFrom-Json
    if ($manifest.lifecycle -cnotin @('SUCCEEDED','STOPPED_AT_VERIFIED_UPDATE_BOUNDARY')) { throw 'Semantic process did not finalize successful execution or verified graceful stop' }
    Write-Output $runDir
} finally {
    $env:PYTHONPATH = $previousPath
    $env:PYTHONNOUSERSITE = $previousNoUser
    $env:PYTHONHASHSEED = $previousHashSeed
    $lock.Dispose()
}
