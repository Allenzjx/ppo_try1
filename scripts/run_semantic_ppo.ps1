[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)][ValidateSet('preflight','smoke','train','eval')][string]$Command,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{40}$')][string]$ExpectedHead,
    [ValidateSet('smoke','phase_suffix','full_episode')][string]$Stage = 'smoke',
    [ValidateRange(1,100000)][int]$Decisions,
    [ValidateRange(1,3000)][int]$MaxDecisions = 3000,
    [int]$Seed = 1001,
    [ValidateSet(1,8)][int]$NumEnvs = 1,
    [ValidateSet('v2','v3')][string]$SemanticVersion = 'v2',
    [ValidateSet('transfer_roles_v1','all_stage_acceptance_v1')][string]$ExperimentId,
    [ValidateSet('P01','P03','P04','P05','P06','P07','P08','P09','P10','P11','P12','P13')][string]$FromPhase = 'P01',
    [ValidateRange(0,1799)][int]$TeacherOffsetDecisions = 0,
    [ValidateSet('frozen_fsm','checkpoint_policy')][string]$PrefixSource = 'frozen_fsm',
    [switch]$NewMdpWarmStart,
    [ValidateSet('history_conditioned_heteroscedastic_log_v1')][string]$TargetPolicyVersion,
    [switch]$PolicyDistributionMigration,
    [string]$VectorSmokeEvidence,
    [string]$Checkpoint,
    [string]$ResumeMigration,
    [ValidateSet('legacy_fsm_eval','semantic_prior_eval','semantic_residual_eval')][string]$Mode = 'semantic_prior_eval',
    [ValidateSet('cpu','cuda:0')][string]$Device = 'cuda:0',
    [ValidateRange(1,100)][int]$CheckpointIntervalUpdates = 10
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not [string]::IsNullOrWhiteSpace($ExperimentId) -and $SemanticVersion -cne 'v3') {
    throw 'ExperimentId transfer_roles_v1 requires SemanticVersion v3'
}
if (-not [string]::IsNullOrWhiteSpace($TargetPolicyVersion) -and (
        -not $NewMdpWarmStart -or $Command -cne 'train' -or $SemanticVersion -cne 'v3' -or $NumEnvs -ne 1 -or
        [string]::IsNullOrWhiteSpace($Checkpoint) -or $PolicyDistributionMigration -or
        -not [string]::IsNullOrWhiteSpace($ResumeMigration) -or $PrefixSource -ceq 'checkpoint_policy')) {
    throw 'TargetPolicyVersion requires exclusive v3 N1 new-MDP training without a checkpoint-policy prefix'
}
if ($PolicyDistributionMigration -and ($Command -cne 'train' -or $SemanticVersion -cne 'v3' -or $NumEnvs -ne 1 -or
        [string]::IsNullOrWhiteSpace($Checkpoint) -or $NewMdpWarmStart -or -not [string]::IsNullOrWhiteSpace($ResumeMigration))) {
    throw 'PolicyDistributionMigration requires v3 N1 train with a checkpoint, exclusive of NewMdpWarmStart or ResumeMigration'
}
$project = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$python = 'C:\Users\kskzz\miniconda3\envs\env_isaaclab\python.exe'
$busy = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^(kit|isaac-sim)(\.exe)?$' -or
    ($_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match '(isaacsim|isaaclab|omni\.kit|wlr50_clean\.ppo\.(cli|semantic_cli))')
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
if ($SemanticVersion -eq 'v3' -and $kind -eq 'interface_smoke') { $kind = 'interface_checks' }
if ($ExperimentId -eq 'all_stage_acceptance_v1') {
    if ($kind -in @('interface_checks','prior_B','baseline_A')) { $kind = 'diagnostics' }
}
$artifactNamespace = if ([string]::IsNullOrWhiteSpace($ExperimentId)) { "ppo_semantic_$SemanticVersion" } else { "ppo_$ExperimentId" }
$runDir = Join-Path $project ("runs\$artifactNamespace\$kind\$runId")
$logDir = Join-Path $project ("runs\$artifactNamespace\$kind\${runId}_launcher")
[void](New-Item -ItemType Directory -Path $logDir)
$lockPath = Join-Path $project 'runs\ppo_semantic_v2\.single_process.lock'
[void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($lockPath))
$lock = [IO.File]::Open($lockPath, [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
$previousPath = $env:PYTHONPATH
$previousNoUser = $env:PYTHONNOUSERSITE
$previousHashSeed = $env:PYTHONHASHSEED
try {
    $env:PYTHONPATH = Join-Path $project 'src'
    $env:PYTHONNOUSERSITE = '1'
    $env:PYTHONHASHSEED = [string]$Seed
    $arguments = @('-P','-m','wlr50_clean.ppo.semantic_cli',$Command,'--run-dir',$runDir,
        '--expected-head',$ExpectedHead,'--stage',$Stage,'--seed',[string]$Seed,'--num-envs',[string]$NumEnvs,
        '--max-decisions',[string]$MaxDecisions,'--mode',$Mode,'--device',$Device,
        '--semantic-version',$SemanticVersion,'--from-phase',$FromPhase,
        '--teacher-offset-decisions',[string]$TeacherOffsetDecisions,
        '--prefix-source',$PrefixSource,
        '--checkpoint-interval-updates',[string]$CheckpointIntervalUpdates,'--headless')
    if ($NewMdpWarmStart) { $arguments += '--new-mdp-warm-start' }
    if (-not [string]::IsNullOrWhiteSpace($ExperimentId)) { $arguments += @('--experiment-id',$ExperimentId) }
    if (-not [string]::IsNullOrWhiteSpace($TargetPolicyVersion)) { $arguments += @('--target-policy-version',$TargetPolicyVersion) }
    if ($PolicyDistributionMigration) { $arguments += '--policy-distribution-migration' }
    if ($PSBoundParameters.ContainsKey('Decisions')) { $arguments += @('--decisions',[string]$Decisions) }
    if (-not [string]::IsNullOrWhiteSpace($VectorSmokeEvidence)) {
        $proof = if ([IO.Path]::IsPathRooted($VectorSmokeEvidence)) { $VectorSmokeEvidence } else { Join-Path $project $VectorSmokeEvidence }
        $arguments += @('--vector-smoke-evidence',[IO.Path]::GetFullPath($proof))
    }
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
