[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory = $true)][ValidateSet('preflight','smoke','train','eval')][string]$Command,
    [Parameter(Mandatory = $true)][ValidatePattern('^[0-9a-f]{40}$')][string]$ExpectedHead,
    [ValidateSet('smoke','phase_suffix','full_episode')][string]$Stage = 'smoke',
    [ValidateRange(1,100000)][int]$Decisions,
    [ValidateRange(1,3000)][int]$MaxDecisions = 3000,
    [int]$Seed = 1001,
    [string]$Checkpoint,
    [ValidateSet('semantic_prior_eval','semantic_residual_eval')][string]$Mode = 'semantic_prior_eval',
    [ValidateSet('cpu','cuda:0')][string]$Device = 'cuda:0',
    [ValidateRange(1,100)][int]$CheckpointIntervalUpdates = 10
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
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
    if ($Mode -eq 'semantic_prior_eval') { 'prior_B' } elseif ($Seed -ge 3001 -and $Seed -le 3005) { 'locked_test' } else { 'validation' }
} }
$runId = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ') + '_g' + $ExpectedHead.Substring(0,12) + '_' + [Guid]::NewGuid().ToString('N')
$runDir = Join-Path $project ("runs\ppo_semantic_v2\$kind\$runId")
$logDir = Join-Path $project ("runs\ppo_semantic_v2\$kind\${runId}_launcher")
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
    $arguments = @('-P','-m','wlr50_clean.ppo.semantic_cli',$Command,'--run-dir',$runDir,
        '--expected-head',$ExpectedHead,'--stage',$Stage,'--seed',[string]$Seed,
        '--max-decisions',[string]$MaxDecisions,'--mode',$Mode,'--device',$Device,
        '--checkpoint-interval-updates',[string]$CheckpointIntervalUpdates,'--headless')
    if ($PSBoundParameters.ContainsKey('Decisions')) { $arguments += @('--decisions',[string]$Decisions) }
    if (-not [string]::IsNullOrWhiteSpace($Checkpoint)) {
        $checkpointPath = if ([IO.Path]::IsPathRooted($Checkpoint)) { $Checkpoint } else { Join-Path $project $Checkpoint }
        $arguments += @('--checkpoint',[IO.Path]::GetFullPath($checkpointPath))
    }
    $arguments | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $logDir 'arguments.json') -Encoding utf8
    & $python @arguments 1> (Join-Path $logDir 'stdout.log') 2> (Join-Path $logDir 'stderr.log')
    if ($LASTEXITCODE -ne 0) { throw "Semantic command failed; preserved run $runDir and logs $logDir" }
    $manifest = Get-Content -LiteralPath (Join-Path $runDir 'run_manifest.json') -Raw | ConvertFrom-Json
    if ($manifest.lifecycle -cne 'SUCCEEDED') { throw 'Semantic process did not finalize successful execution' }
    Write-Output $runDir
} finally {
    $env:PYTHONPATH = $previousPath
    $env:PYTHONNOUSERSITE = $previousNoUser
    $env:PYTHONHASHSEED = $previousHashSeed
    $lock.Dispose()
}
