[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $Source,

    [Parameter(Mandatory = $true)]
    [string] $Destination,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[0-9a-f]{40}$')]
    [string] $ExpectedHead,

    [string] $Python = 'python',

    [switch] $Execute
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Get-Sha256([string] $Path) {
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Require([bool] $Condition, [string] $Message) {
    if (-not $Condition) { throw $Message }
}

$outputRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $outputRoot '..\..')).Path
$reviewRoot = [IO.Path]::GetFullPath((Join-Path $outputRoot 'video_review'))
$sourcePath = (Resolve-Path -LiteralPath $Source).Path
$sourceInfo = Get-Item -LiteralPath $sourcePath
Require $sourceInfo.PSIsContainer 'Source must be a directory.'
$runRoot = Split-Path -Parent $sourcePath
$sourceManifestPath = Join-Path $sourcePath 'semantic_video_source_manifest.json'
$runManifestPath = Join-Path $runRoot 'run_manifest.json'
Require ((Test-Path -LiteralPath $sourceManifestPath -PathType Leaf) -and
         (Test-Path -LiteralPath $runManifestPath -PathType Leaf)) `
        'Source/run manifests are absent; the source is not sealed.'

$sourceManifest = Get-Content -Raw -LiteralPath $sourceManifestPath | ConvertFrom-Json
$runManifest = Get-Content -Raw -LiteralPath $runManifestPath | ConvertFrom-Json
Require ([bool]$runManifest.completed_at_utc -and $runManifest.lifecycle -ne 'RUNNING') `
        'Run is still active or lacks a closed lifecycle.'
Require ($sourceManifest.schema -eq 'wlr50_clean.semantic_video_source.v1' -and
         $sourceManifest.experiment_id -eq 'rr_rl_timing_policy_learning_v1' -and
         $sourceManifest.role -eq 'C' -and $sourceManifest.from_phase -eq 'P01' -and
         $sourceManifest.fresh_process_single_episode -eq $true -and
         [int]$sourceManifest.episode_count -eq 1 -and
         [int]$sourceManifest.optimizer_updates -eq 0) `
        'Source is not one sealed zero-update natural-P01 C evaluation.'

$proof = $sourceManifest.checkpoint_load_provenance
Require ($null -ne $proof -and $proof.checkpoint_loaded_and_verified -eq $true -and
         $sourceManifest.policy_sampling_mode -eq 'deterministic_conditional_mean') `
        'Source does not prove deterministic loading of a saved checkpoint.'
$binding = $proof.source
Require ($null -ne $binding) 'Source manifest lacks checkpoint source binding.'
$checkpointPath = (Resolve-Path -LiteralPath ([string]$binding.checkpoint)).Path
$sidecarPath = (Resolve-Path -LiteralPath ([string]$binding.manifest)).Path
$checkpoint = Get-Item -LiteralPath $checkpointPath
$sidecar = Get-Item -LiteralPath $sidecarPath
Require (-not $checkpoint.PSIsContainer -and -not $sidecar.PSIsContainer) `
        'Checkpoint binding does not name two files.'
$expectedSidecar = Join-Path $checkpoint.DirectoryName ($checkpoint.BaseName + '_manifest.json')
Require ([IO.Path]::GetFullPath($sidecarPath) -eq [IO.Path]::GetFullPath($expectedSidecar)) `
        'Bound sidecar is not the checkpoint companion manifest.'

$checkpointSha = Get-Sha256 $checkpointPath
$sidecarSha = Get-Sha256 $sidecarPath
Require ($checkpointSha -eq [string]$binding.checkpoint_sha256 -and
         $sidecarSha -eq [string]$binding.manifest_sha256) `
        'Checkpoint/sidecar bytes differ from sealed source provenance.'
$metadata = Get-Content -Raw -LiteralPath $sidecarPath | ConvertFrom-Json
Require ($metadata.checkpoint_sha256 -eq $checkpointSha -and
         [IO.Path]::GetFullPath([string]$metadata.checkpoint_path) -eq
            [IO.Path]::GetFullPath($checkpointPath) -and
         $metadata.save_load_round_trip -eq $true) `
        'Checkpoint sidecar lacks exact path/SHA/round-trip identity.'

$globalDecisions = [int64]$metadata.global_policy_decisions
$ppoUpdates = [int64]$metadata.ppo_updates
$optimizerSteps = [int64]$metadata.optimizer_steps
Require ($globalDecisions -ge 0 -and $ppoUpdates -ge 0 -and $optimizerSteps -ge 0 -and
         [int64]$proof.saved_global_policy_decisions -eq $globalDecisions) `
        'Checkpoint counters disagree with the source load proof.'

$destinationPath = [IO.Path]::GetFullPath($Destination)
Require ($destinationPath.StartsWith($reviewRoot + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase)) `
        'Destination must be a new child of this namespace video_review directory.'
Require (-not (Test-Path -LiteralPath $destinationPath)) `
        'Destination already exists; immutable review outputs are never overwritten.'

$sourceManifestSha = Get-Sha256 $sourceManifestPath
$runManifestSha = Get-Sha256 $runManifestPath
$step = $globalDecisions
$auditJson = Join-Path $outputRoot ("CP{0}_cooperative_RR_RL_window_audit.json" -f $step)
$auditMd = Join-Path $outputRoot ("CP{0}_cooperative_RR_RL_window_audit.md" -f $step)
Require (-not (Test-Path -LiteralPath $auditJson) -and
         -not (Test-Path -LiteralPath $auditMd)) `
        'Audit outputs already exist; choose a genuinely new checkpoint/source.'

$exporter = Join-Path $outputRoot 'export_policy_rear_no_assist_video.py'
$analyzer = Join-Path $outputRoot 'analyze_cooperative_rr_rl_window.py'
Require ((Test-Path -LiteralPath $exporter -PathType Leaf) -and
         (Test-Path -LiteralPath $analyzer -PathType Leaf)) `
        'Existing exporter/analyzer is missing.'

$common = @(
    '--source', $sourcePath,
    '--source-manifest-sha256', $sourceManifestSha,
    '--source-run-manifest-sha256', $runManifestSha,
    '--expected-head', $ExpectedHead,
    '--checkpoint', $checkpointPath,
    '--checkpoint-sha256', $checkpointSha,
    '--checkpoint-manifest-sha256', $sidecarSha,
    '--expected-global-policy-decisions', [string]$globalDecisions,
    '--expected-ppo-updates', [string]$ppoUpdates,
    '--expected-optimizer-steps', [string]$optimizerSteps,
    '--expected-new-auxiliary-updates', '0'
)
$exportArgs = @($exporter, '--destination', $destinationPath) + $common
$auditArgs = @($analyzer) + $common + @('--output-json', $auditJson, '--output-md', $auditMd)

$plan = [ordered]@{
    sealed_source = $sourcePath
    source_manifest_sha256 = $sourceManifestSha
    source_run_manifest_sha256 = $runManifestSha
    checkpoint = $checkpointPath
    checkpoint_sha256 = $checkpointSha
    checkpoint_manifest = $sidecarPath
    checkpoint_manifest_sha256 = $sidecarSha
    counters_from_bound_sidecar = [ordered]@{
        global_policy_decisions = $globalDecisions
        ppo_updates = $ppoUpdates
        optimizer_steps = $optimizerSteps
    }
    exact_runtime_head = $ExpectedHead
    destination = $destinationPath
    rear_task_assist = 'OFF (strict exporter validation)'
    front_FL_capture_assist = 'ON (strict exporter validation)'
    new_auxiliary_updates = 0
    export_argv = @($Python) + $exportArgs
    audit_argv = @($Python) + $auditArgs
    execute = [bool]$Execute
}
$plan | ConvertTo-Json -Depth 8

if (-not $Execute) {
    Write-Host 'DRY_RUN_ONLY: add -Execute after the source is sealed and this plan is reviewed.'
    exit 0
}

$env:CUDA_VISIBLE_DEVICES = '-1'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'

& $Python @exportArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python @auditArgs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host ("DELIVERY_COMPLETE CP{0}: {1}" -f $step, $destinationPath)
