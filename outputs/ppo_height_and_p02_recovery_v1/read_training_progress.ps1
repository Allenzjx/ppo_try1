param([Parameter(Mandatory=$true)][string]$Run)
$ErrorActionPreference='Stop'
function Read-LastRecord([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $null }
    try {
        $records=@(Get-Content -LiteralPath $Path -Tail 1 | ConvertFrom-Json)
        if ($records.Count) { return $records[-1] }
    } catch {
        # An active writer may not yet have completed its final JSON record.
        return $null
    }
    return $null
}
$update=Read-LastRecord (Join-Path $Run 'optimizer_updates.jsonl')
$decision=Read-LastRecord (Join-Path $Run 'residual_and_projection_audit.jsonl')
$episode=Read-LastRecord (Join-Path $Run 'completed_episodes.jsonl')
[ordered]@{
    utc=[DateTime]::UtcNow.ToString('o')
    scope='Read-only live progress; not a sealed result or saved-checkpoint claim'
    last_completed_update=if ($update) { [ordered]@{
        policy_decisions=$update.global_policy_decisions
        ppo_update=$update.ppo_update
        optimizer_steps_in_update=$update.optimizer_steps
        actor_changed=$update.actor_parameters_changed
        finite_nonzero_gradient=$update.finite_nonzero_gradient_observed
        effective_lr=$update.optimizer_learning_rate
    }} else { $null }
    last_written_decision=if ($decision) { [ordered]@{
        global_policy_decision=$decision.global_policy_decision
        phase=$decision.applied_audit.phase_id
        end_phase=$decision.applied_audit.end_phase_id
        episode_time_s=$decision.applied_audit.sim_time_s
        terminal=$decision.terminal
        actual_phase_mask=$decision.applied_audit.actuator_target_effect_audit.phase_mask_full12
    }} else { $null }
    last_completed_episode=if ($episode) { [ordered]@{
        index=$episode.episode_index
        policy_decisions=$episode.policy_decisions
        duration_s=$episode.duration_s
        phase=$episode.terminal_info.phase_id
        termination_reason=$episode.termination_reason
        task_success=$episode.task_success
        terminal_bootstrap=$episode.terminal_info.terminal_bootstrap_allowed
    }} else { $null }
} | ConvertTo-Json -Depth 6 -Compress
