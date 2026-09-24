"""Same422/full12/Identity/HISTORY actor; only observed sigma changes."""
import torch
from rsl_rl.utils import unpad_trajectories
from .semantic_history_actor import history_conditioned_head, HISTORY_RHO
from .semantic_p05_capture_actor import p05_capture_request_history
from .semantic_p02_progress_actor import (
    SemanticP02ProgressHistoryMLPModel, validate_p02_progress_latent)
from .semantic_rear_cooperative_prep_sigma import cooperative_prep_effective_log_std


class SemanticRearCooperativePrepHistoryMLPModel(SemanticP02ProgressHistoryMLPModel):
    """No new parameters/buffers. Parent constructor validates unchanged ABI."""
    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        if not stochastic_output:
            # Exact old deterministic/HISTORY implementation, not re-created math.
            return super().forward(obs, masks=masks, hidden_state=hidden_state,
                                   stochastic_output=False)
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        validate_p02_progress_latent(latent)
        center, _ = p05_capture_request_history(latent[..., :389])
        head = history_conditioned_head(self.mlp(latent), center, HISTORY_RHO)
        log_std, _ = cooperative_prep_effective_log_std(head[..., 1, :], latent,
                                                      self.exploration_std_temperature)
        self.distribution.update(torch.stack((head[..., 0, :], log_std), dim=-2))
        return self.distribution.sample()

    def as_jit(self):
        raise NotImplementedError('cooperative prep HISTORY JIT export is not implemented')

    def as_onnx(self, verbose=False):
        raise NotImplementedError('cooperative prep HISTORY ONNX export is not implemented')
