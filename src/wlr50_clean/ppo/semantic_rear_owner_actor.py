"""439-input actor, unchanged conditional Gaussian on its existing422 prefix."""
import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import HeteroscedasticGaussianDistribution
from rsl_rl.utils import unpad_trajectories
from .semantic_history_actor import history_conditioned_head, HISTORY_RHO
from .semantic_p05_capture_actor import p05_capture_request_history
from .semantic_p02_progress_actor import validate_p02_progress_latent
from .semantic_rear_cooperative_prep_sigma import cooperative_prep_effective_log_std
from .semantic_rear_owner_profile import REAR_OWNER_OBSERVATION_LAYOUT


def validate_rear_owner_latent(latent):
    if latent.shape[-1] != 439 or not bool(torch.isfinite(latent).all()):
        raise ValueError('finite439 rear owner observation required')
    validate_p02_progress_latent(latent[..., :422])
    tail = latent[..., 422:]
    if not bool(((tail[..., 8:] == 0) | (tail[..., 8:] == 1)).all()):
        raise ValueError('rear owner flags must be binary')
    active = tail[..., 8:12]
    if (not bool(((active == 1) | (tail[..., :4] == 0)).all())
            or not bool(((active == 1) | (tail[..., 4:8] == 0)).all())):
        raise ValueError('owner state/anchor invariant differs')


def rear_owner_effective_log_std(head_log_std, latent, temperature=.25):
    validate_rear_owner_latent(latent)
    result, evidence = cooperative_prep_effective_log_std(head_log_std, latent[..., :422], temperature)
    evidence = dict(evidence)
    evidence['rear_owner_observed_features'] = latent[..., 422:439]
    return result, evidence


class SemanticRearOwnerRecoveryHistoryMLPModel(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(256,256), activation='elu',
                 obs_normalization=False, distribution_cfg=None, observation_layout=None,
                 exploration_std_temperature=None):
        if (observation_layout != REAR_OWNER_OBSERVATION_LAYOUT or output_dim != 12
                or exploration_std_temperature != .25 or obs_normalization is not False):
            raise ValueError('rear owner actor requires explicit439/full12/Identity/quarter HISTORY')
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg)
        if self.obs_dim != 439 or type(self.distribution) is not HeteroscedasticGaussianDistribution:
            raise ValueError('rear owner actor distribution/layout mismatch')
        self.observation_layout = observation_layout

    @property
    def exploration_std_temperature(self): return .25

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        validate_rear_owner_latent(latent)
        center, _ = p05_capture_request_history(latent[..., :389])
        head = history_conditioned_head(self.mlp(latent), center, HISTORY_RHO)
        if not stochastic_output:
            return self.distribution.deterministic_output(head)
        log_std, _ = rear_owner_effective_log_std(head[..., 1, :], latent, self.exploration_std_temperature)
        self.distribution.update(torch.stack((head[..., 0, :], log_std), dim=-2))
        return self.distribution.sample()

    def as_jit(self): raise NotImplementedError('rear owner HISTORY JIT export is not implemented')
    def as_onnx(self, verbose=False): raise NotImplementedError('rear owner HISTORY ONNX export is not implemented')
