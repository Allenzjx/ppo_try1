"""422-input actor with the exact unchanged rear419 HISTORY and sigma mapping."""
import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import HeteroscedasticGaussianDistribution
from rsl_rl.utils import unpad_trajectories
from .semantic_history_actor import history_conditioned_head, HISTORY_RHO
from .semantic_p05_capture_actor import p05_capture_request_history
from .semantic_rear_policy_timing_actor import rear_policy_timing_effective_log_std
from .semantic_p02_progress_profile import P02_PROGRESS_OBSERVATION_LAYOUT


def validate_p02_progress_latent(latent):
    if latent.shape[-1] != 422 or not bool(torch.isfinite(latent).all()):
        raise ValueError('P02 progress requires finite422 observations')
    tail = latent[..., 419:422]
    if (not bool((tail[..., 0] >= 0).all())
            or not bool(((tail[..., 1] >= 0) & (tail[..., 1] <= 1)).all())
            or not bool(((tail[..., 2] == 0) | (tail[..., 2] == 1)).all())
            or not bool(((latent[..., 1] == 1).unsqueeze(-1) | (tail == 0)).all())):
        raise ValueError('P02 progress state range/phase disagrees')


def p02_progress_effective_log_std(head_log_std, latent, temperature=.25):
    validate_p02_progress_latent(latent)
    return rear_policy_timing_effective_log_std(head_log_std, latent[..., :419], temperature)


class SemanticP02ProgressHistoryMLPModel(MLPModel):
    def __init__(self, obs, obs_groups, obs_set, output_dim, hidden_dims=(256,256), activation='elu',
                 obs_normalization=False, distribution_cfg=None, observation_layout=None,
                 exploration_std_temperature=None):
        if (observation_layout != P02_PROGRESS_OBSERVATION_LAYOUT or output_dim != 12
                or exploration_std_temperature != .25 or obs_normalization is not False):
            raise ValueError('P02 progress requires explicit422/full12/Identity/quarter HISTORY')
        super().__init__(obs,obs_groups,obs_set,output_dim,hidden_dims,activation,obs_normalization,distribution_cfg)
        if self.obs_dim != 422 or type(self.distribution) is not HeteroscedasticGaussianDistribution:
            raise ValueError('P02 progress actor distribution/layout mismatch')
        self.observation_layout = observation_layout

    @property
    def exploration_std_temperature(self): return .25

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        obs = unpad_trajectories(obs,masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs,masks,hidden_state)
        validate_p02_progress_latent(latent)
        center,_ = p05_capture_request_history(latent[..., :389])
        head = history_conditioned_head(self.mlp(latent),center,HISTORY_RHO)
        if not stochastic_output:
            return self.distribution.deterministic_output(head)
        log_std,_ = p02_progress_effective_log_std(head[...,1,:],latent,self.exploration_std_temperature)
        self.distribution.update(torch.stack((head[...,0,:],log_std),dim=-2))
        return self.distribution.sample()

    def as_jit(self): raise NotImplementedError('P02 progress history JIT export is not implemented')
    def as_onnx(self,verbose=False): raise NotImplementedError('P02 progress history ONNX export is not implemented')
