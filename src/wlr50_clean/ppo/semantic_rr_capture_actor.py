"""RR assistance is observable; its final actuator target is never a PPO sample."""
import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import HeteroscedasticGaussianDistribution
from rsl_rl.utils import unpad_trajectories
from .semantic_history_actor import history_conditioned_head, HISTORY_RHO
from .semantic_p05_capture_actor import p05_capture_request_history
from .semantic_receiving_wheel_sigma import receiving_wheel_effective_log_std
from .semantic_rr_capture_profile import RR_CAPTURE_OBSERVATION_DIM, RR_CAPTURE_OBSERVATION_LAYOUT


class SemanticRRCaptureHistoryMLPModel(MLPModel):
    def __init__(self,obs,obs_groups,obs_set,output_dim,hidden_dims=(256,256),activation='elu',
                 obs_normalization=False,distribution_cfg=None,observation_layout=None,
                 exploration_std_temperature=None):
        if (observation_layout != RR_CAPTURE_OBSERVATION_LAYOUT or output_dim != 12
                or exploration_std_temperature != .25 or obs_normalization is not False):
            raise ValueError('RR capture requires explicit appended layout/full12/Identity/quarter HISTORY')
        super().__init__(obs,obs_groups,obs_set,output_dim,hidden_dims,activation,obs_normalization,distribution_cfg)
        if self.obs_dim != RR_CAPTURE_OBSERVATION_DIM or type(self.distribution) is not HeteroscedasticGaussianDistribution:
            raise ValueError('RR capture actor distribution/layout mismatch')
        self.observation_layout=observation_layout

    @property
    def exploration_std_temperature(self):return .25

    def forward(self,obs,masks=None,hidden_state=None,stochastic_output=False):
        obs=unpad_trajectories(obs,masks) if masks is not None and not self.is_recurrent else obs
        latent=self.get_latent(obs,masks,hidden_state)
        center,_=p05_capture_request_history(latent[...,:389])
        head=history_conditioned_head(self.mlp(latent),center,HISTORY_RHO)
        if not stochastic_output:return self.distribution.deterministic_output(head)
        log_std,_=receiving_wheel_effective_log_std(head[...,1,:],latent[...,:372],.25)
        self.distribution.update(torch.stack((head[...,0,:],log_std),dim=-2))
        return self.distribution.sample()

    def as_jit(self):raise NotImplementedError('RR capture history export is not implemented')
    def as_onnx(self,verbose=False):raise NotImplementedError('RR capture history export is not implemented')
