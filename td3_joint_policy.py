"""Custom TD3 policy with a deterministic feature mapper for Craftium action scores."""

import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.td3.policies import Actor, TD3Policy


class JointGaussianMapperActor(Actor):
    """
    Deterministic TD3 actor that passes CNN features through a trainable mapper
    to produce Craftium action scores. No internal Gaussian sampling — consistent
    with TD3's deterministic policy assumption.

    Output layout expected by the env wrapper:
    - first N dims: mapped action scores in [-1, 1] (N = num_discrete_actions)
    - last 2 dims: continuous mouse x/y in [-1, 1]
    """

    def __init__(self, *args, latent_hidden_size: int = 128, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(self.action_space, spaces.Box):
            raise TypeError("JointGaussianMapperActor requires a Box action space")

        action_dim = self.action_space.shape[0]
        if action_dim < 3:
            raise ValueError("Action dim must be >= 3 (at least one discrete action + mouse x/y)")
        # -2 because last 2 are reserved for mouse control, which is continuous and not part of the discrete action set
        self.num_discrete_actions = action_dim - 2

        # Linear projection from CNN features to the discrete-action latent space.
        # Kept separate from the mapper so the projection is directly supervised by the TD3 actor loss.
        self.mean_head = nn.Linear(self.features_dim, self.num_discrete_actions)
        # Mouse control is continuous and passed through directly, so it has its own head.
        self.mouse_head = nn.Linear(self.features_dim, 2)
        # The mapper adds representational depth between the feature projection and the final action
        # scores without introducing stochasticity.
        self.mapper = nn.Sequential(
            nn.Linear(self.num_discrete_actions, latent_hidden_size),
            nn.ReLU(),
            nn.Linear(latent_hidden_size, latent_hidden_size),
            nn.ReLU(),
            nn.Linear(latent_hidden_size, self.num_discrete_actions),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs, self.features_extractor)

        # Deterministic feature projection — no sampling, consistent with TD3's
        # deterministic policy assumption. Exploration comes from NormalActionNoise only.
        latent = th.tanh(self.mean_head(features))

        # Trainable mapper from feature projection to action scores.
        action_scores = th.tanh(self.mapper(latent))
        mouse = th.tanh(self.mouse_head(features))

        return th.cat([action_scores, mouse], dim=1)


class JointGaussianMapperTD3Policy(TD3Policy):
    """TD3 policy that uses JointGaussianMapperActor as its actor network."""

    def make_actor(self, features_extractor=None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return JointGaussianMapperActor(**actor_kwargs).to(self.device)
