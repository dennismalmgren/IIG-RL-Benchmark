# Copyright 2022 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PPO with entmax/sparsemax policies (Tsallis-α)."""

import time
import warnings
import numpy as np
import torch
from torch import nn, optim
from torch.distributions.categorical import Categorical  # only used for fallback
from open_spiel.python.rl_agent import StepOutput
from utils import log_to_csv

INVALID_ACTION_PENALTY = -1e9


# ------------------------ utils ------------------------

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def apply_action_mask(logits, legal_actions_mask, mask_value):
    """Replace invalid-action logits by a large negative number."""
    return torch.where(legal_actions_mask.bool(), logits, mask_value)


# ---------- entmax / sparsemax (with safe fallbacks) ----------

def _sparsemax(logits, dim=-1):
    """Sparsemax probabilities (Martins & Astudillo, 2016)."""
    z = logits
    z_sorted, _ = torch.sort(z, descending=True, dim=dim)
    z_cumsum = torch.cumsum(z_sorted, dim=dim)
    ks = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    ks = ks.view(*([1] * (z.dim() - 1)), -1)
    cond = 1 + ks * z_sorted > z_cumsum
    k = cond.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (torch.gather(z_cumsum, dim, k - 1) - 1) / k
    p = torch.clamp(z - tau, min=0.0)
    return p / (p.sum(dim=dim, keepdim=True) + 1e-12)


def _entmax_probs(logits, alpha: float, dim=-1):
    """
    Return entmax probabilities for given alpha.
    - If 'entmax' package is available, use it (entmax_bisect/entmax15/sparsemax).
    - Else: alpha==2 -> built-in sparsemax; otherwise fall back to softmax (warn once).
    """
    try:
        from entmax import entmax_bisect, entmax15, sparsemax
        if abs(alpha - 2.0) < 1e-9:
            return sparsemax(logits, dim=dim)
        elif abs(alpha - 1.5) < 1e-6:
            return entmax15(logits, dim=dim)
        else:
            return entmax_bisect(logits, alpha=alpha, dim=dim)
    except Exception:
        if abs(alpha - 2.0) < 1e-9:
            return _sparsemax(logits, dim=dim)
        if not hasattr(_entmax_probs, "_warned"):
            warnings.warn(
                "entmax package not found; using softmax fallback for alpha!=2. "
                "Run `pip install entmax` for true entmax.",
                RuntimeWarning,
            )
            _entmax_probs._warned = True
        return torch.softmax(logits, dim=dim)


def _tsallis_entropy_from_probs(probs, alpha: float, dim=-1):
    """Tsallis-α entropy; α→1 recovers Shannon."""
    if abs(alpha - 1.0) < 1e-9:
        return -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=dim)
    return (1.0 - torch.pow(probs, alpha).sum(dim=dim)) / (alpha - 1.0)

class EntmaxCategorical:
    """
    Minimal distribution wrapper backed by entmax probabilities.
    - alpha controls the probability transform (alpha=2 -> sparsemax).
    - entropy_kind controls the entropy used for the bonus/logging.
      Use 'shannon' to always compute Shannon entropy, regardless of alpha.
    Supports: sample(), log_prob(), entropy().
    """
    def __init__(self, logits=None, probs=None, alpha=1.5, dim=-1, entropy_kind: str = "tsallis"):
        assert (logits is None) != (probs is None), "Provide logits OR probs"
        self.alpha = float(alpha)
        self.dim = dim
        self.entropy_kind = entropy_kind.lower()

        if probs is None:
            self._probs = _entmax_probs(logits, alpha=self.alpha, dim=dim)
        else:
            self._probs = probs / (probs.sum(dim=dim, keepdim=True) + 1e-12)

    @property
    def probs(self):
        return self._probs

    def sample(self):
        # multinomial supports exact zeros
        return torch.multinomial(self._probs, num_samples=1).squeeze(-1)

    def log_prob(self, action):
        p = self._probs.gather(self.dim, action.unsqueeze(self.dim)).squeeze(self.dim)
        return (p.clamp_min(1e-12)).log()

    def entropy(self):
        p = self._probs.clamp_min(1e-12)
        if self.entropy_kind == "shannon":
            # Shannon entropy (well-defined with zeros using 0*log 0 := 0)
            return -(p * p.log()).sum(dim=self.dim)
        elif self.entropy_kind in ("tsallis", "tsallis_alpha", "tsallis-α", "tsallis-a"):
            # Tsallis-α matches the entmax geometry
            return (1.0 - torch.pow(p, self.alpha).sum(dim=self.dim)) / (self.alpha - 1.0)
        else:
            raise ValueError(f"Unknown entropy_kind '{self.entropy_kind}'. Use 'shannon' or 'tsallis'.")

class GatedSoftmaxCategorical:
    """
    Entmax-gated categorical:
      1) Gate support S with entmax_alpha (alpha>1 -> sparse).
      2) Do softmax on logits restricted to S (exact zeros off-support).

    Pros:
      - True sparsity from a principled gate (Tsallis/entmax).
      - Smooth, dense gradients inside support (stable for PPO).
      - Guarantees absolute continuity for sampled actions across updates.

    Args:
      logits: tensor [..., K]
      alpha:  entmax alpha used only for gating (alpha=1 -> softmax gate / dense)
      dim:    action dim (default -1)
      legal_mask: optional bool mask [..., K] (illegal=False)
      softmax_temp: optional temperature for the *restricted* softmax (default 1.0)
      entropy_kind: "shannon" or "tsallis" (Tsallis uses same alpha by default)
      gate_eps: threshold to treat extremely small gate probs as zero (default 0.0 exact)
      mask_value: big negative for masking (default -1e9)

    Exposes:
      .probs        : final probabilities after restricted softmax
      .gate_probs   : entmax gating probabilities
      .support_mask : boolean mask of active actions
    """
    def __init__(
        self,
        logits: torch.Tensor,
        alpha: float = 1.5,
        dim: int = -1,
        legal_mask: torch.Tensor | None = None,
        softmax_temp: float = 1.0,
        entropy_kind: str = "shannon",
        gate_eps: float = 0.0,
        mask_value: float = -1e9,
    ):
        assert logits is not None, "Provide logits"
        self.dim = dim
        self.alpha = float(alpha)
        self.softmax_temp = float(softmax_temp)
        self.entropy_kind = entropy_kind.lower()

        # 1) entmax gate (probabilities)
        gate = _entmax_probs(logits, alpha=self.alpha, dim=dim)  # [..., K]
        if gate_eps > 0.0:
            support = gate > gate_eps
        else:
            # exact entmax support (piecewise-linear; true zeros)
            support = gate > 0

        # merge with legal mask if provided
        if legal_mask is not None:
            support = support & legal_mask

        # Ensure each row has at least 1 active action (numerical safety).
        # If a row is all-False (e.g., all illegal), fall back to legal_mask or argmax.
        need_fix = ~support.any(dim=dim, keepdim=True)
        if need_fix.any():
            if legal_mask is not None:
                support = torch.where(need_fix, legal_mask, support)
            else:
                # activate the argmax logit in those rows
                top = logits.argmax(dim=dim, keepdim=True)
                fix = torch.zeros_like(logits, dtype=torch.bool).scatter(dim, top, True)
                support = torch.where(need_fix, fix, support)

        # 2) restricted softmax on support (exact zeros off-support)
        restricted_logits = torch.where(support, logits / self.softmax_temp, torch.tensor(mask_value, device=logits.device, dtype=logits.dtype))
        probs = torch.softmax(restricted_logits, dim=dim)

        # re-zero illegal (paranoia) & renormalize
        probs = torch.where(support, probs, torch.zeros_like(probs))
        probs = probs / (probs.sum(dim=dim, keepdim=True) + 1e-12)

        self._probs = probs
        self.gate_probs = gate.detach()          # for logging/analysis
        self.support_mask = support.detach()     # for logging/analysis

    @property
    def probs(self):
        return self._probs

    def sample(self):
        # multinomial handles exact zeros reliably
        return torch.multinomial(self._probs, num_samples=1).squeeze(-1)

    def log_prob(self, action: torch.Tensor):
        p = self._probs.gather(self.dim, action.unsqueeze(self.dim)).squeeze(self.dim)
        return (p.clamp_min(1e-12)).log()

    def entropy(self, entropy_alpha: float | None = None):
        p = self._probs.clamp_min(1e-12)
        if self.entropy_kind == "shannon":
            return -(p * p.log()).sum(dim=self.dim)
        # tsallis with provided alpha or gate alpha by default
        a = float(self.alpha if entropy_alpha is None else entropy_alpha)
        if abs(a - 1.0) < 1e-9:
            return -(p * p.log()).sum(dim=self.dim)
        return (1.0 - torch.pow(p, a).sum(dim=self.dim)) / (a - 1.0)
# ------------------------ models ------------------------

class PPOAgent(nn.Module):
    """Vector-observation PPO agent with entmax head."""

    def __init__(self, num_actions, observation_shape, device, entmax_alpha=1.5):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_shape).prod(), 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(observation_shape).prod(), 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, 512)),
            nn.Tanh(),
            layer_init(nn.Linear(512, num_actions), std=0.01),
        )
        self.device = device
        self.num_actions = num_actions
        self.entmax_alpha = float(entmax_alpha)
        self.register_buffer("mask_value", torch.tensor(INVALID_ACTION_PENALTY))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(
        self, x, legal_actions_mask=None, action=None, clip_probability_eps=None, alpha_reg=1.0
    ):
        if legal_actions_mask is None:
            legal_actions_mask = torch.ones((len(x), self.num_actions)).bool()

        logits = self.actor(x)
        #logits = apply_action_mask(logits, legal_actions_mask, self.mask_value)

        #dist = EntmaxCategorical(logits=logits, alpha=self.entmax_alpha * alpha_reg, dim=-1)
        dist = GatedSoftmaxCategorical(
            logits=logits,
            alpha=self.entmax_alpha * alpha_reg,  # raise from 1.0 -> target in last 10%
            legal_mask=legal_actions_mask,
            entropy_kind="shannon",               # or "shannon"
            softmax_temp=1.0,                     # no temp needed; keep 1.0
            gate_eps=0.0,                         # exact entmax support
            mask_value=-1e9,
        )

        # optional probability clipping (keeps distribution valid)
        # if clip_probability_eps is not None:
        #     p = dist.probs
        #     keep = (p >= clip_probability_eps).detach()
        #     p_masked = p * keep
        #     has_any = (p_masked.sum(dim=-1, keepdim=True) > 0)
        #     p = torch.where(
        #         has_any, p_masked / (p_masked.sum(dim=-1, keepdim=True) + 1e-12), p
        #     )
        #     dist = EntmaxCategorical(probs=p, alpha=self.entmax_alpha * alpha_reg, dim=-1)

        if action is None:
            action = dist.sample()

        return (
            action,
            dist.log_prob(action),
            dist.entropy(),
            self.critic(x),
            dist.probs,
        )


class PPOAtariAgent(nn.Module):
    """CNN-based PPO agent (Atari-like) with entmax head."""

    def __init__(self, num_actions, observation_shape, device, entmax_alpha=1.0):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)
        self.num_actions = num_actions
        self.device = device
        self.entmax_alpha = float(entmax_alpha)
        self.register_buffer("mask_value", torch.tensor(INVALID_ACTION_PENALTY))

    def get_value(self, x):
        return self.critic(self.network(x / 255.0))

    def get_action_and_value(
        self, x, legal_actions_mask=None, action=None, clip_probability_eps=None, alpha_reg=1.0
    ):
        if legal_actions_mask is None:
            legal_actions_mask = torch.ones((len(x), self.num_actions)).bool()

        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        #logits = apply_action_mask(logits, legal_actions_mask, self.mask_value)

        #dist = EntmaxCategorical(logits=logits, alpha=self.entmax_alpha * alpha_reg, dim=-1)
        dist = GatedSoftmaxCategorical(
            logits=logits,
            alpha=self.entmax_alpha * alpha_reg,  # raise from 1.0 -> target in last 10%
            legal_mask=legal_actions_mask,
            entropy_kind="shannon",               # or "shannon"
            softmax_temp=1.0,                     # no temp needed; keep 1.0
            gate_eps=0.0,                         # exact entmax support
            mask_value=-1e9,
        )
        # if clip_probability_eps is not None:
        #     p = dist.probs
        #     keep = (p >= clip_probability_eps).detach()
        #     p_masked = p * keep
        #     has_any = (p_masked.sum(dim=-1, keepdim=True) > 0)
        #     p = torch.where(
        #         has_any, p_masked / (p_masked.sum(dim=-1, keepdim=True) + 1e-12), p
        #     )
        #     dist = EntmaxCategorical(probs=p, alpha=self.entmax_alpha * alpha_reg, dim=-1)

        if action is None:
            action = dist.sample()

        return (
            action,
            dist.log_prob(action),
            dist.entropy(),
            self.critic(hidden),
            dist.probs,
        )


# ------------------------ env helpers ------------------------

def legal_actions_to_mask(legal_actions_list, num_actions):
    """Converts a list of legal actions to a boolean mask."""
    legal_actions_mask = torch.zeros(
        (len(legal_actions_list), num_actions), dtype=torch.bool
    )
    for i, legal_actions in enumerate(legal_actions_list):
        legal_actions_mask[i, legal_actions] = 1
    return legal_actions_mask


# ------------------------ PPO trainer ------------------------

class PPO(nn.Module):
    """PPO implementation (single-agent) with entmax policy."""

    def __init__(
        self,
        input_shape,
        num_actions,
        num_players,
        num_envs=1,
        steps_per_batch=128,
        num_minibatches=4,
        update_epochs=4,
        learning_rate=2.5e-4,
        gae=True,
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=True,
        clip_coef=0.2,
        clip_vloss=True,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        target_kl=None,
        device="cpu",
        agent_fn=PPOAtariAgent,
        log_file=None,
        entmax_alpha=1.0,         # <---- NEW: α for entmax (2.0==sparsemax)
        **kwargs,
    ):
        super().__init__()

        self.input_shape = (np.array(input_shape).prod(),)
        self.num_actions = num_actions
        self.num_players = num_players
        self.device = device
        self.log_file = log_file
        self.entmax_alpha = float(entmax_alpha)

        # Training settings
        self.num_envs = num_envs
        self.steps_per_batch = steps_per_batch
        self.batch_size = self.num_envs * self.steps_per_batch
        self.num_minibatches = num_minibatches
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.update_epochs = update_epochs
        self.learning_rate = learning_rate
        self.anneal_lr = kwargs.get("anneal_lr", False)

        # Loss settings
        self.gae = gae
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        # Networks
        self.network = agent_fn(
            self.num_actions, self.input_shape, device, entmax_alpha=self.entmax_alpha
        ).to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, eps=1e-5)

        # Buffers
        self.legal_actions_mask = torch.zeros(
            (self.steps_per_batch, self.num_envs, self.num_actions), dtype=torch.bool
        ).to(device)
        self.obs = torch.zeros(
            (self.steps_per_batch, self.num_envs, *self.input_shape)
        ).to(device)
        self.actions = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.logprobs = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.rewards = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.dones = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.values = torch.zeros((self.steps_per_batch, self.num_envs)).to(device)
        self.current_players = torch.zeros((self.steps_per_batch, self.num_envs)).to(
            device
        )

        # Counters
        self.cur_batch_idx = 0
        self.total_steps_done = 0
        self.updates_done = 0
        self.start_time = time.time()
        self.prob_clip_eps = None
        self.entropy_reg = 1.0
        self.alpha_reg = 1.0001

    def get_value(self, x):
        return self.network.get_value(x)

    def get_action_and_value(self, x, legal_actions_mask=None, action=None):
        return self.network.get_action_and_value(
            x, legal_actions_mask, action, self.prob_clip_eps, self.alpha_reg
        )

    # -------- rollout & training --------

    def step(self, time_step, is_evaluation=False):
        if is_evaluation:
            with torch.no_grad():
                legal_actions_mask = legal_actions_to_mask(
                    [ts.observations["legal_actions"][ts.current_player()] for ts in time_step],
                    self.num_actions,
                ).to(self.device)
                obs = torch.Tensor(
                    np.array([
                        np.reshape(ts.observations["info_state"][ts.current_player()], self.input_shape)
                        for ts in time_step
                    ])
                ).to(self.device)
                action, _, _, value, probs = self.get_action_and_value(
                    obs, legal_actions_mask=legal_actions_mask
                )
                return [StepOutput(action=a.item(), probs=p) for (a, p) in zip(action, probs)]
        else:
            with torch.no_grad():
                obs = torch.Tensor(
                    np.array([
                        np.reshape(ts.observations["info_state"][ts.current_player()], self.input_shape)
                        for ts in time_step
                    ])
                ).to(self.device)
                legal_actions_mask = legal_actions_to_mask(
                    [ts.observations["legal_actions"][ts.current_player()] for ts in time_step],
                    self.num_actions,
                ).to(self.device)
                current_players = torch.Tensor([ts.current_player() for ts in time_step]).to(self.device)

                action, logprob, _, value, probs = self.get_action_and_value(
                    obs, legal_actions_mask=legal_actions_mask
                )

                # store
                self.legal_actions_mask[self.cur_batch_idx] = legal_actions_mask
                self.obs[self.cur_batch_idx] = obs
                self.actions[self.cur_batch_idx] = action
                self.logprobs[self.cur_batch_idx] = logprob
                self.values[self.cur_batch_idx] = value.flatten()
                self.current_players[self.cur_batch_idx] = current_players

                return [StepOutput(action=a.item(), probs=p) for (a, p) in zip(action, probs)]

    def post_step(self, reward, done):
        self.rewards[self.cur_batch_idx] = torch.tensor(reward).to(self.device).view(-1)
        self.dones[self.cur_batch_idx] = torch.tensor(done).to(self.device).view(-1)
        self.total_steps_done += self.num_envs
        self.cur_batch_idx += 1

    def learn(self, time_step):
        next_obs = torch.Tensor(
            np.array([
                np.reshape(ts.observations["info_state"][ts.current_player()], self.input_shape)
                for ts in time_step
            ])
        ).to(self.device)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = self.get_value(next_obs).reshape(1, -1)
            if self.gae:
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.steps_per_batch)):
                    nextvalues = next_value if t == self.steps_per_batch - 1 else self.values[t + 1]
                    nextnonterminal = 1.0 - self.dones[t]
                    delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.values
            else:
                returns = torch.zeros_like(self.rewards).to(self.device)
                for t in reversed(range(self.steps_per_batch)):
                    next_return = next_value if t == self.steps_per_batch - 1 else returns[t + 1]
                    nextnonterminal = 1.0 - self.dones[t]
                    returns[t] = self.rewards[t] + self.gamma * nextnonterminal * next_return
                advantages = returns - self.values

        # flatten batch
        b_legal_actions_mask = self.legal_actions_mask.reshape((-1, self.num_actions))
        b_obs = self.obs.reshape((-1,) + self.input_shape)
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)
        b_playersigns = -2.0 * self.current_players.reshape(-1) + 1.0
        b_advantages *= b_playersigns

        # optimize
        b_inds = np.arange(self.batch_size)
        clipfracs = []
        for _ in range(self.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue, _ = self.get_action_and_value(
                    b_obs[mb_inds],
                    legal_actions_mask=b_legal_actions_mask[mb_inds],
                    action=b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if self.normalize_advantages:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                newvalue = newvalue.view(-1)
                if self.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -self.clip_coef, self.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - self.entropy_coef * self.entropy_reg * entropy_loss + v_loss * self.value_coef

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

            if self.target_kl is not None and approx_kl > self.target_kl:
                break

        # diagnostics
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        # (csv logging omitted to keep file light)

        self.updates_done += 1
        self.cur_batch_idx = 0

    # -------- misc --------

    def save(self, path):
        torch.save(self.network.actor.state_dict(), path)

    def load(self, path):
        self.network.actor.load_state_dict(torch.load(path))

    def anneal_learning_rate(self, update, num_total_updates):
        frac = max(0, 1.0 - (update / num_total_updates))
        if frac < 0:
            raise ValueError("Annealing learning rate to < 0")
        self.optimizer.param_groups[0]["lr"] = frac * self.learning_rate

    def anneal_alpha(self, update, num_total_updates):
        self.alpha_reg = 1.1101 if update / num_total_updates >= 0.9 else 1.0001
        # start_update = 9 * (num_total_updates // 10)
        # delta = max(0, update - start_update)
        # tot = max(1, num_total_updates - start_update)
        # # increase alpha from 1.0001 to 1.1 over last 10

        # self.alpha_reg = 1.0001 + 0.1 * (delta / tot)

    def anneal_prob_clip(self, update, num_total_updates):
        start_update = 9 * (num_total_updates // 10)
        delta = max(0, update - start_update)
        tot = max(1, num_total_updates - start_update)

#        self.prob_clip_eps = 1e-2 if update / num_total_updates >= 0.6 else None
        #self.prob_clip_eps = 0.001  if update / num_total_updates >= 0.9 else None
        self.prob_clip_eps = None  # <--- disable for now

    def anneal_entropy_reg(self, update, num_total_updates):
        entropy_reg = 1.0
        # if update / num_total_updates >= 0.4:
        #     delta = ((update / num_total_updates) - 0.4) / 0.6
        #     entropy_reg = 1.0 - delta
        self.entropy_reg = entropy_reg
