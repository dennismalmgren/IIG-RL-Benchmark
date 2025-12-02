import collections
import torch
import copy
from open_spiel.python import rl_agent
from algorithms.ppo.ppo import PPO
from open_spiel.python.rl_environment import TimeStep

Transition = collections.namedtuple(
    "Transition", "info_state action reward discount legal_actions_mask"
)


class PPOWrapper(rl_agent.AbstractAgent):
    def __init__(self, player_id, info_state_size, num_actions, **kwargs):
        self.player_id = player_id
        self.num_actions = num_actions
        self._kwargs = locals()
        self._kwargs.update(kwargs)
        self._kwargs.pop("kwargs")

        self.agent = PPO(info_state_size, num_actions, 2, **kwargs)
        self.cur_step = 0
        self.post_step_called = True

    def step(self, time_step, is_evaluation=False):
        assert self.post_step_called, "post_step must be called before step"
        fixed_obs_dict = copy.copy(time_step.observations)
        fixed_obs_dict["current_player"] = self.player_id
        fixed_time_step = TimeStep(
            observations=fixed_obs_dict,
            rewards=time_step.rewards,
            discounts=time_step.discounts,
            step_type=time_step.step_type,
        )
        agent_output = self.agent.step([fixed_time_step], is_evaluation)

        agent_output = agent_output[0]
        if not is_evaluation:
            self.cur_step += 1
            self.post_step_called = False
        return agent_output

    def post_step(self, time_step, is_evaluation=False):
        assert not is_evaluation, "should not be calling post_step in evaluation"
        # self.agent.post_step([time_step.rewards[self.player_id]], [time_step.last()])
        self.agent.post_step([time_step.rewards[0]], [time_step.last()])
        self.post_step_called = True
        if self.agent.anneal_lr:
            assert False, "max steps is incorrect, and we aren't using annealing lr yet"
            update = self.agent.total_steps_done // self.agent.steps_per_batch
            num_updates = 10_000_000 // self.agent.steps_per_batch
            self.agent.anneal_learning_rate(update, num_updates)

        if self.agent.total_steps_done % self.agent.steps_per_batch == 0:
            # this allows us to learn
            fixed_obs_dict = copy.copy(time_step.observations)
            fixed_obs_dict["current_player"] = self.player_id
            fixed_time_step = TimeStep(
                observations=fixed_obs_dict,
                rewards=time_step.rewards,
                discounts=time_step.discounts,
                step_type=time_step.step_type,
            )
            self.agent.learn([fixed_time_step])

    def save(self, path):
        self.agent.save(path)

    # don't need this function
    def add_transition(self, *args, **kwargs):
        pass

    def copy_with_noise(self, sigma=0.0, copy_weights=True):
        """Copies the object and perturbates its network's weights with noise.

        Args:
        sigma: gaussian dropout variance term : Multiplicative noise following
            (1+sigma*epsilon), epsilon standard gaussian variable, multiplies each
            model weight. sigma=0 means no perturbation.
        copy_weights: Boolean determining whether to copy model weights (True) or
            just model hyperparameters.

        Returns:
        Perturbated copy of the model.
        """
        input_kwargs = copy.copy(self._kwargs)
        _ = input_kwargs.pop("self", None)
        player_id = input_kwargs.pop("player_id", None)
        info_state_size = input_kwargs.pop("info_state_size", None)
        num_actions = input_kwargs.pop("num_actions", None)
        copied_object = PPOWrapper(
            player_id, info_state_size, num_actions, **input_kwargs
        )

        actor = getattr(copied_object.agent.network, "actor")
        critic = getattr(copied_object.agent.network, "critic")
        if copy_weights:
            with torch.no_grad():
                for layer in critic:
                    if hasattr(layer, "weight"):
                        layer.weight *= 1 + sigma * torch.randn(layer.weight.shape)
                for layer in actor:
                    if hasattr(layer, "weight"):
                        layer.weight *= 1 + sigma * torch.randn(layer.weight.shape)
        return copied_object
