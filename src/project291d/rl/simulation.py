# Simulate Z24 damage scenarios and maintenance decisions for RL development.

import numpy as np

from project291d.data.config import SEVERITY
from project291d.rl.reward import RewardConfig, calculate_reward


class BridgeSimulation:
    """
    Temporary simulator; scenario severities and transitions are assumptions, not Z24 measurements.

    Observation: (predicted_severity, load_restricted, just_inspected).
    Actions: 0=do nothing, 1=inspect, 2=restrict load, 3=repair.
    Contract: reset()->observation, step(action)->(observation, reward, done, info).
    """

    def __init__(
        self,
        max_steps=50,
        seed=42,
        observation_noise=0.08,
        normal_progression=0.10,
        restricted_progression=0.04,
        reward_config=None,
    ):

        self.PIER_PATH = (3, 4, 5, 6, 7)
        self.CUMULATIVE_PATH = (9, 10, 11, 12, 13, 14, 15, 16, 17)

        self.max_steps = max_steps  # number of steps in the episode
        self.observation_noise = observation_noise  # TODO: delete once real predictions are implemented
        self.normal_progression = normal_progression  # rate at which damages start and progress
        self.restricted_progression = restricted_progression  # rate when restricting the load on the bridge
        self.reward_config = reward_config or RewardConfig()  # costs of actions and risks
        self.rng = np.random.default_rng(seed)
        self.reset()

    def _observe(self, inspected=False):
        """Return the predicted severity, load restriction status, and inspection status."""
        severity = SEVERITY[self.scenario]
        prediction = (
            severity if inspected else  # return true severity if action was "inspect"
            # TODO: replace with actual predicted severity score from ML classifier
            # I think the environment should load the data from the current scenario and call the classifier here
            float(np.clip(severity + self.rng.normal(0.0, self.observation_noise), 0.0, 1.0))  # simulation of a predicted severity
        )
        return (prediction, int(self.restricted), int(inspected))

    def reset(self):
        """Reset the environment to the initial state and return the first observation."""
        self.scenario = 1  # start from a healthy state
        self.path = None  # no damage progression path by default
        self.restricted = False
        self.steps = 0
        return self._observe()

    def step(self, action: int):
        """Advance the environment by one timestep.

        Parameters
        ----------
        action : 0 = do nothing, 1 = inspect, 2 = restrict load, 3 = repair.

        Returns
        -------
        observation : tuple[float, int, int]
            A tuple containing the predicted or true severity (if ``inspect``),
            whether the load is restricted, and whether an inspection was performed.
        reward : float
            The total reward for the timestep.
        done : bool
            Whether the episode has reached its maximum number of steps.
        info : dict
            ``scenario``, ``true_severity``, ``restricted``, and cost breakdown.
        """
        if action not in (0, 1, 2, 3):
            raise ValueError("Action must be 0, 1, 2, or 3")
        was_restricted = self.restricted

        # repair
        if action == 3:
            self.scenario = 8  # get back to a healthy state
            self.path = None  # reset damage path
            self.restricted = False  # remove load retrictions

        else:
            # restrict load
            if action == 2:
                self.restricted = True

            # slow down the random damage progression if restricted
            p = (
                self.restricted_progression if self.restricted else
                self.normal_progression  # should always be > restricted_progression
            )

            # randomly trigger a damage progression step
            if self.rng.random() < p:
                if self.path is None:  # simulation is on a damage progression path
                    self.path = (  # select a damage path randomly
                        self.PIER_PATH if self.rng.random() < 0.5 else
                        self.CUMULATIVE_PATH
                    )
                    self.scenario = self.path[0]  # start from the first scenario of the path

                else:  # simulation is already on a damage progression path
                    idx = self.path.index(self.scenario)
                    # move to the next scenario of the path OR stay if it's the last
                    self.scenario = self.path[min(idx + 1, len(self.path) - 1)]

        # Outputs
        severity = SEVERITY[self.scenario]
        reward, costs = calculate_reward(
            action, severity, was_restricted, self.restricted, self.reward_config
        )
        self.steps += 1
        done = self.steps >= self.max_steps
        observation = self._observe(inspected=action == 1)
        info = {
            "scenario": self.scenario,
            "true_severity": severity,
            "restricted": self.restricted,
            **costs,
        }
        return observation, reward, done, info
