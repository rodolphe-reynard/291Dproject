# Simulate Z24 damage scenarios and maintenance decisions for RL development.

import numpy as np

from project291d.rl.reward import RewardConfig, calculate_reward


class MockBridgeEnvironment:
    """Temporary simulator; scenario severities and transitions are assumptions, not Z24 measurements.

    Observation: (predicted_severity, load_restricted, just_inspected).
    Actions: 0=do nothing, 1=inspect, 2=restrict load, 3=repair.
    Contract: reset()->observation, step(action)->(observation, reward, done, info).
    """

    # Zero damage in scenarios 1, 2, 8; other scores are illustrative assignments.
    SEVERITY = {1: 0.0, 2: 0.0, 3: 0.15, 4: 0.28, 5: 0.43,
                6: 0.60, 7: 0.75, 8: 0.0, 9: 0.18, 10: 0.30,
                11: 0.42, 12: 0.55, 13: 0.62, 14: 0.70, 15: 0.78,
                16: 0.88, 17: 1.0}
    PIER_PATH = (3, 4, 5, 6, 7)
    CUMULATIVE_PATH = (9, 10, 11, 12, 13, 14, 15, 16, 17)

    def __init__(self, max_steps=50, seed=42, observation_noise=0.08,
                 normal_progression=0.10, restricted_progression=0.04,
                 reward_config=None):
        self.max_steps = max_steps
        self.observation_noise = observation_noise
        self.normal_progression = normal_progression
        self.restricted_progression = restricted_progression
        self.reward_config = reward_config or RewardConfig()
        self.rng = np.random.default_rng(seed)
        self.reset()

    def _observe(self, inspected=False):
        severity = self.SEVERITY[self.scenario]
        prediction = severity if inspected else float(np.clip(
            severity + self.rng.normal(0.0, self.observation_noise), 0.0, 1.0))
        return (prediction, int(self.restricted), int(inspected))

    def reset(self):
        self.scenario = 8
        self.path = None
        self.restricted = False
        self.steps = 0
        return self._observe()

    def step(self, action):
        if action not in (0, 1, 2, 3):
            raise ValueError("Action must be 0, 1, 2, or 3")
        was_restricted = self.restricted

        if action == 3:
            self.scenario = 8
            self.path = None
            self.restricted = False
        else:
            if action == 2:
                self.restricted = True
            p = self.restricted_progression if self.restricted else self.normal_progression
            if self.rng.random() < p:
                if self.path is None:
                    self.path = self.PIER_PATH if self.rng.random() < 0.5 else self.CUMULATIVE_PATH
                    self.scenario = self.path[0]
                else:
                    idx = self.path.index(self.scenario)
                    self.scenario = self.path[min(idx + 1, len(self.path) - 1)]

        severity = self.SEVERITY[self.scenario]
        reward, costs = calculate_reward(action, severity, was_restricted,
                                         self.restricted, self.reward_config)
        self.steps += 1
        done = self.steps >= self.max_steps
        observation = self._observe(inspected=action == 1)
        info = {"scenario": self.scenario, "true_severity": severity,
                "restricted": self.restricted, **costs}
        return observation, reward, done, info
