# Learn a discrete bridge maintenance policy using tabular Q-learning.

from pathlib import Path

import numpy as np


class QLearningAgent:
    """Discretize predicted severity and track restriction/inspection indicators."""

    def __init__(self, n_bins=21, learning_rate=0.15, discount=0.98,
                 epsilon=1.0, epsilon_min=0.03, epsilon_decay=0.9995, seed=123):
        self.n_bins = n_bins
        self.alpha = learning_rate
        self.gamma = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.q_table = np.zeros((n_bins, 2, 2, 4), dtype=np.float64)

    def encode(self, observation):
        predicted_severity, restricted, inspected = observation
        severity_bin = min(self.n_bins - 1, max(0, int(float(predicted_severity) * self.n_bins)))
        return (severity_bin, int(restricted), int(inspected))

    def choose_action(self, observation, greedy=False):
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(4))
        values = self.q_table[self.encode(observation)]
        return int(np.argmax(values))

    def update(self, observation, action, reward, next_observation, done):
        idx = self.encode(observation) + (action,)
        best_next = 0.0 if done else float(np.max(self.q_table[self.encode(next_observation)]))
        target = reward + self.gamma * best_next
        self.q_table[idx] += self.alpha * (target - self.q_table[idx])

    def decay_exploration(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, q_table=self.q_table, n_bins=self.n_bins,
                            gamma=self.gamma, alpha=self.alpha)

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            agent = cls(n_bins=int(data["n_bins"]), learning_rate=float(data["alpha"]),
                        discount=float(data["gamma"]), epsilon=0)
            agent.q_table = data["q_table"].copy()
        return agent
