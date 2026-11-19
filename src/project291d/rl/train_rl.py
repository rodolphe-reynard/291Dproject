# Train and save a Q-learning policy using a temporary Z24 bridge simulator.

import argparse
from pathlib import Path

import numpy as np

from project291d.rl.mock_environment import MockBridgeEnvironment
from project291d.rl.q_learning import QLearningAgent


def train(episodes=10000, model_path="outputs/q_table.npz", seed=42):
    env = MockBridgeEnvironment(seed=seed)
    agent = QLearningAgent(seed=seed + 1)
    returns = []

    for episode in range(episodes):
        observation = env.reset()
        total_reward = 0.0
        while True:
            action = agent.choose_action(observation)
            next_observation, reward, done, _ = env.step(action)
            agent.update(observation, action, reward, next_observation, done)
            total_reward += reward
            observation = next_observation
            if done:
                break
        agent.decay_exploration()
        returns.append(total_reward)
        if (episode + 1) % max(1, episodes // 10) == 0:
            print(f"Episode {episode+1}/{episodes} | Mean return (last 100): "
                  f"{np.mean(returns[-100:]):.2f} | Epsilon: {agent.epsilon:.3f}")

    agent.save(model_path)
    print(f"Saved Q-table to {Path(model_path).resolve()}")
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--model", default="outputs/q_table.npz")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.episodes, args.model, args.seed)
