# Evaluate RL policies and analyze the action distribution.

import argparse
import csv
from pathlib import Path

import numpy as np

from project291d.rl.q_learning import QLearningAgent
from project291d.rl.simulation import BridgeSimulation


def evaluate(model_path="outputs/q_table.npz", episodes=500,
             output_path="outputs/evaluation.csv", seed=2026):
    agent = QLearningAgent.load(model_path)
    results = []
    methods = ("Q-learning", "Never act", "Periodic inspect", "Fixed threshold")
    action_labels = ("Do nothing", "Inspect", "Restrict load", "Repair")

    for method in methods:
        episode_costs, episode_risks, episode_repairs = [], [], []
        action_counts = np.zeros(4, dtype=int)

        for ep in range(episodes):
            # Use the same seed for comparable random conditions.
            env = BridgeSimulation(seed=seed + ep)
            observation = env.reset()
            total_cost = total_risk = repairs = 0.0

            for step in range(env.max_steps):
                if method == "Q-learning":
                    action = agent.choose_action(observation, greedy=True)
                elif method == "Never act":
                    action = 0
                elif method == "Periodic inspect":
                    action = 1 if (step + 1) % 10 == 0 else 0
                else:
                    action = 3 if observation[0] >= 0.65 else (2 if observation[0] >= 0.35 else 0)

                action_counts[action] += 1

                observation, _, done, info = env.step(action)
                total_cost += info["total_cost"]
                total_risk += info["risk_cost"]
                repairs += int(action == 3)

                if done:
                    break

            episode_costs.append(total_cost)
            episode_risks.append(total_risk)
            episode_repairs.append(repairs)

        result = {
            "policy": method,
            "mean_total_cost": round(float(np.mean(episode_costs)), 3),
            "std_total_cost": round(float(np.std(episode_costs)), 3),
            "mean_risk_cost": round(float(np.mean(episode_risks)), 3),
            "mean_repairs": round(float(np.mean(episode_repairs)), 3)
        }
        results.append(result)

        print(f"{method:18s} | Cost: {result['mean_total_cost']:8.2f} | "
              f"Risk: {result['mean_risk_cost']:8.2f} | Repairs: {result['mean_repairs']:.2f}")

        if method == "Q-learning":
            print("\nQ-learning Action Distribution:")
            total_actions = action_counts.sum()
            for label, count in zip(action_labels, action_counts):
                percentage = 100 * count / total_actions
                print(f"  {label:15s}: {count:6d} ({percentage:6.2f}%)")
            print()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved evaluation to {output_path.resolve()}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/q_table.npz")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--output", default="outputs/evaluation.csv")
    args = parser.parse_args()
    evaluate(args.model, args.episodes, args.output)
