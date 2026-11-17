# Calculate normalized bridge maintenance costs and structural risk.

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    inspect_cost: float = 2.0
    restrict_activation_cost: float = 5.0
    repair_cost: float = 20.0
    restrict_running_cost: float = 1.0
    risk_weight: float = 10.0
    restricted_risk_factor: float = 0.4


def calculate_reward(action: int, severity: float, was_restricted: bool,
                     is_restricted: bool, config: RewardConfig = RewardConfig()):
    """Return (reward, cost breakdown) after applying an action and transition."""
    if action not in (0, 1, 2, 3):
        raise ValueError("Action must be 0, 1, 2, or 3")
    if not 0.0 <= severity <= 1.0:
        raise ValueError("Severity must be between 0 and 1")
    action_cost = (config.inspect_cost if action == 1 else
                   config.repair_cost if action == 3 else
                   config.restrict_activation_cost if action == 2 and not was_restricted else 0.0)
    operation_cost = config.restrict_running_cost if is_restricted else 0.0
    risk_cost = config.risk_weight * severity * (config.restricted_risk_factor if is_restricted else 1.0)
    breakdown = {"action_cost": action_cost, "operation_cost": operation_cost,
                 "risk_cost": risk_cost, "total_cost": action_cost + operation_cost + risk_cost}
    return -breakdown["total_cost"], breakdown
