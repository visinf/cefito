"""The task classifier used to constrain the search (Sec. 3.3).

A small MLP that reads the initial and goal observations and predicts the
high-level task c^, which selects the action subspace A(c^) the planner searches
over. It is trained separately from the predictor with a cross-entropy loss.
"""

import torch
import torch.nn.functional as F
from torch import nn


class TaskClassifier(nn.Module):
    """Four-layer MLP over the (initial, goal) observation pair.

    The two observations are embedded independently and pooled by averaging
    before the classification layer, so the classifier is symmetric in how it
    treats them.

    Args:
        observation_dim: dimension of a single visual observation.
        num_tasks: size of the task vocabulary.
    """

    def __init__(self, observation_dim: int, num_tasks: int):
        super().__init__()
        bottleneck = observation_dim // 3
        expanded = observation_dim * 4

        self.fc1 = nn.Linear(observation_dim, bottleneck)
        self.fc2 = nn.Linear(bottleneck, expanded)
        self.fc3 = nn.Linear(expanded, bottleneck)
        self.fc4 = nn.Linear(bottleneck, num_tasks)

        for layer in (self.fc1, self.fc2, self.fc3, self.fc4):
            nn.init.kaiming_normal_(layer.weight, mode="fan_in")
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Args: ``[B, 2, observation_dim]`` (initial, goal). Returns ``[B, num_tasks]``."""
        x = self.fc2(self.fc1(observations))
        x = F.relu(self.fc3(F.relu(x)))
        return self.fc4(x.mean(dim=1))
