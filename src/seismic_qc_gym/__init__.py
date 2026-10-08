"""Seismic QC Gym: an environment for training and evaluating tool-using seismic QC agents."""

from .defects import DEFECT_TYPES, RECOMMENDED_FIX
from .env import TOOL_SPECS, QCEnv
from .grading import RewardWeights, grade
from .tasks import TaskSpec, make_tasks

__version__ = "0.1.0"

__all__ = [
    "DEFECT_TYPES",
    "RECOMMENDED_FIX",
    "TOOL_SPECS",
    "QCEnv",
    "RewardWeights",
    "grade",
    "TaskSpec",
    "make_tasks",
    "__version__",
]
