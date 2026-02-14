from dataclasses import dataclass

import numpy as np

from modeling.base import VFPModel
from models.common import FlowType


@dataclass(frozen=True, slots=True)
class TrainingData:
    features: np.ndarray
    targets: np.ndarray

@dataclass(frozen=True, slots=True)
class ModelPredictions:
    prediction_features: np.ndarray
    predicted_target: np.ndarray

type WellName = str
type WellsTrainingData = dict[WellName, FlowType[TrainingData]]
type WellsFittedModels = dict[WellName, FlowType[VFPModel]]
type WellsPredictions = dict[WellName, FlowType[ModelPredictions]]
type WellsVFPContent = dict[WellName, FlowType[np.ndarray]]
