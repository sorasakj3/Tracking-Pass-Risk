from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn

from .features import FEATURES


class PassRiskMLP(nn.Module):
    def __init__(self, inputs: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(inputs, 32),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


@dataclass(frozen=True)
class Result:
    samples: int
    train_samples: int
    validation_samples: int
    test_samples: int
    test_pressure_rate: float
    neural_average_precision: float
    neural_roc_auc: float
    neural_brier: float
    logistic_average_precision: float
    best_epoch: int
    recommended_model: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _split(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    first = int(len(table) * 0.7)
    second = int(len(table) * 0.85)
    return table.iloc[:first], table.iloc[first:second], table.iloc[second:]


def train(
    table: pd.DataFrame, epochs: int = 160, seed: int = 17
) -> tuple[PassRiskMLP, StandardScaler, Result, np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_frame, validation_frame, test_frame = _split(table)
    scaler = StandardScaler().fit(train_frame[FEATURES])

    def tensors(frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(scaler.transform(frame[FEATURES]), dtype=torch.float32)
        y = torch.tensor(frame["receiver_pressured_at_arrival"].to_numpy(), dtype=torch.float32)
        return x, y

    x_train, y_train = tensors(train_frame)
    x_validation, y_validation = tensors(validation_frame)
    x_test, y_test = tensors(test_frame)
    model = PassRiskMLP(len(FEATURES))
    positive = max(float(y_train.sum()), 1.0)
    pos_weight = torch.tensor((len(y_train) - positive) / positive)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=0.015)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    patience = 22
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimiser.zero_grad()
        loss = loss_fn(model(x_train), y_train)
        loss.backward()
        optimiser.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(x_validation), y_validation))
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        neural_probability = torch.sigmoid(model(x_test)).numpy()

    baseline = LogisticRegression(class_weight="balanced", max_iter=2_000)
    baseline.fit(scaler.transform(train_frame[FEATURES]), train_frame["receiver_pressured_at_arrival"])
    baseline_probability = baseline.predict_proba(scaler.transform(test_frame[FEATURES]))[:, 1]
    y = y_test.numpy()
    neural_ap = float(average_precision_score(y, neural_probability))
    logistic_ap = float(average_precision_score(y, baseline_probability))
    result = Result(
        samples=len(table),
        train_samples=len(train_frame),
        validation_samples=len(validation_frame),
        test_samples=len(test_frame),
        test_pressure_rate=float(y.mean()),
        neural_average_precision=neural_ap,
        neural_roc_auc=float(roc_auc_score(y, neural_probability)),
        neural_brier=float(brier_score_loss(y, neural_probability)),
        logistic_average_precision=logistic_ap,
        best_epoch=best_epoch,
        recommended_model="neural_mlp" if neural_ap > logistic_ap + 0.01 else "logistic_baseline",
    )
    return model, scaler, result, neural_probability
