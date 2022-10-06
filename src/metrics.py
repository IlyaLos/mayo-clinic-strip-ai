from collections import defaultdict
from typing import List, Tuple

import numpy as np


def _group_by_patients(
        y_true: List[float],
        y_pred: List[float],
        image_ids: List[str],
) -> Tuple[List[float], np.ndarray, List[str]]:
    patients = [image_id.split('_')[0] for image_id in image_ids]
    patient_to_y_true, patient_to_y_pred = defaultdict(list), defaultdict(list)
    for y, y_hat, patient in zip(y_true, y_pred, patients):
        patient_to_y_true[patient].append(y)
        patient_to_y_pred[patient].append(y_hat)
    patient_to_y_true = {
        patient: np.mean(y_true)
        for patient, y_true in patient_to_y_true.items()
    }
    patient_to_y_pred = {
        patient: np.mean(y_pred).tolist()
        for patient, y_pred in patient_to_y_pred.items()
    }
    y_true, y_pred, patients = [], [], []
    for patient, y in patient_to_y_true.items():
        y_true.append(y)
        y_pred.append(patient_to_y_pred[patient])
        patients.append(patient)

    return y_true, np.array([[1 - p, p] for p in y_pred]), patients


def get_target_metric(y_true: List[float], y_pred: List[float], image_ids: List[str]) -> float:
    return _weighted_mc_log_loss(*_group_by_patients(y_true, y_pred, image_ids)[:2])


def _weighted_mc_log_loss(y_true: List[float], y_pred: np.ndarray, epsilon: float = 1e-15) -> float:
    class_cnt = [sum(int(val == cl) for val in y_true) for cl in range(2)]
    w = [0.5 for _ in range(2)]
    return -sum(
        w[cl] * sum(
            (y == cl) / class_cnt[cl] * np.log(max(min(y_hat, 1 - epsilon), epsilon))
            for y, y_hat in zip(y_true, y_pred[:, cl])
        )
        for cl in range(2)
    ) / sum(w[cl] for cl in range(2))
