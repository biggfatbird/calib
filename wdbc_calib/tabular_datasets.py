"""Public binary tables for the cross-dataset protocol.

Each table is numeric, has no missing values, and codes one declared event as class 1.
OpenML class codes that are only integers in the file are tied to the UCI class counts
named below. Those counts are the reason for the coding. They are not a guess about
which class is clinically important.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from wdbc_calib.data import N_FEATURES, N_MALIGNANT, N_ROWS, load_wdbc


@dataclass(frozen=True)
class BinaryTable:
    key: str
    title: str
    source: str
    positive_class: str
    X: np.ndarray
    y: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_positive(self) -> int:
        return int(self.y.sum())

    def sha256(self) -> str:
        payload = np.ascontiguousarray(self.X).tobytes() + np.ascontiguousarray(self.y).tobytes()
        return hashlib.sha256(payload).hexdigest()

    def card_row(self) -> dict:
        return {
            "dataset": self.key,
            "title": self.title,
            "source": self.source,
            "positive_class": self.positive_class,
            "n_rows": self.n_rows,
            "n_features": self.n_features,
            "n_positive": self.n_positive,
            "positive_rate": self.n_positive / self.n_rows,
            "sha256": self.sha256(),
        }


def encode_binary(labels: pd.Series, positive: str, allowed: set[str]) -> np.ndarray:
    text = labels.astype(str)
    found = set(text.unique())
    if found != allowed:
        raise RuntimeError(f"Expected labels {sorted(allowed)}, found {sorted(found)}.")
    if positive not in allowed:
        raise RuntimeError(f"Positive label {positive!r} is not in {sorted(allowed)}.")
    return (text.to_numpy() == positive).astype(np.int8)


def _openml_numeric(name: str, version: int, data_id: int) -> tuple[pd.DataFrame, pd.Series, str]:
    bunch = fetch_openml(name=name, version=version, as_frame=True, parser="auto")
    found_id = int(bunch.details["id"])
    found_version = str(bunch.details["version"])
    if found_id != data_id or found_version != str(version):
        raise RuntimeError(
            f"{name}: expected OpenML id {data_id} version {version}, found {found_id} version {found_version}."
        )
    frame = bunch.data
    if frame.isna().any().any():
        raise RuntimeError(f"{name} contains missing values.")
    X = frame.apply(pd.to_numeric, errors="raise")
    source = f"OpenML data_id {data_id}, name {name}, version {version}"
    return X, bunch.target, source


def load_wdbc_table() -> BinaryTable:
    X, y, _names = load_wdbc()
    if X.shape != (N_ROWS, N_FEATURES) or int(y.sum()) != N_MALIGNANT:
        raise RuntimeError("WDBC table does not match the pinned shape.")
    return BinaryTable(
        key="wdbc",
        title="Wisconsin Diagnostic Breast Cancer",
        source="sklearn.datasets.load_breast_cancer",
        positive_class="malignant (sklearn target == 0)",
        X=X,
        y=y,
    )


def load_pima_diabetes() -> BinaryTable:
    X, target, source = _openml_numeric("diabetes", 1, 37)
    y = encode_binary(target, "tested_positive", {"tested_negative", "tested_positive"})
    if (X.shape[0], int(y.sum())) != (768, 268):
        raise RuntimeError(f"Unexpected Pima shape or positive count: {X.shape}, {int(y.sum())}.")
    return BinaryTable("pima_diabetes", "Pima Indians diabetes", source, "tested_positive", X.to_numpy(float), y)


def load_ionosphere() -> BinaryTable:
    X, target, source = _openml_numeric("ionosphere", 1, 59)
    y = encode_binary(target, "b", {"b", "g"})
    if (X.shape[0], int(y.sum())) != (351, 126):
        raise RuntimeError(f"Unexpected ionosphere shape or positive count: {X.shape}, {int(y.sum())}.")
    return BinaryTable(
        "ionosphere",
        "Johns Hopkins ionosphere",
        source,
        "b (bad radar return)",
        X.to_numpy(float),
        y,
    )


def load_blood() -> BinaryTable:
    X, target, source = _openml_numeric("blood-transfusion-service-center", 1, 1464)
    y = encode_binary(target, "2", {"1", "2"})
    # UCI reports 178 donations in March 2007. OpenML label "2" has that count.
    if (X.shape[0], int(y.sum())) != (748, 178):
        raise RuntimeError(f"Unexpected blood-transfusion shape or positive count: {X.shape}, {int(y.sum())}.")
    return BinaryTable(
        "blood_transfusion",
        "Blood transfusion service center",
        source,
        "2 (donated in March 2007; 178 rows, matching the UCI positive count)",
        X.to_numpy(float),
        y,
    )


def load_banknote() -> BinaryTable:
    X, target, source = _openml_numeric("banknote-authentication", 1, 1462)
    y = encode_binary(target, "2", {"1", "2"})
    # UCI reports 762 genuine and 610 forged. OpenML label "2" has the forged count.
    if (X.shape[0], int(y.sum())) != (1372, 610):
        raise RuntimeError(f"Unexpected banknote shape or positive count: {X.shape}, {int(y.sum())}.")
    return BinaryTable(
        "banknote",
        "Banknote authentication",
        source,
        "2 (610 rows, matching the UCI forged count)",
        X.to_numpy(float),
        y,
    )


def load_sonar() -> BinaryTable:
    X, target, source = _openml_numeric("sonar", 1, 40)
    y = encode_binary(target, "Mine", {"Mine", "Rock"})
    if (X.shape[0], int(y.sum())) != (208, 111):
        raise RuntimeError(f"Unexpected sonar shape or positive count: {X.shape}, {int(y.sum())}.")
    return BinaryTable("sonar", "Sonar mines versus rocks", source, "Mine", X.to_numpy(float), y)


def load_qsar() -> BinaryTable:
    X, target, source = _openml_numeric("qsar-biodeg", 1, 1494)
    y = encode_binary(target, "2", {"1", "2"})
    # Mansouri et al. report 356 ready-biodegradable chemicals. OpenML label "2" has that count.
    if (X.shape[0], int(y.sum())) != (1055, 356):
        raise RuntimeError(f"Unexpected QSAR shape or positive count: {X.shape}, {int(y.sum())}.")
    return BinaryTable(
        "qsar_biodeg",
        "QSAR biodegradation",
        source,
        "2 (ready biodegradable; 356 rows, matching Mansouri et al. 2013)",
        X.to_numpy(float),
        y,
    )


def load_spambase() -> BinaryTable:
    X, target, source = _openml_numeric("spambase", 1, 44)
    y = encode_binary(target, "1", {"0", "1"})
    if (X.shape[0], int(y.sum())) != (4601, 1813):
        raise RuntimeError(f"Unexpected spambase shape or positive count: {X.shape}, {int(y.sum())}.")
    return BinaryTable("spambase", "Spambase", source, "1 (spam)", X.to_numpy(float), y)


LOADERS = (
    load_wdbc_table,
    load_pima_diabetes,
    load_ionosphere,
    load_blood,
    load_banknote,
    load_sonar,
    load_qsar,
    load_spambase,
)


def load_study_tables() -> list[BinaryTable]:
    return [loader() for loader in LOADERS]
