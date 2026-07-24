from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
FEATURES = [
    "pass_length_m",
    "delta_x_m",
    "delta_y_m",
    "nearest_defender_start_m",
    "nearest_defender_target_m",
    "lane_clearance_m",
    "nearest_support_target_m",
    "team_width_m",
    "opponent_compactness_m",
    "start_x_m",
    "start_y_m",
]


def point_segment_distance(
    point: np.ndarray, start: np.ndarray, end: np.ndarray
) -> float:
    segment = end - start
    denominator = float(segment @ segment)
    if denominator == 0:
        return float(np.linalg.norm(point - start))
    t = float(np.clip(((point - start) @ segment) / denominator, 0, 1))
    return float(np.linalg.norm(point - (start + t * segment)))


def _load_selected_frames(path: str | Path, frames: set[int]) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, skiprows=2, chunksize=20_000):
        chosen = chunk[chunk["Frame"].isin(frames)]
        if not chosen.empty:
            selected.append(chosen)
    if not selected:
        raise ValueError(f"No requested frames found in {path}")
    return pd.concat(selected).drop_duplicates("Frame").set_index("Frame")


def _positions(row: pd.Series) -> np.ndarray:
    values = row.to_numpy()
    coordinates: list[list[float]] = []
    for index in range(3, len(values) - 2, 2):
        x, y = values[index], values[index + 1]
        if pd.notna(x) and pd.notna(y):
            coordinates.append([float(x) * PITCH_LENGTH, float(y) * PITCH_WIDTH])
    return np.asarray(coordinates, dtype=float)


def _turnover_labels(events: pd.DataFrame, seconds: float = 5.0) -> pd.Series:
    labels: list[int] = []
    for index, event in events.iterrows():
        future = events.loc[index + 1 : index + 4]
        elapsed = future["Start Time [s]"] - event["End Time [s]"]
        turnover = (
            (future["Team"] == event["Team"])
            & (future["Type"] == "BALL LOST")
            & (elapsed >= 0)
            & (elapsed <= seconds)
        ).any()
        labels.append(int(turnover))
    return pd.Series(labels, index=events.index, dtype=int)


def _compactness(points: np.ndarray) -> float:
    if len(points) < 2:
        return float("nan")
    centre = points.mean(axis=0)
    return float(np.linalg.norm(points - centre, axis=1).mean())


def build_pass_features(
    events_path: str | Path, home_tracking_path: str | Path, away_tracking_path: str | Path
) -> pd.DataFrame:
    all_events = pd.read_csv(events_path)
    pass_mask = (
        (all_events["Type"] == "PASS")
        & all_events[["Start X", "Start Y", "End X", "End Y"]].notna().all(axis=1)
    )
    passes = all_events[pass_mask].copy()
    passes["turnover_within_5s"] = _turnover_labels(all_events).loc[passes.index]
    frames = set(passes["Start Frame"].astype(int)) | set(passes["End Frame"].astype(int))
    home = _load_selected_frames(home_tracking_path, frames)
    away = _load_selected_frames(away_tracking_path, frames)

    rows: list[dict] = []
    for event_index, event in passes.iterrows():
        frame = int(event["Start Frame"])
        end_frame = int(event["End Frame"])
        if (
            frame not in home.index
            or frame not in away.index
            or end_frame not in home.index
            or end_frame not in away.index
        ):
            continue
        home_positions = _positions(home.loc[frame])
        away_positions = _positions(away.loc[frame])
        teammates, opponents = (
            (home_positions, away_positions)
            if event["Team"] == "Home"
            else (away_positions, home_positions)
        )
        end_opponents = (
            _positions(away.loc[end_frame])
            if event["Team"] == "Home"
            else _positions(home.loc[end_frame])
        )
        end_teammates = (
            _positions(home.loc[end_frame])
            if event["Team"] == "Home"
            else _positions(away.loc[end_frame])
        )
        if len(teammates) < 2 or len(opponents) < 2 or len(end_opponents) < 2:
            continue
        start = np.array(
            [float(event["Start X"]) * PITCH_LENGTH, float(event["Start Y"]) * PITCH_WIDTH]
        )
        end = np.array(
            [float(event["End X"]) * PITCH_LENGTH, float(event["End Y"]) * PITCH_WIDTH]
        )
        pass_vector = end - start
        defender_start = np.linalg.norm(opponents - start, axis=1)
        defender_target = np.linalg.norm(opponents - end, axis=1)
        support_target = np.linalg.norm(teammates - end, axis=1)
        lane = [point_segment_distance(player, start, end) for player in opponents]
        arrival_pressure = float(np.linalg.norm(end_opponents - end, axis=1).min())
        arrival_support = float(np.linalg.norm(end_teammates - end, axis=1).min())
        rows.append(
            {
                "event_index": int(event_index),
                "team": event["Team"],
                "period": int(event["Period"]),
                "time_s": float(event["Start Time [s]"]),
                "from_player": event["From"],
                "to_player": event["To"],
                "pass_length_m": float(np.linalg.norm(pass_vector)),
                "delta_x_m": float(pass_vector[0]),
                "delta_y_m": float(pass_vector[1]),
                "nearest_defender_start_m": float(defender_start.min()),
                "nearest_defender_target_m": float(defender_target.min()),
                "lane_clearance_m": float(min(lane)),
                "nearest_support_target_m": float(support_target.min()),
                "team_width_m": float(np.ptp(teammates[:, 1])),
                "opponent_compactness_m": _compactness(opponents),
                "start_x_m": float(start[0]),
                "start_y_m": float(start[1]),
                "end_x_m": float(end[0]),
                "end_y_m": float(end[1]),
                "turnover_within_5s": int(event["turnover_within_5s"]),
                "arrival_pressure_m": arrival_pressure,
                "arrival_support_m": arrival_support,
                "receiver_pressured_at_arrival": int(arrival_pressure <= 8.0),
            }
        )
    table = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)
    if table.empty:
        raise ValueError("No synchronized pass samples could be constructed")
    return table.sort_values(["period", "time_s", "event_index"]).reset_index(drop=True)
