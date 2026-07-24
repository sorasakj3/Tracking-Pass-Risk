import numpy as np

from tracking_pass_risk.features import point_segment_distance


def test_point_on_lane_has_zero_clearance() -> None:
    distance = point_segment_distance(
        np.array([5.0, 0.0]), np.array([0.0, 0.0]), np.array([10.0, 0.0])
    )
    assert distance == 0.0


def test_projection_is_clipped_to_segment() -> None:
    distance = point_segment_distance(
        np.array([12.0, 0.0]), np.array([0.0, 0.0]), np.array([10.0, 0.0])
    )
    assert distance == 2.0

