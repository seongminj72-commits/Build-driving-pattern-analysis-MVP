import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.trip_analysis import analyze, route_segments


FIXTURE_DIR = Path(__file__).parent / 'fixtures' / 'trips'


class ValidationAnalysisTests(unittest.TestCase):
    def load(self, name):
        return pd.read_csv(FIXTURE_DIR / f'{name}.csv')

    def assert_close(self, actual, expected):
        self.assertTrue(np.isclose(actual, expected, rtol=0, atol=1e-9), (actual, expected))

    def test_constant_speed(self):
        points, events, summary, report = analyze(self.load('constant_speed'))
        np.testing.assert_allclose(points.acceleration_mps2.iloc[1:], [0, 0, 0], rtol=0, atol=1e-9)
        self.assertTrue(pd.isna(points.acceleration_mps2.iloc[0]))
        self.assertEqual(len(events), 0)
        self.assertEqual(summary['acceleration_count'], 0)
        self.assertEqual(summary['braking_count'], 0)
        self.assertEqual(summary['duration_seconds'], 6)
        self.assert_close(summary['average_speed'], 36)
        self.assertEqual(summary['max_speed'], 36)
        self.assertEqual(report['긴 측정 공백'], 0)
        self.assertEqual(report['GPS 결측·오류 (위치 분석만 제외)'], 4)
        self.assertEqual(route_segments(points), [])

    def test_constant_acceleration_is_one_merged_event(self):
        points, events, summary, _ = analyze(self.load('constant_acceleration'))
        np.testing.assert_allclose(points.acceleration_mps2.iloc[1:], [3, 3, 3], rtol=0, atol=1e-9)
        self.assertEqual(len(events), 1)
        event = events.iloc[0]
        self.assertEqual(event.event_type, '급가속')
        self.assertEqual(event.start_time, pd.Timestamp('2026-01-01T00:00:00Z'))
        self.assertEqual(event.end_time, pd.Timestamp('2026-01-01T00:00:06Z'))
        self.assertEqual(event.duration_seconds, 6)
        self.assert_close(event.peak_acceleration_mps2, 3)
        self.assertEqual(summary['acceleration_count'], 1)
        self.assertEqual(summary['braking_count'], 0)
        self.assertEqual(summary['duration_seconds'], 6)
        self.assert_close(summary['average_speed'], 32.4)
        self.assertEqual(summary['max_speed'], 64.8)

    def test_stop_and_gap_do_not_create_false_acceleration(self):
        points, events, summary, report = analyze(self.load('stop_time_gap'))
        np.testing.assert_allclose(points.acceleration_mps2.iloc[[1, 2, 3, 5, 6]], [-4, -4, 0, 0, 0], rtol=0, atol=1e-9)
        self.assertTrue(pd.isna(points.acceleration_mps2.iloc[0]))
        self.assertTrue(pd.isna(points.acceleration_mps2.iloc[4]))
        self.assertEqual(points.interval_seconds.iloc[4], 0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0].event_type, '급감속')
        self.assertEqual(events.iloc[0].duration_seconds, 2)
        self.assert_close(events.iloc[0].peak_acceleration_mps2, -4)
        self.assertEqual(summary['acceleration_count'], 0)
        self.assertEqual(summary['braking_count'], 1)
        self.assertEqual(summary['duration_seconds'], 7)
        self.assert_close(summary['average_speed'], 385.2 / 7)
        self.assertEqual(summary['max_speed'], 118.8)
        self.assertEqual(report['긴 측정 공백'], 1)
        self.assertEqual(route_segments(points), [])


if __name__ == '__main__':
    unittest.main()
