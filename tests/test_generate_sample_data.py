import tempfile
import unittest
from pathlib import Path

import pandas as pd

from generate_sample_data import generate_validation_fixtures


class ValidationFixtureTests(unittest.TestCase):
    def test_validation_fixtures_are_deterministic(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            generate_validation_fixtures(first)
            generate_validation_fixtures(second)
            for name in ('constant_speed', 'constant_acceleration', 'stop_time_gap'):
                left = Path(first, f'{name}.csv')
                right = Path(second, f'{name}.csv')
                self.assertEqual(left.read_bytes(), right.read_bytes())
                frame = pd.read_csv(left)
                self.assertEqual(list(frame.columns), ['driver_id', 'trip_id', 'timestamp', 'speed_kmh'])

    def test_fixture_shapes_and_values(self):
        with tempfile.TemporaryDirectory() as directory:
            fixtures = generate_validation_fixtures(directory)
            self.assertEqual(len(fixtures['constant_speed']), 4)
            self.assertEqual(len(fixtures['constant_acceleration']), 4)
            self.assertEqual(len(fixtures['stop_time_gap']), 7)
            self.assertEqual(fixtures['stop_time_gap'].timestamp.iloc[4], '2026-01-01T00:00:15Z')


if __name__ == '__main__':
    unittest.main()
