import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

class DashboardTests(unittest.TestCase):
    def test_sample_and_demo(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'dashboard/app.py')).run(timeout=30)
        self.assertFalse(list(app.exception))
        self.assertFalse(list(app.error))
        self.assertEqual(len(app.metric),5)
        app.checkbox[1].check().run(timeout=30)
        self.assertFalse(list(app.exception))
        self.assertFalse(list(app.error))
        self.assertTrue(list(app.warning))
        self.assertEqual(app.dataframe[-1].value.event_id.nunique(),2)
        app.selectbox[0].select('기존 군집 분석 예제').run(timeout=30)
        self.assertFalse(list(app.exception))
        self.assertFalse(list(app.error))
