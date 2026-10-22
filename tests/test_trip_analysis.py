import unittest
import pandas as pd
from src.trip_analysis import analyze, route_segments, normalize_hotspots, nearby_events

class AnalysisTests(unittest.TestCase):
    def sample(self, seconds, speeds, **extra):
        return pd.DataFrame(dict(timestamp=[f'2026-01-01T00:00:{s:02}' for s in seconds],speed=speeds,**extra))

    def test_weighted_speed_and_merged_event(self):
        p,e,s,_=analyze(self.sample([0,1,3],[0,36,72]))
        self.assertEqual(len(e),1)
        self.assertEqual(e.iloc[0].duration_seconds,3)
        self.assertAlmostEqual(s['average_speed'],42)
        self.assertEqual(s['duration_seconds'],3)

    def test_trip_gap_and_gps_breaks(self):
        p,e,s,_=analyze(self.sample([0,1,20,21],[0,36,72,108],latitude=[37,None,37,37],longitude=[127,127,127,127]))
        self.assertEqual(len(e),2)
        self.assertEqual(len(route_segments(p)),1)
        self.assertEqual(s['duration_seconds'],2)
        p,e,s,_=analyze(self.sample([0,1],[0,100],trip_id=['a','b']))
        self.assertTrue(e.empty)
        self.assertIsNone(s['average_speed'])

    def test_invalid_and_duplicate(self):
        p,e,s,r=analyze(self.sample([0,1,1,2,3,4],[0,36,80,-1,'bad',72]))
        self.assertEqual(r['식별자·시각·속도 오류 제외'],2)
        self.assertEqual(r['동일 주행 중복 시각 제외'],1)
        self.assertTrue(pd.isna(p.iloc[-1].acceleration_mps2))
        self.assertEqual(len(p),3)

    def test_hotspot_mapping_and_matching(self):
        h,n=normalize_hotspots(pd.DataFrame({'위도':[37,999],'경도':[127,127],'사고지역위치명':['용인시 예제','오류']}))
        self.assertEqual(n,1)
        self.assertTrue(pd.isna(h.iloc[0].accident_count))
        e=pd.DataFrame({'event_id':['e','f'],'latitude':[37,None],'longitude':[127,None]})
        self.assertEqual(len(nearby_events(e,h)),1)
        self.assertEqual(nearby_events(e,h).iloc[0].distance_m,0)

    def test_missing_columns(self):
        with self.assertRaisesRegex(ValueError,'필수 컬럼'):
            analyze(pd.DataFrame({'rpm':[1]}))

    def test_missing_peak_gps_not_invented(self):
        p,e,_,_=analyze(self.sample([0,1,2],[0,36,50],latitude=[37,None,37],longitude=[127,None,127]))
        self.assertTrue(pd.isna(e.iloc[0].latitude))
        self.assertEqual(route_segments(p),[])

if __name__=='__main__':
    unittest.main()
