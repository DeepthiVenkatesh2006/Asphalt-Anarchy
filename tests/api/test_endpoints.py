import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.api.service import MapMatchingService
from src.visualization.map_builder import build_folium_map

class TestAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = MapMatchingService()

    def test_health_service(self):
        trips = self.service.list_available_trips()
        self.assertGreater(len(trips), 0)
        self.assertIsNotNone(self.service.champion_model)

    def test_list_trips(self):
        trips = self.service.list_available_trips()
        partitions = {t["partition"] for t in trips}
        self.assertIn("student_test", partitions)

    def test_pipeline_on_student_test_trip(self):
        res = self.service.process_trip("00000000", partition="student_test")
        self.assertEqual(res["trip_id"], "00000000")
        m = res["metrics"]
        self.assertGreater(m["top1_accuracy"], 0.70)
        self.assertGreater(m["route_continuity"], 0.80)
        self.assertGreater(len(res["predicted_route"]), 0)

    def test_folium_map_generation(self):
        res = self.service.process_trip("00000000", partition="student_test")
        out_path = os.path.join(PROJECT_ROOT, "artifacts", "maps", "test_00000000.html")
        m = build_folium_map(res, output_html_path=out_path)
        self.assertTrue(os.path.exists(out_path))

if __name__ == "__main__":
    unittest.main(verbosity=2)
