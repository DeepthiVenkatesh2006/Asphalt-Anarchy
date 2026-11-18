import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.api.service import MapMatchingService

class TestPipelineIntegration(unittest.TestCase):
    def test_evaluate_endpoint_output_schema(self):
        srv = MapMatchingService()
        res = srv.process_trip("00000007", partition="student_test", use_viterbi=True)
        self.assertIn("metrics", res)
        self.assertIn("predicted_route", res)
        self.assertIn("condensed_route", res)
        self.assertIn("preview_arcs", res)
        
        m = res["metrics"]
        for key in ["total_observations", "top1_accuracy", "route_continuity", "candidate_miss_rate", "inference_time_ms"]:
            self.assertIn(key, m)

if __name__ == "__main__":
    unittest.main(verbosity=2)
