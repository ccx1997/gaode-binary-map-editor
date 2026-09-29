import unittest
from pathlib import Path

import cv2
import numpy as np

from map_converter import convert_map, estimate_outline_width, read_rgb, thin_centerline


class ConversionTests(unittest.TestCase):
    def test_roads_are_centered_and_buildings_stay_outlines(self):
        rgb = np.full((400, 600, 3), (179, 230, 153), np.uint8)
        rgb[50:120, 100:220] = (217, 227, 236)
        rgb[200:220, 30:570] = (217, 227, 236)
        output, report = convert_map(rgb, 'synthetic.png', 3, 34, 9)
        self.assertEqual(set(np.unique(output)), {0, 127, 255})
        self.assertEqual(report.contour_count, 1)
        self.assertEqual(report.road_component_count, 1)
        road_rows = np.flatnonzero(output[:, 300, 0] == 127)
        self.assertEqual(len(road_rows), 9)
        self.assertLessEqual(abs(road_rows.mean() - 209.5), 1)
        self.assertFalse(np.any(output[180:240, :, 0] == 0))
        self.assertEqual(int(output[85, 160, 0]), 255)
        self.assertTrue(np.any(output[45:55, 120:200, 0] == 0))

    def test_thinning_preserves_loops_and_junctions(self):
        road = np.zeros((180, 220), np.uint8)
        cv2.rectangle(road, (30, 30), (180, 130), 255, 15)
        cv2.line(road, (100, 130), (100, 170), 255, 15)
        skeleton = thin_centerline(road)
        self.assertEqual(cv2.connectedComponents(skeleton, connectivity=8)[0], 2)
        contours, hierarchy = cv2.findContours(skeleton, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        self.assertTrue(any(h[3] >= 0 for h in hierarchy[0]), 'loop must stay closed')
        self.assertTrue(np.any(skeleton[150:165, 98:103]))
        self.assertLess(np.count_nonzero(skeleton), np.count_nonzero(road) / 5)

    def test_exported_palette_round_trip(self):
        gray = np.tile(np.array([0, 127, 255], np.uint8), (60, 20))
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        output, report = convert_map(rgb, 'export.png', 3, 34)
        np.testing.assert_array_equal(output, rgb)
        self.assertTrue(report.input_was_grayscale_map)
        self.assertFalse(report.input_was_binary)

    def test_legacy_binary_round_trip(self):
        rgb = np.full((100, 100, 3), 255, np.uint8)
        rgb[10:90, 20:25] = 0
        output, report = convert_map(rgb, 'binary.png', 3, 34)
        np.testing.assert_array_equal(output, rgb)
        self.assertTrue(report.input_was_binary)

    def test_rectangle_default_matches_rasterized_building_width(self):
        for nominal, actual in [(1, 1), (3, 5), (5, 7)]:
            rgb = np.full((120, 200, 3), 255, np.uint8)
            cv2.rectangle(rgb, (20, 20), (180, 100), (0, 0, 0), nominal)
            self.assertEqual(estimate_outline_width(rgb), actual)
            self.assertEqual(np.count_nonzero(rgb[:40, 100, 0] == 0), actual)
        self.assertEqual(estimate_outline_width(np.full((20, 20, 3), 255, np.uint8)), 5)

    def test_repository_screenshot(self):
        path = Path(__file__).resolve().parents[1] / 'communities/dushiyuan-1/raw_inputs/source.jpeg'
        output, report = convert_map(read_rgb(path), path.name, 3, 34)
        self.assertEqual(report.contour_count, 7)
        self.assertEqual(estimate_outline_width(output), 5)
        self.assertEqual(report.road_component_count, 1)
        self.assertEqual(set(np.unique(output)), {0, 127, 255})
        self.assertTrue(np.all(output[100:300, :14] == 255), 'screenshot gutter is not a road')
        roads = (output[:, :, 0] == 127).astype(np.uint8)
        self.assertEqual(cv2.connectedComponents(roads, connectivity=8)[0], 2)


if __name__ == '__main__':
    unittest.main()
