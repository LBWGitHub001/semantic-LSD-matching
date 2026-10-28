"""Concrete layer graphs stay aligned with the matching networks."""

from __future__ import annotations

import unittest

from visualization.layer_graph import build_layer_graph


class LayerGraphTests(unittest.TestCase):
    def _assert_well_formed(self, graph: dict) -> None:
        ids = [node["id"] for node in graph["nodes"]]
        self.assertEqual(len(ids), len(set(ids)))
        known = set(ids)
        occupied = set()
        for node in graph["nodes"]:
            self.assertTrue(node["meaning"].strip())
            self.assertGreaterEqual(node["row"], 0)
            cell = (node["col"], node["row"])
            self.assertNotIn(cell, occupied)
            occupied.add(cell)
        for edge in graph["edges"]:
            self.assertIn(edge["from"], known)
            self.assertIn(edge["to"], known)

    def test_xfeat_line_matches_documented_parameter_count(self) -> None:
        graph = build_layer_graph("xfeat_line_matcher", use_pyramid=True, d_model=128)
        self._assert_well_formed(graph)
        self.assertEqual(graph["param_count"], 1_722_081)
        by_id = {node["id"]: node for node in graph["nodes"]}
        self.assertEqual(by_id["to5d"]["formula"], "[cᵤ, cᵥ, ‖Δ‖, sin 2θ, cos 2θ]")
        self.assertIn("二倍角", by_id["to5d"]["meaning"])
        self.assertEqual(by_id["geo_9d"]["shape_out"], "(B, M, 9)")
        self.assertEqual(by_id["dustbin"]["params"], 1)
        self.assertIn("null_line_token", by_id["line_where"]["module"])
        self.assertTrue(any(edge["kind"] == "cross" for edge in graph["edges"]))
        self.assertTrue(any(edge["kind"] == "skip" and edge["to"] == "s1_res_add" for edge in graph["edges"]))

    def test_object_matcher_has_no_line_tower(self) -> None:
        graph = build_layer_graph("xfeat_object")
        self._assert_well_formed(graph)
        ids = {node["id"] for node in graph["nodes"]}
        self.assertNotIn("line_fc1", ids)
        self.assertIn("fuse_fc", ids)
        self.assertIn("sinkhorn", ids)

    def test_mask_line_freezes_stem_and_keeps_canonical_matrix(self) -> None:
        graph = build_layer_graph("mask_line")
        self._assert_well_formed(graph)
        by_id = {node["id"]: node for node in graph["nodes"]}
        self.assertTrue(by_id["stem_conv"]["frozen"])
        self.assertTrue(by_id["l1b0_conv1"]["frozen"])
        self.assertFalse(by_id["l2b0_conv2"]["frozen"])
        self.assertEqual(by_id["stem_conv"]["shape_in"], "(B, 6, 480×640)")
        self.assertIn("matrix_255", by_id)
        self.assertEqual(by_id["dustbin"]["params"], 0)

    def test_line_transformer_freezes_entire_backbone(self) -> None:
        graph = build_layer_graph("line_transformer")
        self._assert_well_formed(graph)
        by_id = {node["id"]: node for node in graph["nodes"]}
        self.assertTrue(by_id["l2b3_relu3"]["frozen"])
        self.assertNotIn("matrix_255", by_id)
        self.assertNotIn("exp", by_id)
        self.assertIn("grid_sample", by_id)


if __name__ == "__main__":
    unittest.main()
