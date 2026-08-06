"""Regression tests for explicit --memory_policy mapping and flag gating."""

from __future__ import annotations

import argparse
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from generate_fast import _apply_memory_policy


def _args(**kwargs):
    defaults = dict(
        memory_policy=None,
        enable_motion_adaptive_kv_eviction=False,
        selector="learned",
        selector_ckpt=None,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestMemoryPolicy(unittest.TestCase):
    def test_legacy_preserves_flags(self):
        a = _args(memory_policy="legacy", enable_motion_adaptive_kv_eviction=True, selector="heuristic")
        _apply_memory_policy(a)
        self.assertTrue(a.enable_motion_adaptive_kv_eviction)
        self.assertEqual(a.selector, "heuristic")

    def test_window(self):
        a = _args(memory_policy="window", enable_motion_adaptive_kv_eviction=True, selector="learned")
        _apply_memory_policy(a)
        self.assertFalse(a.enable_motion_adaptive_kv_eviction)
        self.assertEqual(a.selector, "heuristic")

    def test_heuristic_cr(self):
        a = _args(memory_policy="heuristic_cr")
        _apply_memory_policy(a)
        self.assertTrue(a.enable_motion_adaptive_kv_eviction)
        self.assertEqual(a.selector, "heuristic")

    def test_learned_cr(self):
        a = _args(memory_policy="learned_cr", selector="heuristic")
        _apply_memory_policy(a)
        self.assertTrue(a.enable_motion_adaptive_kv_eviction)
        self.assertEqual(a.selector, "learned")
        self.assertTrue(a.selector_ckpt.endswith("selector_all4.pt"))

    def test_world_state_cr_defaults_to_future(self):
        a = _args(memory_policy="world_state_cr", selector="heuristic")
        _apply_memory_policy(a)
        self.assertTrue(a.enable_motion_adaptive_kv_eviction)
        self.assertEqual(a.selector, "learned")
        self.assertTrue(a.selector_ckpt.endswith("selector_ws_future_v1.pt"))

    def test_world_state_cr_future_is_alias(self):
        a = _args(memory_policy="world_state_cr_future", selector="heuristic")
        _apply_memory_policy(a)
        self.assertEqual(a.memory_policy, "world_state_cr")
        self.assertTrue(a.selector_ckpt.endswith("selector_ws_future_v1.pt"))

    def test_world_state_cr_v1_ablation(self):
        a = _args(memory_policy="world_state_cr_v1", selector="heuristic")
        _apply_memory_policy(a)
        self.assertTrue(a.enable_motion_adaptive_kv_eviction)
        self.assertEqual(a.selector, "learned")
        self.assertTrue(a.selector_ckpt.endswith("selector_ws_v1.pt"))

    def test_old_names_rejected(self):
        for policy in ("moce", "m1", "stategraph"):
            a = _args(memory_policy=policy)
            with self.assertRaises(AssertionError):
                _apply_memory_policy(a)

    def test_unknown_raises(self):
        a = _args(memory_policy="not_a_policy")
        with self.assertRaises(AssertionError):
            _apply_memory_policy(a)


class TestBatchGenerateMethodNames(unittest.TestCase):
    def test_official_methods_only(self):
        import importlib.util
        path = os.path.join(ROOT, "bench", "batch_generate.py")
        spec = importlib.util.spec_from_file_location("batch_generate_mod", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        expected = {
            "window", "heuristic_cr", "learned_cr",
            "world_state_cr", "world_state_cr_future", "world_state_cr_v1",
            "swtp", "mosaic",
        }
        self.assertEqual(set(mod.METHODS), expected)
        # Default world_state_cr uses future-use ckpt.
        self.assertTrue(
            mod.METHODS["world_state_cr"]["selector_ckpt"].endswith(
                "selector_ws_future_v1.pt"))
        self.assertEqual(
            mod.METHODS["world_state_cr"]["selector_ckpt"],
            mod.METHODS["world_state_cr_future"]["selector_ckpt"],
        )
        self.assertTrue(
            mod.METHODS["world_state_cr_v1"]["selector_ckpt"].endswith(
                "selector_ws_v1.pt"))
        for dead in ("m1", "m1_ws", "m1_legacy", "moce", "MoCE", "stategraph", "MoSaiC_2step"):
            self.assertNotIn(dead, mod.METHODS)


if __name__ == "__main__":
    unittest.main()
