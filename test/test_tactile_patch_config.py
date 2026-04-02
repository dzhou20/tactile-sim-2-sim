import unittest

from tactile.patch_config import (
    TactilePatch,
    TactilePatchRegistry,
    build_patch_sample_points,
)
from tactile.presets.umi import (
    make_umi_finger1_inner_patch,
)


class TactilePatchConfigTest(unittest.TestCase):
    def test_build_patch_sample_points_count(self) -> None:
        patch = TactilePatch(
            name="test_patch",
            parent_prim="/World/Test",
            origin=(0.0, 0.0, 0.0),
            normal=(1.0, 0.0, 0.0),
            axis_u=(0.0, 1.0, 0.0),
            axis_v=(0.0, 0.0, 1.0),
            size_u=0.02,
            size_v=0.01,
            rows=3,
            cols=2,
            sensor_radius=0.002,
        )

        points = build_patch_sample_points(patch)

        self.assertEqual(len(points), 6)
        self.assertEqual(points[0].sensor_name, "Tactile_0_0")
        self.assertEqual(points[-1].sensor_name, "Tactile_2_1")

    def test_registry_rejects_duplicate_names(self) -> None:
        patch = make_umi_finger1_inner_patch()
        registry = TactilePatchRegistry([patch])

        with self.assertRaises(ValueError):
            registry.add(patch)

    def test_patch_rejects_non_orthogonal_axes(self) -> None:
        patch = TactilePatch(
            name="bad_patch",
            parent_prim="/World/Test",
            origin=(0.0, 0.0, 0.0),
            normal=(1.0, 0.0, 0.0),
            axis_u=(1.0, 0.0, 0.0),
            axis_v=(0.0, 0.0, 1.0),
            size_u=0.02,
            size_v=0.01,
            rows=2,
            cols=2,
            sensor_radius=0.002,
        )

        with self.assertRaises(ValueError):
            patch.validate()


if __name__ == "__main__":
    unittest.main()
