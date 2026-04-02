"""UMI tactile patch presets."""

from tactile.patch_config import TactilePatch, TactilePatchRegistry


def make_umi_finger1_inner_patch() -> TactilePatch:
    """First-pass UMI finger1 tactile patch based on the current script layout."""

    return TactilePatch(
        name="umi_finger1_inner",
        parent_prim="/ur3/umi/finger1_1",
        origin=(0.0, 0.018, 0.003),
        normal=(-1.0, 0.0, 0.0),
        axis_u=(0.0, 1.0, 0.0),
        axis_v=(0.0, 0.0, 1.0),
        size_u=0.036,
        size_v=0.006,
        rows=10,
        cols=4,
        sensor_radius=0.006,
        offset=0.0,
        min_threshold=-1.0,
        max_threshold=100000.0,
        sensor_period=0.0,
        name_prefix="Finger1Contact",
        metadata={
            "robot": "UMI",
            "side": "finger1",
            "surface": "inner_pad",
            "status": "first_pass_preset",
        },
    )


def make_default_patch_registry() -> TactilePatchRegistry:
    return TactilePatchRegistry([make_umi_finger1_inner_patch()])
