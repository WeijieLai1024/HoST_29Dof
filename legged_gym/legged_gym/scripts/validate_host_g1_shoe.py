from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legged_gym.isaaclab_envs.g1_shoe_constants import (  # noqa: E402
    ACTION_SCALE,
    ACTION_SCALE_EXPECTED_RANGES,
    ACTION_SCALE_GROUPS,
    ACTUATOR_EFFORT_LIMITS,
    ACTUATOR_JOINT_GROUPS,
    BASE_BODY_NAME,
    DEFAULT_MOTION_DIR,
    DEFAULT_MOTION_SELECTION,
    DEFAULT_JOINT_POS,
    HEAD_BODY_NAME,
    HISTORY_LENGTH,
    HOST_G1_SHOE_URDF_PATH,
    KEYFRAME_HEAD_BODY_NAME,
    LEFT_FOOT_BODY_NAME,
    LEFT_SHOE_FRAME_NAME,
    NONFOOT_CONTACT_BODY_NAMES,
    NUM_ACTIONS,
    OBS_SIZE,
    ONE_STEP_OBS_SIZE,
    RIGHT_FOOT_BODY_NAME,
    RIGHT_SHOE_FRAME_NAME,
    SHOE_HEIGHT_OFFSET,
    SOLE_POINT_OFFSETS,
    SOURCE_G1_SHOE_URDF_PATH,
    SUCCESS_THRESHOLDS,
    URDF_G1_29DOF_JOINT_NAMES,
    XLoco_G1_29DOF_JOINT_NAMES,
    ensure_host_g1_shoe_urdf,
    parse_urdf_contract,
)
from legged_gym.isaaclab_envs.g1_shoe_motion import validate_motion_dataset  # noqa: E402


REQUIRED_LINKS = {
    BASE_BODY_NAME,
    HEAD_BODY_NAME,
    KEYFRAME_HEAD_BODY_NAME,
    LEFT_FOOT_BODY_NAME,
    RIGHT_FOOT_BODY_NAME,
    "LL_FOOT",
    "LR_FOOT",
    "keyframe_torso_link",
    "keyframe_pelvis_link",
    "keyframe_left_wrist_link",
    "keyframe_right_wrist_link",
    "keyframe_left_ankle_link",
    "keyframe_right_ankle_link",
    LEFT_SHOE_FRAME_NAME,
    RIGHT_SHOE_FRAME_NAME,
    "auxiliary_left_ankle_roll_link1",
    "auxiliary_left_ankle_roll_link2",
    "auxiliary_left_ankle_roll_link3",
    "auxiliary_left_ankle_roll_link4",
    "auxiliary_right_ankle_roll_link1",
    "auxiliary_right_ankle_roll_link2",
    "auxiliary_right_ankle_roll_link3",
    "auxiliary_right_ankle_roll_link4",
}


def _fail(message: str) -> None:
    raise SystemExit(f"[host-g1-shoe validator] FAIL: {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate HoST G1 29-DoF shoe static contracts.")
    parser.add_argument("--motion_dir", type=str, default=str(DEFAULT_MOTION_DIR))
    parser.add_argument("--motion_selection", type=str, default=DEFAULT_MOTION_SELECTION)
    parser.add_argument("--disable_motion_reference", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated_path = ensure_host_g1_shoe_urdf()
    host_contract = parse_urdf_contract(generated_path)

    host_moving = host_contract["moving_joints"]
    host_links = host_contract["links"]
    host_collision_links = host_contract["collision_links"]
    host_fixed_children = host_contract["fixed_joint_children"]

    if SOURCE_G1_SHOE_URDF_PATH.is_file():
        source_contract = parse_urdf_contract(SOURCE_G1_SHOE_URDF_PATH)
        source_moving = source_contract["moving_joints"]
        source_collision_links = source_contract["collision_links"]
        if len(source_moving) != NUM_ACTIONS:
            _fail(f"source URDF moving joints expected {NUM_ACTIONS}, got {len(source_moving)}")
        if list(source_moving) != URDF_G1_29DOF_JOINT_NAMES:
            _fail(f"source URDF joint order changed: {source_moving}")
        if LEFT_FOOT_BODY_NAME not in source_collision_links or RIGHT_FOOT_BODY_NAME not in source_collision_links:
            _fail("source URDF foot collision must stay on ankle_roll links")

    if len(host_moving) != NUM_ACTIONS:
        _fail(f"generated URDF moving joints expected {NUM_ACTIONS}, got {len(host_moving)}")
    if set(host_moving) != set(XLoco_G1_29DOF_JOINT_NAMES):
        missing = sorted(set(XLoco_G1_29DOF_JOINT_NAMES) - set(host_moving))
        extra = sorted(set(host_moving) - set(XLoco_G1_29DOF_JOINT_NAMES))
        _fail(f"joint set mismatch; missing={missing}, extra={extra}")

    missing_links = sorted(REQUIRED_LINKS - set(host_links))
    if missing_links:
        _fail(f"generated URDF missing links: {missing_links}")
    missing_nonfoot = sorted(set(NONFOOT_CONTACT_BODY_NAMES) - set(host_links))
    if missing_nonfoot:
        _fail(f"non-foot contact body names missing in URDF: {missing_nonfoot}")
    if LEFT_FOOT_BODY_NAME not in host_collision_links or RIGHT_FOOT_BODY_NAME not in host_collision_links:
        _fail("generated URDF foot collision must stay on ankle_roll links")
    shoe_collision = sorted({LEFT_SHOE_FRAME_NAME, RIGHT_SHOE_FRAME_NAME} & set(host_collision_links))
    if shoe_collision:
        _fail(f"shoe fixed frames must not be used as collision bodies: {shoe_collision}")
    for frame_name in (LEFT_SHOE_FRAME_NAME, RIGHT_SHOE_FRAME_NAME):
        if frame_name not in host_fixed_children:
            _fail(f"{frame_name} must be preserved as a fixed child frame")

    missing_scales = [name for name in XLoco_G1_29DOF_JOINT_NAMES if name not in ACTION_SCALE]
    if missing_scales:
        _fail(f"action scale missing joints: {missing_scales}")
    for group_name, joint_names in ACTION_SCALE_GROUPS.items():
        lower, upper = ACTION_SCALE_EXPECTED_RANGES[group_name]
        for joint_name in joint_names:
            value = ACTION_SCALE[joint_name]
            if not lower <= value <= upper:
                _fail(f"action scale out of expected range for {joint_name}: {value} not in [{lower}, {upper}]")
    if set(DEFAULT_JOINT_POS) != set(XLoco_G1_29DOF_JOINT_NAMES):
        missing = sorted(set(XLoco_G1_29DOF_JOINT_NAMES) - set(DEFAULT_JOINT_POS))
        extra = sorted(set(DEFAULT_JOINT_POS) - set(XLoco_G1_29DOF_JOINT_NAMES))
        _fail(f"default joint coverage mismatch; missing={missing}, extra={extra}")
    actuator_covered = [joint for names in ACTUATOR_JOINT_GROUPS.values() for joint in names]
    duplicates = sorted({joint for joint in actuator_covered if actuator_covered.count(joint) > 1})
    if duplicates:
        _fail(f"actuator group overlap: {duplicates}")
    if set(actuator_covered) != set(XLoco_G1_29DOF_JOINT_NAMES):
        missing = sorted(set(XLoco_G1_29DOF_JOINT_NAMES) - set(actuator_covered))
        extra = sorted(set(actuator_covered) - set(XLoco_G1_29DOF_JOINT_NAMES))
        _fail(f"actuator group coverage mismatch; missing={missing}, extra={extra}")
    if set(ACTUATOR_EFFORT_LIMITS) != set(XLoco_G1_29DOF_JOINT_NAMES):
        _fail("actuator effort limit coverage mismatch")
    if len(XLoco_G1_29DOF_JOINT_NAMES) != NUM_ACTIONS:
        _fail(f"policy joint order size expected {NUM_ACTIONS}, got {len(XLoco_G1_29DOF_JOINT_NAMES)}")
    if XLoco_G1_29DOF_JOINT_NAMES == URDF_G1_29DOF_JOINT_NAMES:
        _fail("policy semantic order unexpectedly equals URDF order; remap contract is not being exercised")
    if ONE_STEP_OBS_SIZE != 3 + 3 + 3 * NUM_ACTIONS:
        _fail(f"one-step observation size mismatch: {ONE_STEP_OBS_SIZE}")
    if OBS_SIZE != ONE_STEP_OBS_SIZE * HISTORY_LENGTH:
        _fail(f"history observation size mismatch: {OBS_SIZE}")
    if SHOE_HEIGHT_OFFSET <= 0.0:
        _fail(f"invalid shoe height offset: {SHOE_HEIGHT_OFFSET}")
    if len(SOLE_POINT_OFFSETS) != 4:
        _fail(f"expected four virtual sole points, got {len(SOLE_POINT_OFFSETS)}")
    if SUCCESS_THRESHOLDS["head_height"] <= SUCCESS_THRESHOLDS["base_height"]:
        _fail(f"success height thresholds look invalid: {SUCCESS_THRESHOLDS}")

    urdf_joint_order = list(host_moving)
    policy_to_urdf_joint_order = [urdf_joint_order.index(name) for name in XLoco_G1_29DOF_JOINT_NAMES]
    motion_summary = None
    if not args.disable_motion_reference:
        try:
            motion_summary = validate_motion_dataset(args.motion_dir, args.motion_selection)
        except Exception as exc:
            _fail(f"motion dataset validation failed: {exc}")
        if motion_summary.joint_order != URDF_G1_29DOF_JOINT_NAMES:
            _fail("motion dataset joint order is not the expected URDF order")
        if motion_summary.low_start_total <= 0:
            _fail("motion dataset low-start coverage is empty across all selected files")

    print("[host-g1-shoe validator] PASS")
    print(f"  source_asset: {SOURCE_G1_SHOE_URDF_PATH}")
    print(f"  generated_asset: {HOST_G1_SHOE_URDF_PATH}")
    print(f"  moving_joints: {len(host_moving)}")
    print(f"  action_dim: {NUM_ACTIONS}")
    print(f"  one_step_obs: {ONE_STEP_OBS_SIZE}")
    print(f"  history_obs: {OBS_SIZE}")
    print(f"  head_body: {HEAD_BODY_NAME}")
    print(f"  keyframe_head_body: {KEYFRAME_HEAD_BODY_NAME}")
    print(f"  foot_bodies: {LEFT_FOOT_BODY_NAME}, {RIGHT_FOOT_BODY_NAME}")
    print(f"  shoe_height_offset: {SHOE_HEIGHT_OFFSET}")
    print(f"  policy_to_urdf_joint_order: {policy_to_urdf_joint_order}")
    print(f"  nonfoot_contact_bodies: {len(NONFOOT_CONTACT_BODY_NAMES)}")
    print(f"  actuator_groups: {sorted(ACTUATOR_JOINT_GROUPS)}")
    if motion_summary is not None:
        print(f"  motion_selection: {motion_summary.selection_path}")
        print(f"  motion_files: {motion_summary.num_selected}")
        print(f"  motion_frames_min_max: {motion_summary.min_frames}, {motion_summary.max_frames}")
        print(f"  motion_duration_min_max: {motion_summary.min_duration_s:.3f}, {motion_summary.max_duration_s:.3f}")
        print(
            "  motion_low_start_min_max_total: "
            f"{motion_summary.low_start_min}, {motion_summary.low_start_max}, {motion_summary.low_start_total}"
        )
        print(f"  motion_low_start_zero_files: {motion_summary.low_start_zero_files}")
        print(f"  policy_to_dataset_joint_order: {motion_summary.policy_to_dataset_joint_order}")


if __name__ == "__main__":
    main()
