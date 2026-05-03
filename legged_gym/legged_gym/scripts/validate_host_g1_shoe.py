from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from legged_gym.isaaclab_envs.g1_shoe_constants import (  # noqa: E402
    ACTION_SCALE,
    BASE_BODY_NAME,
    DEFAULT_MOTION_DIR,
    DEFAULT_MOTION_SELECTION,
    HEAD_BODY_NAME,
    HISTORY_LENGTH,
    HOST_G1_SHOE_URDF_PATH,
    KEYFRAME_HEAD_BODY_NAME,
    LEFT_FOOT_BODY_NAME,
    NUM_ACTIONS,
    OBS_SIZE,
    ONE_STEP_OBS_SIZE,
    RIGHT_FOOT_BODY_NAME,
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

    if SOURCE_G1_SHOE_URDF_PATH.is_file():
        source_contract = parse_urdf_contract(SOURCE_G1_SHOE_URDF_PATH)
        source_moving = source_contract["moving_joints"]
        if len(source_moving) != NUM_ACTIONS:
            _fail(f"source URDF moving joints expected {NUM_ACTIONS}, got {len(source_moving)}")
        if list(source_moving) != URDF_G1_29DOF_JOINT_NAMES:
            _fail(f"source URDF joint order changed: {source_moving}")

    if len(host_moving) != NUM_ACTIONS:
        _fail(f"generated URDF moving joints expected {NUM_ACTIONS}, got {len(host_moving)}")
    if set(host_moving) != set(XLoco_G1_29DOF_JOINT_NAMES):
        missing = sorted(set(XLoco_G1_29DOF_JOINT_NAMES) - set(host_moving))
        extra = sorted(set(host_moving) - set(XLoco_G1_29DOF_JOINT_NAMES))
        _fail(f"joint set mismatch; missing={missing}, extra={extra}")

    missing_links = sorted(REQUIRED_LINKS - set(host_links))
    if missing_links:
        _fail(f"generated URDF missing links: {missing_links}")

    missing_scales = [name for name in XLoco_G1_29DOF_JOINT_NAMES if name not in ACTION_SCALE]
    if missing_scales:
        _fail(f"action scale missing joints: {missing_scales}")
    if len(XLoco_G1_29DOF_JOINT_NAMES) != NUM_ACTIONS:
        _fail(f"policy joint order size expected {NUM_ACTIONS}, got {len(XLoco_G1_29DOF_JOINT_NAMES)}")
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
    if motion_summary is not None:
        print(f"  motion_selection: {motion_summary.selection_path}")
        print(f"  motion_files: {motion_summary.num_selected}")
        print(f"  motion_frames_min_max: {motion_summary.min_frames}, {motion_summary.max_frames}")
        print(f"  policy_to_dataset_joint_order: {motion_summary.policy_to_dataset_joint_order}")


if __name__ == "__main__":
    main()
