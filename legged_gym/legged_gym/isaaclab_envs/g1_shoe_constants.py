"""G1 29-DoF with-shoe action and asset contract for the Isaac Lab HoST backend.

This file intentionally avoids importing Isaac Lab.  The validator and URDF
preparation path should be runnable with a plain Python interpreter.
"""

from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from legged_gym import LEGGED_GYM_ROOT_DIR


LEGGED_GYM_ROOT_PATH = Path(LEGGED_GYM_ROOT_DIR)
ISAAC_WS_SRC_PATH = LEGGED_GYM_ROOT_PATH.parent

SOURCE_G1_SHOE_URDF_RELATIVE_PATH = Path(
    "unitree_rl_lab/source/unitree_rl_lab/instinctlab/tasks/parkour/urdf/"
    "g1_29dof_torsoBase_popsicle_with_shoe.urdf"
)


def _resolve_source_g1_shoe_urdf_path() -> Path:
    """Resolve the read-only InstinctLab source asset across workspace layouts."""

    candidates: list[Path] = []
    env_path = os.environ.get("HOST_G1_SHOE_SOURCE_URDF")
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.extend(
        [
            ISAAC_WS_SRC_PATH / SOURCE_G1_SHOE_URDF_RELATIVE_PATH,
            Path("/home/weijielai/isaac_ws/src") / SOURCE_G1_SHOE_URDF_RELATIVE_PATH,
            LEGGED_GYM_ROOT_PATH.parent.parent / SOURCE_G1_SHOE_URDF_RELATIVE_PATH,
        ]
    )
    for path in candidates:
        if path.is_file():
            return path
    return candidates[0]


SOURCE_G1_SHOE_URDF_PATH = _resolve_source_g1_shoe_urdf_path()
HOST_G1_SHOE_URDF_PATH = (
    LEGGED_GYM_ROOT_PATH / "resources/robots/g1/g1_29dof_torsoBase_popsicle_with_shoe_host.urdf"
)

NUM_ACTIONS = 29
ONE_STEP_OBS_SIZE = 93
HISTORY_LENGTH = 6
OBS_SIZE = ONE_STEP_OBS_SIZE * HISTORY_LENGTH

BASE_BODY_NAME = "torso_link"
LEFT_FOOT_BODY_NAME = "left_ankle_roll_link"
RIGHT_FOOT_BODY_NAME = "right_ankle_roll_link"
LEFT_SHOE_FRAME_NAME = "LL_FOOT"
RIGHT_SHOE_FRAME_NAME = "LR_FOOT"
HEAD_BODY_NAME = "head_link"
KEYFRAME_HEAD_BODY_NAME = "keyframe_head_link"
SHOE_HEIGHT_OFFSET = 0.058
DEFAULT_MOTION_DIR = Path("/home/weijielai/Datasets/xloco_standup")
DEFAULT_MOTION_SELECTION = "standup_motion_selection.yaml"

SOLE_POINT_OFFSETS = (
    (0.035, 0.10, -SHOE_HEIGHT_OFFSET),
    (0.035, -0.10, -SHOE_HEIGHT_OFFSET),
    (0.20, 0.0, -SHOE_HEIGHT_OFFSET),
    (-0.15, 0.0, -SHOE_HEIGHT_OFFSET),
)

SUCCESS_THRESHOLDS = {
    "hold_time_s": 1.0,
    "base_height": 0.68,
    "head_height": 1.10,
    "uprightness": 0.92,
    "base_lin_xy": 0.30,
    "base_lin_z": 0.20,
    "base_ang_xy": 0.45,
    "leg_joint_error": 0.40,
    "contact_force": 5.0,
}

NONFOOT_CONTACT_BODY_NAMES = (
    "pelvis",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
)

URDF_G1_29DOF_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

XLoco_G1_29DOF_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "waist_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "waist_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "waist_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]

LEG_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
]

WAIST_JOINT_NAMES = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]

UPPER_BODY_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    *WAIST_JOINT_NAMES,
]

DEFAULT_JOINT_POS = {name: 0.0 for name in XLoco_G1_29DOF_JOINT_NAMES}
DEFAULT_JOINT_POS.update(
    {
        "left_hip_pitch_joint": -0.312,
        "right_hip_pitch_joint": -0.312,
        "left_knee_joint": 0.669,
        "right_knee_joint": 0.669,
        "left_ankle_pitch_joint": -0.363,
        "right_ankle_pitch_joint": -0.363,
        "left_elbow_joint": 0.6,
        "right_elbow_joint": 0.6,
        "left_shoulder_roll_joint": 0.2,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "right_shoulder_pitch_joint": 0.2,
    }
)

STANDUP_LEG_TARGET_POS = {name: DEFAULT_JOINT_POS[name] for name in LEG_JOINT_NAMES}

# BeyondMimic actuator constants copied from unitree_rl_lab's current G1 asset
# definition.  Keeping the numeric derivation here makes the action scale
# contract auditable without importing that dirty worktree.
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10.0 * 2.0 * math.pi
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ


def _scale(effort: float, stiffness: float) -> float:
    return 0.25 * effort / stiffness


def _joint_scale(name: str) -> float:
    if "_hip_roll_joint" in name or "_knee_joint" in name:
        return _scale(139.0, STIFFNESS_7520_22)
    if "_hip_pitch_joint" in name or "_hip_yaw_joint" in name or name == "waist_yaw_joint":
        return _scale(88.0, STIFFNESS_7520_14)
    if "_ankle_" in name or name in ("waist_roll_joint", "waist_pitch_joint"):
        return _scale(50.0, 2.0 * STIFFNESS_5020)
    if "_wrist_pitch_joint" in name or "_wrist_yaw_joint" in name:
        return _scale(5.0, STIFFNESS_4010)
    return _scale(25.0, STIFFNESS_5020)


ACTION_SCALE = {name: _joint_scale(name) for name in XLoco_G1_29DOF_JOINT_NAMES}
ACTION_SCALE_IN_POLICY_ORDER = [ACTION_SCALE[name] for name in XLoco_G1_29DOF_JOINT_NAMES]

KEYFRAME_ATTACHMENTS = {
    "keyframe_head_link": ("torso_link", "0 0 0.45"),
    "keyframe_torso_link": ("torso_link", "0 0 0.2"),
    "keyframe_pelvis_link": ("pelvis", "0 0 -0.075"),
    "keyframe_left_collar_link": ("left_shoulder_pitch_link", "0 0 0"),
    "keyframe_right_collar_link": ("right_shoulder_pitch_link", "0 0 0"),
    "keyframe_left_shoulder_link": ("left_shoulder_roll_link", "0 0 0"),
    "keyframe_right_shoulder_link": ("right_shoulder_roll_link", "0 0 0"),
    "keyframe_left_elbow_link": ("left_elbow_link", "0 0 0"),
    "keyframe_right_elbow_link": ("right_elbow_link", "0 0 0"),
    "keyframe_left_wrist_link": ("left_wrist_yaw_link", "0 0 0"),
    "keyframe_right_wrist_link": ("right_wrist_yaw_link", "0 0 0"),
    "keyframe_left_hip_link": ("left_hip_roll_link", "0 0 0"),
    "keyframe_right_hip_link": ("right_hip_roll_link", "0 0 0"),
    "keyframe_left_knee_link": ("left_knee_link", "0 0 0"),
    "keyframe_right_knee_link": ("right_knee_link", "0 0 0"),
    "keyframe_left_ankle_link": ("left_ankle_roll_link", "0 0 0"),
    "keyframe_right_ankle_link": ("right_ankle_roll_link", "0 0 0"),
}

AUXILIARY_ANKLE_ATTACHMENTS = {
    "left": {
        "parent": "left_ankle_roll_link",
        "offsets": ["0.035 0.1 0", "0.035 -0.1 0", "0.2 0 0", "-0.15 0 0"],
    },
    "right": {
        "parent": "right_ankle_roll_link",
        "offsets": ["0.035 0.1 0", "0.035 -0.1 0", "0.2 0 0", "-0.15 0 0"],
    },
}


def _add_marker_link(robot: ET.Element, name: str) -> None:
    link = ET.SubElement(robot, "link", {"name": name})
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": "0.001"})
    ET.SubElement(
        inertial,
        "inertia",
        {"ixx": "1e-7", "ixy": "0", "ixz": "0", "iyy": "1e-7", "iyz": "0", "izz": "1e-7"},
    )


def _add_fixed_joint(robot: ET.Element, name: str, parent: str, child: str, xyz: str) -> None:
    joint = ET.SubElement(robot, "joint", {"name": name, "type": "fixed", "dont_collapse": "true"})
    ET.SubElement(joint, "origin", {"xyz": xyz, "rpy": "0 0 0"})
    ET.SubElement(joint, "parent", {"link": parent})
    ET.SubElement(joint, "child", {"link": child})


def ensure_host_g1_shoe_urdf() -> Path:
    """Create the HoST-compatible 29-DoF with-shoe URDF copy if needed."""

    if not SOURCE_G1_SHOE_URDF_PATH.is_file():
        if HOST_G1_SHOE_URDF_PATH.is_file():
            return HOST_G1_SHOE_URDF_PATH
        raise FileNotFoundError(f"Missing source shoe URDF: {SOURCE_G1_SHOE_URDF_PATH}")

    needs_update = not HOST_G1_SHOE_URDF_PATH.is_file()
    if not needs_update:
        needs_update = SOURCE_G1_SHOE_URDF_PATH.stat().st_mtime > HOST_G1_SHOE_URDF_PATH.stat().st_mtime
    if not needs_update:
        return HOST_G1_SHOE_URDF_PATH

    tree = ET.parse(SOURCE_G1_SHOE_URDF_PATH)
    robot = tree.getroot()
    existing_links = {link.attrib["name"] for link in robot.findall("link") if "name" in link.attrib}
    existing_joints = {joint.attrib["name"] for joint in robot.findall("joint") if "name" in joint.attrib}

    for link_name, (parent_name, xyz) in KEYFRAME_ATTACHMENTS.items():
        if parent_name not in existing_links:
            raise ValueError(f"Cannot attach {link_name}: parent link {parent_name!r} is missing")
        if link_name not in existing_links:
            _add_marker_link(robot, link_name)
            existing_links.add(link_name)
        joint_name = f"{link_name}_joint"
        if joint_name not in existing_joints:
            _add_fixed_joint(robot, joint_name, parent_name, link_name, xyz)
            existing_joints.add(joint_name)

    for side, spec in AUXILIARY_ANKLE_ATTACHMENTS.items():
        parent_name = spec["parent"]
        if parent_name not in existing_links:
            raise ValueError(f"Cannot attach auxiliary {side} ankle links: parent {parent_name!r} is missing")
        for idx, xyz in enumerate(spec["offsets"], start=1):
            link_name = f"auxiliary_{side}_ankle_roll_link{idx}"
            if link_name not in existing_links:
                _add_marker_link(robot, link_name)
                existing_links.add(link_name)
            joint_name = f"auxiliary_{side}_ankle_roll_joint{idx}"
            if joint_name not in existing_joints:
                _add_fixed_joint(robot, joint_name, parent_name, link_name, xyz)
                existing_joints.add(joint_name)

    HOST_G1_SHOE_URDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(HOST_G1_SHOE_URDF_PATH, encoding="utf-8", xml_declaration=True)
    return HOST_G1_SHOE_URDF_PATH


def parse_urdf_contract(path: Path) -> dict[str, set[str] | list[str]]:
    root = ET.parse(path).getroot()
    links = {element.attrib["name"] for element in root.findall("link") if "name" in element.attrib}
    joints = []
    moving_joints = []
    for element in root.findall("joint"):
        name = element.attrib.get("name")
        joint_type = element.attrib.get("type", "")
        if not name:
            continue
        joints.append(name)
        if joint_type not in ("fixed", "floating"):
            moving_joints.append(name)
    return {"links": links, "joints": joints, "moving_joints": moving_joints}


def match_action_scale(scale_by_pattern: dict[str, float], joint_name: str) -> float:
    for pattern, value in scale_by_pattern.items():
        if pattern == joint_name or re.fullmatch(pattern, joint_name):
            return value
    raise KeyError(joint_name)
