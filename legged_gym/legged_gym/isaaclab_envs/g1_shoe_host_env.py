from __future__ import annotations

from collections.abc import Sequence
import math
import os
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass

from legged_gym.isaaclab_envs.g1_shoe_constants import (
    ACTION_SCALE_IN_POLICY_ORDER,
    BASE_BODY_NAME,
    DAMPING_4010,
    DAMPING_5020,
    DAMPING_7520_14,
    DAMPING_7520_22,
    DEFAULT_JOINT_POS,
    DEFAULT_MOTION_DIR,
    DEFAULT_MOTION_SELECTION,
    HEAD_BODY_NAME,
    HISTORY_LENGTH,
    LEFT_FOOT_BODY_NAME,
    LEG_JOINT_NAMES,
    NONFOOT_CONTACT_BODY_NAMES,
    NUM_ACTIONS,
    OBS_SIZE,
    ONE_STEP_OBS_SIZE,
    RIGHT_FOOT_BODY_NAME,
    SHOE_HEIGHT_OFFSET,
    SOLE_POINT_OFFSETS,
    STANDUP_LEG_TARGET_POS,
    STIFFNESS_4010,
    STIFFNESS_5020,
    STIFFNESS_7520_14,
    STIFFNESS_7520_22,
    SUCCESS_THRESHOLDS,
    UPPER_BODY_JOINT_NAMES,
    WAIST_JOINT_NAMES,
    XLoco_G1_29DOF_JOINT_NAMES,
    ensure_host_g1_shoe_urdf,
)
from legged_gym.isaaclab_envs.g1_shoe_motion import HostG1MotionBank


def _make_delayed_actuators() -> dict[str, DelayedPDActuatorCfg]:
    return {
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_7520_14,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_knee_joint": STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_7520_14,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_knee_joint": DAMPING_7520_22,
            },
            min_delay=0,
            max_delay=2,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            min_delay=0,
            max_delay=2,
        ),
        "waist": DelayedPDActuatorCfg(
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            min_delay=0,
            max_delay=2,
        ),
        "waist_yaw": DelayedPDActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            min_delay=0,
            max_delay=2,
        ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_5020,
                ".*_shoulder_roll_joint": STIFFNESS_5020,
                ".*_shoulder_yaw_joint": STIFFNESS_5020,
                ".*_elbow_joint": STIFFNESS_5020,
                ".*_wrist_roll_joint": STIFFNESS_5020,
                ".*_wrist_pitch_joint": STIFFNESS_4010,
                ".*_wrist_yaw_joint": STIFFNESS_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_5020,
                ".*_shoulder_roll_joint": DAMPING_5020,
                ".*_shoulder_yaw_joint": DAMPING_5020,
                ".*_elbow_joint": DAMPING_5020,
                ".*_wrist_roll_joint": DAMPING_5020,
                ".*_wrist_pitch_joint": DAMPING_4010,
                ".*_wrist_yaw_joint": DAMPING_4010,
            },
            min_delay=0,
            max_delay=2,
        ),
    }


def make_g1_shoe_articulation_cfg() -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(ensure_host_g1_shoe_urdf()),
            fix_base=False,
            root_link_name=BASE_BODY_NAME,
            replace_cylinders_with_capsules=False,
            merge_fixed_joints=False,
            self_collision=True,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.82),
            joint_pos=DEFAULT_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators=_make_delayed_actuators(),
    )


@configclass
class HostG1ShoeDirectEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 10.0
    action_space = NUM_ACTIONS
    observation_space = OBS_SIZE
    state_space = 0
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=decimation)
    robot_cfg: ArticulationCfg = make_g1_shoe_articulation_cfg()
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=SUCCESS_THRESHOLDS["contact_force"],
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=3.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    clip_actions = 1.0
    clip_observations = 100.0
    unactuated_steps = 20
    reset_xy_range = 0.10
    joint_vel_obs_scale = 0.05
    base_ang_vel_obs_scale = 0.25

    use_motion_reference = True
    motion_dir = str(DEFAULT_MOTION_DIR)
    motion_selection = DEFAULT_MOTION_SELECTION
    reset_mode = "mixed"
    motion_reset_probability = 0.70
    reference_root_min_z = 0.28
    action_rescale = 1.0
    success_hold_time = SUCCESS_THRESHOLDS["hold_time_s"]

    target_head_height = SUCCESS_THRESHOLDS["head_height"]
    target_base_height = SUCCESS_THRESHOLDS["base_height"]
    target_uprightness = SUCCESS_THRESHOLDS["uprightness"]
    clean_support_upright_gate = 0.80
    clean_support_base_height_gate = 0.58
    nonfoot_contact_penalty_gate_s = 2.0
    contact_force_threshold = SUCCESS_THRESHOLDS["contact_force"]
    dof_vel_limit = 300.0
    base_vel_limit = 20.0
    fallen_through_z = -0.25
    reference_exhaustion_grace_s = 0.30
    pull_force = 80.0
    pull_force_enabled = True
    pull_force_upright_gate = -0.8

    reward_group_weights = (2.0, 0.75, 1.0, 0.05)


class HostG1ShoeDirectEnv(DirectRLEnv):
    cfg: HostG1ShoeDirectEnvCfg

    def __init__(self, cfg: HostG1ShoeDirectEnvCfg, render_mode: str | None = None, **kwargs):
        if cfg.reset_mode not in ("mixed", "motion", "synthetic"):
            raise ValueError(f"reset_mode must be mixed, motion, or synthetic; got {cfg.reset_mode!r}")
        super().__init__(cfg, render_mode, **kwargs)

        self.policy_joint_ids, policy_names = self.robot.find_joints(
            XLoco_G1_29DOF_JOINT_NAMES, preserve_order=True
        )
        if policy_names != XLoco_G1_29DOF_JOINT_NAMES:
            raise RuntimeError(f"Policy joint order mismatch: {policy_names}")
        self.sim_joint_index_by_policy_action = dict(zip(XLoco_G1_29DOF_JOINT_NAMES, self.policy_joint_ids))

        self.action_scale = torch.tensor(
            ACTION_SCALE_IN_POLICY_ORDER, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self.actions = torch.zeros(self.num_envs, NUM_ACTIONS, dtype=torch.float32, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)
        self.joint_pos_target = self.robot.data.default_joint_pos.clone()
        self.last_joint_vel = self.robot.data.joint_vel.clone()
        self.obs_history = torch.zeros(
            self.num_envs, HISTORY_LENGTH, ONE_STEP_OBS_SIZE, dtype=torch.float32, device=self.device
        )
        self.refresh_history = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.reward_group_weights = torch.tensor(
            self.cfg.reward_group_weights, dtype=torch.float32, device=self.device
        )

        self.success_hold_steps = max(1, int(round(self.cfg.success_hold_time / self.step_dt)))
        self.success_hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.success_ever = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.max_head_height = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.max_base_height = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.max_uprightness = torch.full((self.num_envs,), -1.0, dtype=torch.float32, device=self.device)
        self.last_done_reason = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.motion_bank: HostG1MotionBank | None = None
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.motion_z_shift = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.motion_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.cfg.use_motion_reference:
            self.motion_bank = HostG1MotionBank(
                Path(self.cfg.motion_dir),
                self.cfg.motion_selection,
                device=self.device,
                policy_joint_names=XLoco_G1_29DOF_JOINT_NAMES,
            )

        self._cache_body_and_joint_indices()
        self._cache_contact_indices()
        self._cache_policy_joint_groups()

    def _setup_scene(self):
        debug_setup = os.environ.get("HOST_G1_DEBUG_SETUP", "0") == "1"
        if debug_setup:
            print("[host-g1-shoe setup] begin", flush=True)
        self.robot = Articulation(self.cfg.robot_cfg)
        if debug_setup:
            print("[host-g1-shoe setup] articulation spawned", flush=True)

        ground_cfg = GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.7,
                restitution=0.3,
            )
        )
        spawn_ground_plane(prim_path="/World/ground", cfg=ground_cfg)
        if debug_setup:
            print("[host-g1-shoe setup] ground spawned", flush=True)
        self.scene.clone_environments(copy_from_source=False)
        if debug_setup:
            print("[host-g1-shoe setup] envs cloned", flush=True)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot
        if debug_setup:
            print("[host-g1-shoe setup] articulation registered", flush=True)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_forces"] = self.contact_sensor
        if debug_setup:
            print("[host-g1-shoe setup] contact sensor registered", flush=True)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        if debug_setup:
            print("[host-g1-shoe setup] light spawned", flush=True)

    def _cache_body_and_joint_indices(self) -> None:
        self.base_body_id = self._find_body(BASE_BODY_NAME)
        self.head_body_id = self._find_body(HEAD_BODY_NAME)
        self.left_foot_body_id = self._find_body(LEFT_FOOT_BODY_NAME)
        self.right_foot_body_id = self._find_body(RIGHT_FOOT_BODY_NAME)
        self.left_knee_body_id = self._find_body("left_knee_link")
        self.right_knee_body_id = self._find_body("right_knee_link")

        self.waist_joint_ids = self._find_joints(WAIST_JOINT_NAMES)
        self.hip_yaw_joint_ids = self._find_joints(["left_hip_yaw_joint", "right_hip_yaw_joint"])
        self.hip_roll_joint_ids = self._find_joints(["left_hip_roll_joint", "right_hip_roll_joint"])
        self.knee_joint_ids = self._find_joints(["left_knee_joint", "right_knee_joint"])
        self.shoulder_roll_joint_ids = self._find_joints(["left_shoulder_roll_joint", "right_shoulder_roll_joint"])
        self.upper_body_joint_ids = self._find_joints(UPPER_BODY_JOINT_NAMES)
        self.leg_joint_ids = self._find_joints(LEG_JOINT_NAMES)

    def _cache_contact_indices(self) -> None:
        self.contact_left_foot_ids, left_names = self.contact_sensor.find_bodies(LEFT_FOOT_BODY_NAME, preserve_order=True)
        self.contact_right_foot_ids, right_names = self.contact_sensor.find_bodies(
            RIGHT_FOOT_BODY_NAME, preserve_order=True
        )
        if left_names != [LEFT_FOOT_BODY_NAME] or right_names != [RIGHT_FOOT_BODY_NAME]:
            raise RuntimeError(f"Foot contact body mismatch: left={left_names}, right={right_names}")
        self.contact_nonfoot_ids, self.contact_nonfoot_names = self.contact_sensor.find_bodies(
            list(NONFOOT_CONTACT_BODY_NAMES), preserve_order=True
        )
        if self.contact_nonfoot_names != list(NONFOOT_CONTACT_BODY_NAMES):
            raise RuntimeError(f"Non-foot contact body mismatch: {self.contact_nonfoot_names}")

    def _cache_policy_joint_groups(self) -> None:
        policy_name_to_idx = {name: idx for idx, name in enumerate(XLoco_G1_29DOF_JOINT_NAMES)}
        self.leg_policy_ids = torch.tensor(
            [policy_name_to_idx[name] for name in LEG_JOINT_NAMES], dtype=torch.long, device=self.device
        )
        self.upper_policy_ids = torch.tensor(
            [policy_name_to_idx[name] for name in UPPER_BODY_JOINT_NAMES], dtype=torch.long, device=self.device
        )
        self.wrist_policy_ids = torch.tensor(
            [
                policy_name_to_idx["left_wrist_pitch_joint"],
                policy_name_to_idx["right_wrist_pitch_joint"],
                policy_name_to_idx["left_wrist_yaw_joint"],
                policy_name_to_idx["right_wrist_yaw_joint"],
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.waist_policy_ids = torch.tensor(
            [policy_name_to_idx[name] for name in WAIST_JOINT_NAMES], dtype=torch.long, device=self.device
        )
        self.leg_target_policy = torch.tensor(
            [STANDUP_LEG_TARGET_POS[name] for name in LEG_JOINT_NAMES],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        self.sole_offsets = torch.tensor(SOLE_POINT_OFFSETS, dtype=torch.float32, device=self.device)

    def _find_body(self, name: str) -> int:
        ids, names = self.robot.find_bodies(name, preserve_order=True)
        if len(ids) != 1:
            raise RuntimeError(f"Expected exactly one body for {name!r}, got {names}")
        return ids[0]

    def _find_joints(self, names: list[str] | tuple[str, ...]) -> list[int]:
        ids, resolved = self.robot.find_joints(list(names), preserve_order=True)
        if resolved != list(names):
            raise RuntimeError(f"Joint lookup mismatch: expected {names}, got {resolved}")
        return ids

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        clipped_actions = torch.clamp(actions, -self.cfg.clip_actions, self.cfg.clip_actions)
        active = (self.episode_length_buf >= self.cfg.unactuated_steps).unsqueeze(-1)
        clipped_actions = torch.where(active, clipped_actions, torch.zeros_like(clipped_actions))

        self.last_last_actions.copy_(self.last_actions)
        self.last_actions.copy_(self.actions)
        self.actions.copy_(clipped_actions)

        self.joint_pos_target.copy_(self.robot.data.default_joint_pos)
        default_policy_pos = self.robot.data.default_joint_pos[:, self.policy_joint_ids]
        target_policy_pos = default_policy_pos + self.cfg.action_rescale * self.actions * self.action_scale
        lower = self.robot.data.soft_joint_pos_limits[:, self.policy_joint_ids, 0]
        upper = self.robot.data.soft_joint_pos_limits[:, self.policy_joint_ids, 1]
        target_policy_pos = torch.maximum(torch.minimum(target_policy_pos, upper), lower)
        self.joint_pos_target[:, self.policy_joint_ids] = target_policy_pos

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(
            self.joint_pos_target[:, self.policy_joint_ids], joint_ids=self.policy_joint_ids
        )
        if self.cfg.pull_force_enabled:
            pull_mask = (self.episode_length_buf >= self.cfg.unactuated_steps) & (
                self.robot.data.projected_gravity_b[:, 2] < self.cfg.pull_force_upright_gate
            )
            forces = torch.zeros(self.num_envs, 1, 3, dtype=torch.float32, device=self.device)
            forces[:, 0, 2] = self.cfg.pull_force * pull_mask.float()
            torques = torch.zeros_like(forces)
            self.robot.set_external_force_and_torque(
                forces, torques, body_ids=[self.base_body_id], is_global=True
            )

    def _get_observations(self) -> dict:
        one_step_obs = self._compute_one_step_observation()
        self.obs_history = torch.roll(self.obs_history, shifts=-1, dims=1)
        self.obs_history[:, -1, :] = one_step_obs
        if torch.any(self.refresh_history):
            env_ids = torch.nonzero(self.refresh_history, as_tuple=False).squeeze(-1)
            self.obs_history[env_ids] = one_step_obs[env_ids].unsqueeze(1).repeat(1, HISTORY_LENGTH, 1)
            self.refresh_history[env_ids] = False
        obs = self.obs_history.reshape(self.num_envs, -1)
        obs = torch.clamp(obs, -self.cfg.clip_observations, self.cfg.clip_observations)
        return {"policy": obs}

    def _compute_one_step_observation(self) -> torch.Tensor:
        joint_ids = self.policy_joint_ids
        joint_pos_rel = self.robot.data.joint_pos[:, joint_ids] - self.robot.data.default_joint_pos[:, joint_ids]
        joint_vel = self.robot.data.joint_vel[:, joint_ids] * self.cfg.joint_vel_obs_scale
        one_step = torch.cat(
            (
                self.robot.data.root_ang_vel_b * self.cfg.base_ang_vel_obs_scale,
                self.robot.data.projected_gravity_b,
                joint_pos_rel,
                joint_vel,
                self.actions,
            ),
            dim=-1,
        )
        if one_step.shape[-1] != ONE_STEP_OBS_SIZE:
            raise RuntimeError(f"Observation contract mismatch: expected {ONE_STEP_OBS_SIZE}, got {one_step.shape[-1]}")
        return one_step

    def _get_rewards(self) -> torch.Tensor:
        groups = torch.stack(
            (
                self._reward_group_task(),
                self._reward_group_imitation(),
                self._reward_group_support(),
                self._reward_group_regularization(),
            ),
            dim=-1,
        )
        reward = groups @ self.reward_group_weights
        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

        success_step = self._success_step_mask()
        head_height = self._head_height_above_sole()
        base_height = self._base_height_above_sole()
        uprightness = self._uprightness()
        self.max_head_height = torch.maximum(self.max_head_height, head_height)
        self.max_base_height = torch.maximum(self.max_base_height, base_height)
        self.max_uprightness = torch.maximum(self.max_uprightness, uprightness)

        self.extras["log"] = {
            "reward/task": groups[:, 0].mean(),
            "reward/imitation": groups[:, 1].mean(),
            "reward/support": groups[:, 2].mean(),
            "reward/regularization": groups[:, 3].mean(),
            "reward/total": reward.mean(),
            "state/base_height": base_height.mean(),
            "state/head_height": head_height.mean(),
            "state/uprightness": uprightness.mean(),
            "state/success_step_rate": success_step.float().mean(),
            "state/success_hold_rate": self.success_ever.float().mean(),
            "state/nonfoot_contact_rate": self._nonfoot_contact_mask().float().mean(),
            "state/clean_support_rate": self._clean_support_mask().float().mean(),
            "state/max_head_height": self.max_head_height.mean(),
            "state/max_base_height": self.max_base_height.mean(),
            "state/max_uprightness": self.max_uprightness.mean(),
            "state/done_reason": self.last_done_reason.float().mean(),
        }
        self.last_joint_vel.copy_(self.robot.data.joint_vel)
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        success_step = self._success_step_mask()
        self.success_hold_counter = torch.where(
            success_step, self.success_hold_counter + 1, torch.zeros_like(self.success_hold_counter)
        )
        success = self.success_hold_counter >= self.success_hold_steps
        self.success_ever |= success

        joint_vel_bad = torch.max(torch.abs(self.robot.data.joint_vel), dim=1).values > self.cfg.dof_vel_limit
        root_vel_bad = torch.max(torch.abs(self.robot.data.root_vel_w), dim=1).values > self.cfg.base_vel_limit
        finite = torch.isfinite(self.robot.data.root_state_w).all(dim=1) & torch.isfinite(self.robot.data.joint_pos).all(dim=1)
        fallen_through = self.robot.data.root_pos_w[:, 2] < self.cfg.fallen_through_z
        terminated = joint_vel_bad | root_vel_bad | (~finite) | fallen_through

        episode_timeout = self.episode_length_buf >= self.max_episode_length - 1
        reference_timeout = self._reference_exhausted()
        time_out = success | reference_timeout | episode_timeout

        self.last_done_reason[:] = 0
        self.last_done_reason = torch.where(success, torch.full_like(self.last_done_reason, 1), self.last_done_reason)
        self.last_done_reason = torch.where(reference_timeout, torch.full_like(self.last_done_reason, 2), self.last_done_reason)
        self.last_done_reason = torch.where(episode_timeout, torch.full_like(self.last_done_reason, 3), self.last_done_reason)
        self.last_done_reason = torch.where(fallen_through, torch.full_like(self.last_done_reason, 4), self.last_done_reason)
        self.last_done_reason = torch.where(joint_vel_bad, torch.full_like(self.last_done_reason, 5), self.last_done_reason)
        self.last_done_reason = torch.where(root_vel_bad, torch.full_like(self.last_done_reason, 6), self.last_done_reason)
        self.last_done_reason = torch.where(~finite, torch.full_like(self.last_done_reason, 7), self.last_done_reason)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        super()._reset_idx(env_ids)

        count = env_ids.numel()
        use_motion = self._sample_motion_reset_mask(count)

        if use_motion.any():
            self._reset_from_motion(env_ids[use_motion])
        if (~use_motion).any():
            self._reset_synthetic(env_ids[~use_motion])

        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        self.joint_pos_target[env_ids] = self.robot.data.joint_pos[env_ids]
        self.last_joint_vel[env_ids] = self.robot.data.joint_vel[env_ids]
        self.obs_history[env_ids] = 0.0
        self.refresh_history[env_ids] = True
        self.success_hold_counter[env_ids] = 0
        self.success_ever[env_ids] = False
        self.max_head_height[env_ids] = 0.0
        self.max_base_height[env_ids] = 0.0
        self.max_uprightness[env_ids] = -1.0
        self.last_done_reason[env_ids] = 0

    def _sample_motion_reset_mask(self, count: int) -> torch.Tensor:
        if self.motion_bank is None or not self.cfg.use_motion_reference or self.cfg.reset_mode == "synthetic":
            return torch.zeros(count, dtype=torch.bool, device=self.device)
        if self.cfg.reset_mode == "motion":
            return torch.ones(count, dtype=torch.bool, device=self.device)
        return torch.rand(count, dtype=torch.float32, device=self.device) < self.cfg.motion_reset_probability

    def _reset_from_motion(self, env_ids: torch.Tensor) -> None:
        assert self.motion_bank is not None
        count = env_ids.numel()
        motion_ids, _, start_times = self.motion_bank.sample_starts(count)
        state = self.motion_bank.state_at(motion_ids, start_times)

        root_pose = torch.zeros(count, 7, dtype=torch.float32, device=self.device)
        root_pose[:, :3] = self.scene.env_origins[env_ids]
        xy_noise = torch.empty(count, 2, dtype=torch.float32, device=self.device).uniform_(
            -self.cfg.reset_xy_range, self.cfg.reset_xy_range
        )
        root_pose[:, :2] += xy_noise
        z_shift = (self.cfg.reference_root_min_z - state["base_pos_w"][:, 2]).clamp(min=0.0)
        root_pose[:, 2] = state["base_pos_w"][:, 2] + z_shift
        root_pose[:, 3:7] = state["base_quat_w"]
        root_vel = torch.zeros(count, 6, dtype=torch.float32, device=self.device)

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        joint_pos[:, self.policy_joint_ids] = state["joint_pos"]
        joint_vel[:, self.policy_joint_ids] = state["joint_vel"]
        self._apply_grouped_joint_noise(joint_pos, scale=0.5)
        joint_pos = self._clip_joint_pos(env_ids, joint_pos)

        self.robot.write_root_pose_to_sim(root_pose, env_ids)
        self.robot.write_root_velocity_to_sim(root_vel, env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.motion_ids[env_ids] = motion_ids
        self.motion_start_times[env_ids] = start_times
        self.motion_z_shift[env_ids] = z_shift
        self.motion_active[env_ids] = True

    def _reset_synthetic(self, env_ids: torch.Tensor) -> None:
        count = env_ids.numel()
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        root_vel = torch.empty(count, 6, dtype=torch.float32, device=self.device).uniform_(-0.03, 0.03)
        joint_policy = joint_pos[:, self.policy_joint_ids].clone()

        def set_rows(mask: torch.Tensor, indices: int | list[int], values) -> None:
            rows = torch.nonzero(mask, as_tuple=False).squeeze(-1)
            if rows.numel() == 0:
                return
            if isinstance(indices, int):
                joint_policy[rows, indices] = values
            else:
                value_tensor = values
                if not isinstance(value_tensor, torch.Tensor):
                    value_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
                joint_policy[rows[:, None], torch.tensor(indices, dtype=torch.long, device=self.device)] = value_tensor

        L_SHOULDER_PITCH = 0
        R_SHOULDER_PITCH = 1
        WAIST_PITCH = 2
        L_SHOULDER_ROLL = 3
        R_SHOULDER_ROLL = 4
        WAIST_ROLL = 5
        WAIST_YAW = 8
        L_ELBOW = 9
        R_ELBOW = 10
        L_HIP_PITCH = 11
        R_HIP_PITCH = 12
        L_HIP_ROLL = 15
        R_HIP_ROLL = 16
        L_KNEE = 23
        R_KNEE = 24
        L_ANKLE_PITCH = 25
        R_ANKLE_PITCH = 26

        posture_type = torch.randint(0, 5, (count,), dtype=torch.long, device=self.device)
        yaw = torch.empty(count, dtype=torch.float32, device=self.device).uniform_(-math.pi, math.pi)
        roll = torch.zeros_like(yaw)
        pitch = torch.zeros_like(yaw)
        z = torch.full_like(yaw, self.cfg.reference_root_min_z)

        supine = posture_type == 0
        pitch[supine] = math.pi / 2
        set_rows(supine, [L_HIP_PITCH, R_HIP_PITCH], [-1.15, -1.15])
        set_rows(supine, [L_KNEE, R_KNEE], [2.05, 2.05])
        set_rows(supine, [L_ANKLE_PITCH, R_ANKLE_PITCH], [-1.05, -1.05])
        set_rows(supine, [L_SHOULDER_PITCH, R_SHOULDER_PITCH], [-0.55, -0.55])
        set_rows(supine, [L_ELBOW, R_ELBOW], [1.25, 1.25])

        prone = posture_type == 1
        pitch[prone] = -math.pi / 2
        set_rows(prone, [L_HIP_PITCH, R_HIP_PITCH], [0.85, 0.85])
        set_rows(prone, [L_KNEE, R_KNEE], [1.45, 1.45])
        set_rows(prone, [L_ANKLE_PITCH, R_ANKLE_PITCH], [-0.55, -0.55])
        set_rows(prone, [L_SHOULDER_PITCH, R_SHOULDER_PITCH], [0.75, 0.75])
        set_rows(prone, [L_ELBOW, R_ELBOW], [1.15, 1.15])

        side_l = posture_type == 2
        roll[side_l] = math.pi / 2
        set_rows(side_l, L_HIP_ROLL, 0.55)
        set_rows(side_l, R_HIP_ROLL, -0.35)
        set_rows(side_l, [L_HIP_PITCH, R_HIP_PITCH], [-0.85, -0.85])
        set_rows(side_l, [L_KNEE, R_KNEE], [1.75, 1.75])
        set_rows(side_l, [L_SHOULDER_ROLL, R_SHOULDER_ROLL], [1.0, -0.35])
        set_rows(side_l, [L_ELBOW, R_ELBOW], [1.15, 1.15])

        side_r = posture_type == 3
        roll[side_r] = -math.pi / 2
        set_rows(side_r, L_HIP_ROLL, 0.35)
        set_rows(side_r, R_HIP_ROLL, -0.55)
        set_rows(side_r, [L_HIP_PITCH, R_HIP_PITCH], [-0.85, -0.85])
        set_rows(side_r, [L_KNEE, R_KNEE], [1.75, 1.75])
        set_rows(side_r, [L_SHOULDER_ROLL, R_SHOULDER_ROLL], [0.35, -1.0])
        set_rows(side_r, [L_ELBOW, R_ELBOW], [1.15, 1.15])

        twisted = posture_type == 4
        twisted_count = int(twisted.sum().item())
        if twisted_count:
            roll[twisted] = torch.empty(twisted_count, dtype=torch.float32, device=self.device).uniform_(-0.75, 0.75)
            pitch[twisted] = torch.empty(twisted_count, dtype=torch.float32, device=self.device).uniform_(-1.1, 1.1)
        set_rows(twisted, [L_HIP_PITCH, R_HIP_PITCH], [-0.95, -0.35])
        set_rows(twisted, [L_KNEE, R_KNEE], [1.8, 1.2])
        set_rows(twisted, [L_ANKLE_PITCH, R_ANKLE_PITCH], [-0.8, -0.35])
        set_rows(twisted, [L_SHOULDER_PITCH, R_SHOULDER_PITCH], [0.8, -0.25])
        set_rows(twisted, [L_ELBOW, R_ELBOW], [1.2, 0.8])
        set_rows(twisted, WAIST_ROLL, 0.35)
        set_rows(twisted, WAIST_YAW, 0.45)
        set_rows(twisted, WAIST_PITCH, 0.20)

        joint_pos[:, self.policy_joint_ids] = joint_policy
        self._apply_grouped_joint_noise(joint_pos, scale=1.0)
        joint_pos = self._clip_joint_pos(env_ids, joint_pos)

        root_pos = self.scene.env_origins[env_ids].clone()
        root_pos[:, 0] += torch.empty(count, dtype=torch.float32, device=self.device).uniform_(-0.1, 0.1)
        root_pos[:, 1] += torch.empty(count, dtype=torch.float32, device=self.device).uniform_(-0.1, 0.1)
        root_pos[:, 2] = z
        root_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

        self.robot.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), env_ids)
        self.robot.write_root_velocity_to_sim(root_vel, env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.motion_active[env_ids] = False
        self.motion_start_times[env_ids] = 0.0
        self.motion_z_shift[env_ids] = 0.0

    def _apply_grouped_joint_noise(self, joint_pos: torch.Tensor, *, scale: float) -> None:
        policy = joint_pos[:, self.policy_joint_ids]
        noise = torch.zeros_like(policy)
        noise[:, self.leg_policy_ids] = torch.empty(
            policy.shape[0], self.leg_policy_ids.numel(), dtype=torch.float32, device=self.device
        ).uniform_(-0.08, 0.08)
        noise[:, self.upper_policy_ids] = torch.empty(
            policy.shape[0], self.upper_policy_ids.numel(), dtype=torch.float32, device=self.device
        ).uniform_(-0.06, 0.06)
        noise[:, self.waist_policy_ids] = torch.empty(
            policy.shape[0], self.waist_policy_ids.numel(), dtype=torch.float32, device=self.device
        ).uniform_(-0.04, 0.04)
        noise[:, self.wrist_policy_ids] = torch.empty(
            policy.shape[0], self.wrist_policy_ids.numel(), dtype=torch.float32, device=self.device
        ).uniform_(-0.02, 0.02)
        joint_pos[:, self.policy_joint_ids] = policy + scale * noise

    def _clip_joint_pos(self, env_ids: torch.Tensor, joint_pos: torch.Tensor) -> torch.Tensor:
        limits = self.robot.data.soft_joint_pos_limits[env_ids]
        return torch.maximum(torch.minimum(joint_pos, limits[:, :, 1]), limits[:, :, 0])

    def _reward_group_task(self) -> torch.Tensor:
        uprightness = self._uprightness()
        head_height = self._head_height_above_sole()
        base_height = self._base_height_above_sole()
        success_step = self._success_step_mask().float()
        hold_shaping = torch.clamp(self.success_hold_counter.float() / float(self.success_hold_steps), 0.0, 1.0)
        return (
            2.0 * _tolerance(uprightness, lower=self.cfg.target_uprightness, upper=1.0, margin=1.0, value_at_margin=0.05)
            + 1.5
            * _tolerance(head_height, lower=self.cfg.target_head_height, upper=float("inf"), margin=0.9, value_at_margin=0.1)
            + 1.0
            * _tolerance(base_height, lower=self.cfg.target_base_height, upper=float("inf"), margin=0.5, value_at_margin=0.1)
            + 2.0 * success_step
            + 1.0 * hold_shaping
        )

    def _reward_group_imitation(self) -> torch.Tensor:
        if self.motion_bank is None:
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        ref = self._reference_state()
        active = self.motion_active.float()
        current_policy = self.robot.data.joint_pos[:, self.policy_joint_ids]
        ref_joint = ref["joint_pos"]

        ref_base_height = ref["base_pos_w"][:, 2] + self.motion_z_shift
        base_height_score = torch.exp(-6.0 * torch.square(self.robot.data.root_pos_w[:, 2] - ref_base_height))
        ref_proj_gravity = math_utils.quat_apply_inverse(
            ref["base_quat_w"], torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        )
        orientation_score = torch.exp(-3.0 * torch.sum(torch.square(self.robot.data.projected_gravity_b - ref_proj_gravity), dim=1))
        leg_score = torch.exp(
            -2.5
            * torch.mean(torch.square(current_policy[:, self.leg_policy_ids] - ref_joint[:, self.leg_policy_ids]), dim=1)
        )
        upper_score = torch.exp(
            -0.8
            * torch.mean(torch.square(current_policy[:, self.upper_policy_ids] - ref_joint[:, self.upper_policy_ids]), dim=1)
        )
        return active * (0.8 * base_height_score + 0.8 * orientation_score + 1.2 * leg_score + 0.4 * upper_score)

    def _reward_group_support(self) -> torch.Tensor:
        left_contact, right_contact = self._feet_contact_masks()
        both_feet = (left_contact & right_contact).float()
        nonfoot = self._nonfoot_contact_mask()
        gated_nonfoot = nonfoot & self._late_contact_gate()
        clean_support = (left_contact & right_contact & (~gated_nonfoot)).float()
        left_sole, right_sole = self._sole_centers()
        feet_distance = torch.linalg.norm(left_sole[:, :2] - right_sole[:, :2], dim=1)
        feet_distance_score = _tolerance(feet_distance, lower=0.12, upper=0.85, margin=0.35, value_at_margin=0.1)
        plane_score = torch.exp(-40.0 * self._sole_z_variance())
        foot_orientation = self._foot_uprightness_score()
        stable_xy = torch.exp(-4.0 * torch.sum(torch.square(self.robot.data.root_lin_vel_b[:, :2]), dim=1))
        stable_ang = torch.exp(-3.0 * torch.sum(torch.square(self.robot.data.root_ang_vel_b[:, :2]), dim=1))
        return (
            1.5 * both_feet
            + 2.0 * clean_support
            + 1.0 * plane_score
            + 1.0 * foot_orientation
            + 0.75 * feet_distance_score
            + 0.5 * stable_xy
            + 0.5 * stable_ang
            - 2.0 * gated_nonfoot.float()
        )

    def _reward_group_regularization(self) -> torch.Tensor:
        joint_vel = self.robot.data.joint_vel[:, self.policy_joint_ids]
        last_joint_vel = self.last_joint_vel[:, self.policy_joint_ids]
        joint_acc = (joint_vel - last_joint_vel) / self.step_dt
        torque = self.robot.data.applied_torque
        if torque is None:
            torque = torch.zeros_like(self.robot.data.joint_vel)
        torque = torque[:, self.policy_joint_ids]
        effort_limits = self.robot.data.joint_effort_limits[:, self.policy_joint_ids].clamp_min(1.0)
        vel_limits = self.robot.data.soft_joint_vel_limits[:, self.policy_joint_ids].clamp_min(1.0)
        lower = self.robot.data.soft_joint_pos_limits[:, self.policy_joint_ids, 0]
        upper = self.robot.data.soft_joint_pos_limits[:, self.policy_joint_ids, 1]
        q = self.robot.data.joint_pos[:, self.policy_joint_ids]
        pos_limit = (lower - q).clamp(min=0.0) + (q - upper).clamp(min=0.0)
        vel_limit = (torch.abs(joint_vel) / vel_limits - 0.9).clamp(min=0.0)

        return (
            -0.05 * torch.sum(torch.square(self.actions - self.last_actions), dim=1)
            -0.05 * torch.sum(torch.square(self.actions - 2.0 * self.last_actions + self.last_last_actions), dim=1)
            -0.25 * torch.mean(torch.square(torque / effort_limits), dim=1)
            -0.10 * torch.mean(torch.square(joint_vel / vel_limits), dim=1)
            -2.0e-7 * torch.sum(torch.square(joint_acc), dim=1)
            -5.0 * torch.sum(pos_limit, dim=1)
            -0.5 * torch.sum(vel_limit, dim=1)
        )

    def _reference_state(self) -> dict[str, torch.Tensor]:
        assert self.motion_bank is not None
        motion_times = self.motion_start_times + self.episode_length_buf.float() * self.step_dt
        return self.motion_bank.state_at(self.motion_ids, motion_times)

    def _reference_exhausted(self) -> torch.Tensor:
        if self.motion_bank is None:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        ref = self._reference_state()
        grace = int(round(self.cfg.reference_exhaustion_grace_s / self.step_dt))
        past_grace = self.episode_length_buf > grace
        return self.motion_active & ref["exhausted"] & past_grace & (~self.success_ever)

    def _body_pos(self, body_id: int) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, body_id, :]

    def _body_quat(self, body_id: int) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, body_id, :]

    def _uprightness(self) -> torch.Tensor:
        return -self.robot.data.projected_gravity_b[:, 2]

    def _foot_sole_points(self, body_id: int) -> torch.Tensor:
        quat = self._body_quat(body_id)
        pos = self._body_pos(body_id)
        count, num_points, _ = self.num_envs, self.sole_offsets.shape[0], self.sole_offsets.shape[1]
        offsets = self.sole_offsets.unsqueeze(0).expand(count, -1, -1)
        world_offsets = math_utils.quat_apply(
            quat.unsqueeze(1).expand(-1, num_points, -1).reshape(-1, 4),
            offsets.reshape(-1, 3),
        ).reshape(count, num_points, 3)
        return pos.unsqueeze(1) + world_offsets

    def _sole_points(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._foot_sole_points(self.left_foot_body_id), self._foot_sole_points(self.right_foot_body_id)

    def _sole_centers(self) -> tuple[torch.Tensor, torch.Tensor]:
        left_points, right_points = self._sole_points()
        return left_points.mean(dim=1), right_points.mean(dim=1)

    def _mean_sole_z(self) -> torch.Tensor:
        left_sole, right_sole = self._sole_centers()
        return 0.5 * (left_sole[:, 2] + right_sole[:, 2])

    def _head_height_above_sole(self) -> torch.Tensor:
        return self._body_pos(self.head_body_id)[:, 2] - self._mean_sole_z()

    def _base_height_above_sole(self) -> torch.Tensor:
        return self.robot.data.root_pos_w[:, 2] - self._mean_sole_z()

    def _sole_z_variance(self) -> torch.Tensor:
        left_points, right_points = self._sole_points()
        return 0.5 * (left_points[:, :, 2].var(dim=1, unbiased=False) + right_points[:, :, 2].var(dim=1, unbiased=False))

    def _foot_uprightness_score(self) -> torch.Tensor:
        world_z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=self.device).repeat(self.num_envs, 1)
        left_z = math_utils.quat_apply(self._body_quat(self.left_foot_body_id), world_z)[:, 2]
        right_z = math_utils.quat_apply(self._body_quat(self.right_foot_body_id), world_z)[:, 2]
        return torch.clamp(0.5 * (left_z + right_z), 0.0, 1.0)

    def _contact_force_norms(self, ids: list[int]) -> torch.Tensor:
        forces = self.contact_sensor.data.net_forces_w
        return torch.linalg.norm(forces[:, ids, :], dim=-1)

    def _feet_contact_masks(self) -> tuple[torch.Tensor, torch.Tensor]:
        force_threshold = self.cfg.contact_force_threshold
        left_force = self._contact_force_norms(self.contact_left_foot_ids).amax(dim=1) > force_threshold
        right_force = self._contact_force_norms(self.contact_right_foot_ids).amax(dim=1) > force_threshold
        left_sole, right_sole = self._sole_centers()
        left_height = left_sole[:, 2] < 0.035
        right_height = right_sole[:, 2] < 0.035
        return left_force | left_height, right_force | right_height

    def _nonfoot_contact_mask(self) -> torch.Tensor:
        if not self.contact_nonfoot_ids:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        force_contact = self._contact_force_norms(self.contact_nonfoot_ids).amax(dim=1) > self.cfg.contact_force_threshold
        z_contact = self.robot.data.body_pos_w[:, self.contact_nonfoot_ids, 2].amin(dim=1) < 0.035
        return force_contact | z_contact

    def _late_contact_gate(self) -> torch.Tensor:
        gate_steps = int(round(self.cfg.nonfoot_contact_penalty_gate_s / self.step_dt))
        return (self.episode_length_buf >= gate_steps) | (
            (self._base_height_above_sole() > self.cfg.clean_support_base_height_gate)
            & (self._uprightness() > self.cfg.clean_support_upright_gate)
        )

    def _clean_support_mask(self) -> torch.Tensor:
        left_contact, right_contact = self._feet_contact_masks()
        return left_contact & right_contact & (~(self._nonfoot_contact_mask() & self._late_contact_gate()))

    def _success_step_mask(self) -> torch.Tensor:
        left_contact, right_contact = self._feet_contact_masks()
        leg_error = torch.mean(
            torch.abs(self.robot.data.joint_pos[:, self.policy_joint_ids][:, self.leg_policy_ids] - self.leg_target_policy),
            dim=1,
        )
        nonfoot = self._nonfoot_contact_mask() & self._late_contact_gate()
        root_lin = self.robot.data.root_lin_vel_b
        root_ang = self.robot.data.root_ang_vel_b
        return (
            (self._base_height_above_sole() >= self.cfg.target_base_height)
            & (self._head_height_above_sole() >= self.cfg.target_head_height)
            & (self._uprightness() >= self.cfg.target_uprightness)
            & (torch.linalg.norm(root_lin[:, :2], dim=1) <= SUCCESS_THRESHOLDS["base_lin_xy"])
            & (torch.abs(root_lin[:, 2]) <= SUCCESS_THRESHOLDS["base_lin_z"])
            & (torch.linalg.norm(root_ang[:, :2], dim=1) <= SUCCESS_THRESHOLDS["base_ang_xy"])
            & (leg_error <= SUCCESS_THRESHOLDS["leg_joint_error"])
            & left_contact
            & right_contact
            & (~nonfoot)
        )


def _tolerance(
    x: torch.Tensor,
    *,
    lower: float,
    upper: float,
    margin: float,
    value_at_margin: float,
) -> torch.Tensor:
    in_bounds = (x >= lower) & (x <= upper)
    if margin <= 0.0:
        return in_bounds.float()
    distance = torch.where(x < lower, lower - x, torch.zeros_like(x))
    distance = torch.where(x > upper, x - upper, distance)
    scale = math.sqrt(-2.0 * math.log(value_at_margin))
    value = torch.exp(-0.5 * torch.square(distance / margin * scale))
    return torch.where(in_bounds, torch.ones_like(value), value)
