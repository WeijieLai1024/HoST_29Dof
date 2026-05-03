from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _append_kit_args(args: argparse.Namespace, extra: str) -> None:
    extra = extra.strip()
    if not extra:
        return
    current = str(getattr(args, "kit_args", "") or "")
    setattr(args, "kit_args", extra if not current else f"{current} {extra}")


def _ensure_default_kit_log_level(args: argparse.Namespace) -> None:
    current = str(getattr(args, "kit_args", "") or "")
    if "--/log/level" in current or "-/log/level" in current:
        return
    if any(arg.startswith("--/log/level") or arg.startswith("-/log/level") for arg in sys.argv[1:]):
        return
    _append_kit_args(args, "--/log/level=error")


def _arg_was_passed(flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:])


def _auto_select_device(current_device: str) -> str:
    if _arg_was_passed("--device") or _arg_was_passed("--cpu"):
        return current_device
    if current_device not in ("cuda", "cuda:0"):
        return current_device
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return current_device

    best_index = None
    best_free_mib = -1
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            index = int(parts[0])
            free_mib = int(parts[1])
        except ValueError:
            continue
        if free_mib > best_free_mib:
            best_index = index
            best_free_mib = free_mib
    if best_index is None:
        return current_device
    selected = f"cuda:{best_index}"
    if selected != current_device:
        print(f"[probe] auto-selected device={selected} free_mib={best_free_mib}", flush=True)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runtime probe for the HoST G1 29-DoF shoe backend.")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--probe_steps", type=int, default=200)
    parser.add_argument("--settle_steps", type=int, default=20)
    parser.add_argument("--geometry_margin", type=float, default=0.0)
    parser.add_argument("--no_geometry_assert", action="store_true", default=False)
    parser.add_argument("--motion_dir", type=str, default="/home/weijielai/Datasets/xloco_standup")
    parser.add_argument("--motion_selection", type=str, default="standup_motion_selection.yaml")
    parser.add_argument("--disable_motion_reference", action="store_true", default=False)
    parser.add_argument("--reset_mode", type=str, default="mixed", choices=("mixed", "motion", "synthetic"))
    parser.add_argument("--success_hold_time", type=float, default=1.0)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args, unknown = parser.parse_known_args()
    kit_overrides = [item for item in unknown if item.startswith("--/") or item.startswith("-/")]
    other_unknown = [item for item in unknown if item not in kit_overrides]
    if other_unknown:
        parser.error(f"unrecognized arguments: {' '.join(other_unknown)}")
    if kit_overrides:
        _append_kit_args(args, " ".join(kit_overrides))
    _ensure_default_kit_log_level(args)
    return args


def _fail(message: str) -> None:
    raise RuntimeError(f"[probe] contract failure: {message}")


def _write_standing_pose(env, torch) -> None:
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    root_pose = torch.zeros(env.num_envs, 7, dtype=torch.float32, device=env.device)
    root_pose[:, :3] = env.scene.env_origins
    root_pose[:, 2] = float(env.cfg.robot_cfg.init_state.pos[2])
    root_pose[:, 3] = 1.0
    root_vel = torch.zeros(env.num_envs, 6, dtype=torch.float32, device=env.device)
    joint_pos = env.robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)
    root_pose, root_vel, joint_pos, joint_vel = env._preflight_reset_state(env_ids, root_pose, root_vel, joint_pos, joint_vel)
    env.robot.write_root_pose_to_sim(root_pose, env_ids)
    env.robot.write_root_velocity_to_sim(root_vel, env_ids)
    env.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
    env.actions.zero_()
    env.last_actions.zero_()
    env.last_last_actions.zero_()
    env.joint_pos_target.copy_(env.robot.data.default_joint_pos)
    env.episode_length_buf[:] = env.cfg.unactuated_steps
    env.success_hold_counter.zero_()
    env.success_ever.zero_()


def _reward_groups(env, torch) -> torch.Tensor:
    return torch.stack(
        (
            env._reward_group_task(),
            env._reward_group_imitation(),
            env._reward_group_support(),
            env._reward_group_regularization(),
        ),
        dim=-1,
    )


def main() -> None:
    os.environ.setdefault("TORCH_CUDNN_V8_API_DISABLED", "1")
    os.environ.setdefault("HOST_G1_DEBUG_SETUP", "1")
    args = parse_args()
    args.device = _auto_select_device(args.device)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    env = None

    try:
        import torch

        from legged_gym.isaaclab_envs.g1_shoe_constants import XLoco_G1_29DOF_JOINT_NAMES
        from legged_gym.isaaclab_envs.g1_shoe_host_env import HostG1ShoeDirectEnv, HostG1ShoeDirectEnvCfg

        cfg = HostG1ShoeDirectEnvCfg()
        cfg.scene.num_envs = int(args.num_envs)
        cfg.sim.device = args.device
        cfg.motion_dir = args.motion_dir
        cfg.motion_selection = args.motion_selection
        cfg.use_motion_reference = not args.disable_motion_reference
        cfg.reset_mode = args.reset_mode
        cfg.success_hold_time = float(args.success_hold_time)
        cfg.pull_force_enabled = False

        print("[probe] creating env", flush=True)
        env = HostG1ShoeDirectEnv(cfg)
        print("[probe] env created", flush=True)
        print("[probe] resetting env", flush=True)
        env.reset()
        print("[probe] env reset complete", flush=True)

        print("[probe] resolved ids", flush=True)
        print(f"  base_body_id: {env.base_body_id}", flush=True)
        print(f"  head_body_id: {env.head_body_id}", flush=True)
        print(f"  left_foot_body_id: {env.left_foot_body_id}", flush=True)
        print(f"  right_foot_body_id: {env.right_foot_body_id}", flush=True)
        print(f"  contact_left_foot_ids: {env.contact_left_foot_ids}", flush=True)
        print(f"  contact_right_foot_ids: {env.contact_right_foot_ids}", flush=True)
        print(f"  contact_nonfoot_count: {len(env.contact_nonfoot_ids)}", flush=True)
        print(f"  policy_joint_ids[0:5]: {env.policy_joint_ids[:5]}", flush=True)

        print("[probe] standing geometry calibration", flush=True)
        _write_standing_pose(env, torch)
        zero = torch.zeros(env.num_envs, env.cfg.action_space, dtype=torch.float32, device=env.device)
        for _ in range(args.settle_steps):
            obs, rew, terminated, truncated, extras = env.step(zero)
            if bool((terminated | truncated).any()):
                _write_standing_pose(env, torch)
        base_height = env._base_height_above_sole()
        head_height = env._head_height_above_sole()
        uprightness = env._uprightness()
        sole_z = env._mean_sole_z()
        left_contact, right_contact = env._feet_contact_masks()
        nonfoot = env._nonfoot_contact_mask()
        print(f"  sole_z_mean: {float(sole_z.mean()):.4f}", flush=True)
        print(f"  base_height_mean: {float(base_height.mean()):.4f}", flush=True)
        print(f"  head_height_mean: {float(head_height.mean()):.4f}", flush=True)
        print(f"  uprightness_mean: {float(uprightness.mean()):.4f}", flush=True)
        print(f"  feet_contact_rate: {float((left_contact & right_contact).float().mean()):.4f}", flush=True)
        print(f"  nonfoot_contact_rate: {float(nonfoot.float().mean()):.4f}", flush=True)
        if not args.no_geometry_assert:
            margin = float(args.geometry_margin)
            if float(base_height.min()) + 1.0e-6 < env.cfg.target_base_height + margin:
                print(f"  recommended_base_height_threshold: {float(base_height.min() - margin):.4f}", flush=True)
                _fail("standing base height does not reach success threshold")
            if float(head_height.min()) + 1.0e-6 < env.cfg.target_head_height + margin:
                print(f"  recommended_head_height_threshold: {float(head_height.min() - margin):.4f}", flush=True)
                _fail("standing head height does not reach success threshold")
            if float(uprightness.min()) + 1.0e-6 < env.cfg.target_uprightness + margin:
                print(f"  recommended_uprightness_threshold: {float(uprightness.min() - margin):.4f}", flush=True)
                _fail("standing uprightness does not reach success threshold")

        env.episode_length_buf[:] = env.cfg.unactuated_steps
        action = torch.zeros(env.num_envs, env.cfg.action_space, dtype=torch.float32, device=env.device)
        action[:, 0] = 1.0
        env._pre_physics_step(action)
        target_delta = env.joint_pos_target[:, env.policy_joint_ids] - env.robot.data.default_joint_pos[:, env.policy_joint_ids]
        nonzero = torch.nonzero(torch.abs(target_delta[0]) > 1.0e-6, as_tuple=False).squeeze(-1).tolist()
        nonzero_names = [XLoco_G1_29DOF_JOINT_NAMES[idx] for idx in nonzero]
        print("[probe] single-action semantic test", flush=True)
        print(f"  action[0] target_nonzero: {nonzero_names}", flush=True)
        print(f"  action[0] target_delta: {float(target_delta[0, 0]):.6f}", flush=True)
        if nonzero_names != ["left_shoulder_pitch_joint"]:
            _fail(f"action[0] drove unexpected joints: {nonzero_names}")

        groups = _reward_groups(env, torch)
        if not torch.isfinite(groups).all().item():
            _fail(f"reward groups contain non-finite values: {groups}")
        print("[probe] reward finite test", flush=True)
        print(f"  reward_group_means: {[round(float(x), 4) for x in groups.mean(dim=0)]}", flush=True)

        finite = True
        done_count = 0
        for _ in range(args.probe_steps):
            obs, rew, terminated, truncated, extras = env.step(zero)
            finite = finite and torch.isfinite(obs["policy"]).all().item() and torch.isfinite(rew).all().item()
            done_count += int((terminated | truncated).sum().item())
            if bool((terminated | truncated).all()):
                env.reset()
        logs = extras.get("log", {})
        if not finite:
            _fail("zero-action rollout produced non-finite obs or rewards")
        print("[probe] zero-action rollout", flush=True)
        print(f"  steps: {args.probe_steps}", flush=True)
        print(f"  finite: {finite}", flush=True)
        print(f"  done_count: {done_count}", flush=True)
        print(f"  reward_mean: {float(rew.mean()):.4f}", flush=True)
        print(f"  max_head_height: {float(logs.get('state/max_head_height', 0.0)):.4f}", flush=True)
        print(f"  max_base_height: {float(logs.get('state/max_base_height', 0.0)):.4f}", flush=True)
        print(f"  max_uprightness: {float(logs.get('state/max_uprightness', -1.0)):.4f}", flush=True)
        print(f"  clean_support_rate: {float(logs.get('state/clean_support_rate', 0.0)):.4f}", flush=True)
        print(f"  nonfoot_contact_rate: {float(logs.get('state/nonfoot_contact_rate', 0.0)):.4f}", flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
