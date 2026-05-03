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

        print("[probe] standing/reset geometry")
        print(f"  base_height_mean: {float(env._base_height_above_sole().mean()):.4f}")
        print(f"  head_height_mean: {float(env._head_height_above_sole().mean()):.4f}")
        print(f"  uprightness_mean: {float(env._uprightness().mean()):.4f}")
        left_contact, right_contact = env._feet_contact_masks()
        print(f"  feet_contact_rate: {float((left_contact & right_contact).float().mean()):.4f}")
        print(f"  nonfoot_contact_rate: {float(env._nonfoot_contact_mask().float().mean()):.4f}")

        env.episode_length_buf[:] = env.cfg.unactuated_steps
        action = torch.zeros(env.num_envs, env.cfg.action_space, dtype=torch.float32, device=env.device)
        action[:, 0] = 1.0
        env._pre_physics_step(action)
        target_delta = env.joint_pos_target[:, env.policy_joint_ids] - env.robot.data.default_joint_pos[:, env.policy_joint_ids]
        nonzero = torch.nonzero(torch.abs(target_delta[0]) > 1.0e-6, as_tuple=False).squeeze(-1).tolist()
        nonzero_names = [XLoco_G1_29DOF_JOINT_NAMES[idx] for idx in nonzero]
        print("[probe] single-action semantic test")
        print(f"  action[0] target_nonzero: {nonzero_names}")
        print(f"  action[0] target_delta: {float(target_delta[0, 0]):.6f}")

        zero = torch.zeros(env.num_envs, env.cfg.action_space, dtype=torch.float32, device=env.device)
        finite = True
        for _ in range(args.probe_steps):
            obs, rew, terminated, truncated, extras = env.step(zero)
            finite = finite and torch.isfinite(obs["policy"]).all().item() and torch.isfinite(rew).all().item()
            if bool((terminated | truncated).all()):
                env.reset()
        logs = extras.get("log", {})
        print("[probe] zero-action rollout")
        print(f"  steps: {args.probe_steps}")
        print(f"  finite: {finite}")
        print(f"  reward_mean: {float(rew.mean()):.4f}")
        print(f"  max_head_height: {float(logs.get('state/max_head_height', 0.0)):.4f}")
        print(f"  max_base_height: {float(logs.get('state/max_base_height', 0.0)):.4f}")
        print(f"  max_uprightness: {float(logs.get('state/max_uprightness', -1.0)):.4f}")
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
