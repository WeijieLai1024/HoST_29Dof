from __future__ import annotations

import argparse
import os
import subprocess
import sys
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


def _build_play_train_cfg(rl_device: str) -> dict:
    return {
        "seed": 1,
        "device": rl_device,
        "num_steps_per_env": 1,
        "max_iterations": 1,
        "empirical_normalization": None,
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
        "clip_actions": 1.0,
        "save_interval": 1,
        "experiment_name": "g1_ground",
        "run_name": "play",
        "logger": "tensorboard",
        "neptune_project": "legged_gym",
        "wandb_project": "legged_gym",
        "resume": True,
        "load_run": ".*",
        "load_checkpoint": "model_.*.pt",
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 0.8,
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 1,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }


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
        print(f"[play] auto-selected device={selected} free_mib={best_free_mib}", flush=True)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a HoST G1 stand-up checkpoint on Isaac Sim / Isaac Lab.")
    parser.add_argument("--task", type=str, default="g1_ground", help="Task name. Currently migrated: g1_ground.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model_*.pt checkpoint.")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to rollout.")
    parser.add_argument("--num_steps", type=int, default=2000, help="Number of policy steps to rollout.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--rl_device", type=str, default=None, help="RL device. Defaults to --device.")
    parser.add_argument(
        "--motion_dir",
        type=str,
        default="/home/weijielai/Datasets/xloco_standup",
        help="Directory containing stand-up NPZ motions and selection YAML.",
    )
    parser.add_argument(
        "--motion_selection",
        type=str,
        default="standup_motion_selection.yaml",
        help="Motion selection YAML, relative to --motion_dir unless absolute.",
    )
    parser.add_argument(
        "--disable_motion_reference",
        action="store_true",
        default=False,
        help="Disable motion reset/reference shaping during rollout.",
    )
    parser.add_argument(
        "--reset_mode",
        type=str,
        default="mixed",
        choices=("mixed", "motion", "synthetic"),
        help="Reset source for rollout.",
    )
    parser.add_argument(
        "--success_hold_time",
        type=float,
        default=1.0,
        help="Seconds that success conditions must hold before episode success.",
    )
    parser.add_argument("--eval_episodes", type=int, default=100, help="Number of episodes to target for eval summary.")

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
    args = parse_args()
    if args.task != "g1_ground":
        raise SystemExit(
            f"Task {args.task!r} has not been migrated yet. Use --task g1_ground for the IsaacLab G1 shoe backend."
        )
    args.device = _auto_select_device(args.device)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    env = None
    wrapped_env = None

    try:
        import torch
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        from legged_gym.isaaclab_envs.g1_shoe_host_env import HostG1ShoeDirectEnv, HostG1ShoeDirectEnvCfg

        cfg = HostG1ShoeDirectEnvCfg()
        cfg.scene.num_envs = int(args.num_envs)
        cfg.seed = int(args.seed)
        cfg.sim.device = args.device
        cfg.pull_force_enabled = False
        cfg.motion_dir = args.motion_dir
        cfg.motion_selection = args.motion_selection
        cfg.use_motion_reference = not args.disable_motion_reference
        cfg.reset_mode = args.reset_mode
        cfg.success_hold_time = float(args.success_hold_time)

        env = HostG1ShoeDirectEnv(cfg)
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=cfg.clip_actions)
        rl_device = args.rl_device or args.device
        runner = OnPolicyRunner(wrapped_env, _build_play_train_cfg(rl_device), log_dir=None, device=rl_device)
        runner.load(args.checkpoint_path, load_optimizer=False, map_location=rl_device)
        policy = runner.get_inference_policy(device=rl_device)

        obs = wrapped_env.get_observations().to(rl_device)
        total_reward = torch.zeros(args.num_envs, dtype=torch.float32, device=rl_device)
        success_episodes = 0
        finished_episodes = 0
        max_head = 0.0
        max_base = 0.0
        max_upright = -1.0
        nonfoot_contact_accum = 0.0
        clean_support_accum = 0.0
        log_count = 0
        for step in range(args.num_steps):
            with torch.inference_mode():
                actions = policy(obs)
            obs, rewards, dones, extras = wrapped_env.step(actions.to(wrapped_env.device))
            obs = obs.to(rl_device)
            total_reward += rewards.to(rl_device)
            logs = extras.get("log", {})
            if logs:
                max_head = max(max_head, float(logs.get("state/max_head_height", 0.0)))
                max_base = max(max_base, float(logs.get("state/max_base_height", 0.0)))
                max_upright = max(max_upright, float(logs.get("state/max_uprightness", -1.0)))
                nonfoot_contact_accum += float(logs.get("state/nonfoot_contact_rate", 0.0))
                clean_support_accum += float(logs.get("state/clean_support_rate", 0.0))
                log_count += 1
            if step % 200 == 0:
                success = logs.get("state/success_hold_rate", None)
                success_msg = f" success={float(success):.3f}" if success is not None else ""
                print(
                    f"[play] step={step} mean_reward={float(rewards.mean()):.3f}{success_msg} "
                    f"max_head={max_head:.3f} max_base={max_base:.3f} max_upright={max_upright:.3f}",
                    flush=True,
                )
            done_count = int(dones.sum().item())
            if done_count > 0:
                finished_episodes += done_count
                if logs:
                    success_episodes += int(round(float(logs.get("state/success_hold_rate", 0.0)) * done_count))
                total_reward.zero_()
            if finished_episodes >= args.eval_episodes:
                break
        denom = max(1, log_count)
        print(
            f"[play] finished steps={step + 1} finished_episodes={finished_episodes} "
            f"success_episodes~={success_episodes} max_head={max_head:.3f} max_base={max_base:.3f} "
            f"max_upright={max_upright:.3f} nonfoot_contact={nonfoot_contact_accum / denom:.3f} "
            f"clean_support={clean_support_accum / denom:.3f}",
            flush=True,
        )
    finally:
        if wrapped_env is not None:
            wrapped_env.close()
        elif env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
