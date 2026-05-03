from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
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


def _build_train_cfg(args: argparse.Namespace, rl_device: str) -> dict:
    return {
        "seed": args.seed,
        "device": rl_device,
        "num_steps_per_env": args.num_steps_per_env,
        "max_iterations": args.max_iterations,
        "empirical_normalization": None,
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
        "clip_actions": 1.0,
        "save_interval": args.save_interval,
        "experiment_name": args.experiment_name,
        "run_name": args.run_name,
        "logger": "tensorboard",
        "neptune_project": "legged_gym",
        "wandb_project": "legged_gym",
        "resume": args.resume,
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
            "num_mini_batches": 4,
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
        print(f"[train] auto-selected device={selected} free_mib={best_free_mib}", flush=True)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HoST G1 stand-up on Isaac Sim / Isaac Lab.")
    parser.add_argument("--task", type=str, default="g1_ground", help="Task name. Currently migrated: g1_ground.")
    parser.add_argument("--num_envs", type=int, default=4096, help="Number of parallel environments.")
    parser.add_argument("--max_iterations", type=int, default=12000, help="Number of PPO updates.")
    parser.add_argument("--num_steps_per_env", type=int, default=50, help="Rollout steps per environment per update.")
    parser.add_argument("--save_interval", type=int, default=500, help="Model save interval in iterations.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--run_name", type=str, default="", help="Optional run-name suffix.")
    parser.add_argument("--experiment_name", type=str, default="g1_ground", help="RSL-RL experiment name.")
    parser.add_argument("--log_root", type=str, default=None, help="Override log root directory.")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from --checkpoint_path.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Checkpoint path for resume.")
    parser.add_argument("--rl_device", type=str, default=None, help="RL device. Defaults to --device.")
    parser.add_argument("--init_at_random_ep_len", action="store_true", default=True)
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
        help="Disable motion reset/reference shaping and use procedural synthetic resets only.",
    )
    parser.add_argument(
        "--reset_mode",
        type=str,
        default="mixed",
        choices=("mixed", "motion", "synthetic"),
        help="Reset source for stand-up training.",
    )
    parser.add_argument(
        "--success_hold_time",
        type=float,
        default=1.0,
        help="Seconds that success conditions must hold before episode success.",
    )

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

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False

        cfg = HostG1ShoeDirectEnvCfg()
        cfg.scene.num_envs = int(args.num_envs)
        cfg.seed = int(args.seed)
        cfg.sim.device = args.device
        cfg.motion_dir = args.motion_dir
        cfg.motion_selection = args.motion_selection
        cfg.use_motion_reference = not args.disable_motion_reference
        cfg.reset_mode = args.reset_mode
        cfg.success_hold_time = float(args.success_hold_time)

        env = HostG1ShoeDirectEnv(cfg)
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=cfg.clip_actions)

        rl_device = args.rl_device or args.device
        train_cfg = _build_train_cfg(args, rl_device)
        log_root = Path(args.log_root) if args.log_root else REPO_ROOT / "logs" / "rsl_rl" / args.experiment_name
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.run_name:
            run_name = f"{run_name}_{args.run_name}"
        log_dir = log_root / run_name

        runner = OnPolicyRunner(wrapped_env, train_cfg, log_dir=str(log_dir), device=rl_device)
        if args.resume:
            if not args.checkpoint_path:
                raise SystemExit("--resume requires --checkpoint_path")
            runner.load(args.checkpoint_path, load_optimizer=True, map_location=rl_device)

        print(f"[train] task=g1_ground num_envs={args.num_envs} max_iterations={args.max_iterations}", flush=True)
        print(
            "[train] "
            f"motion_reference={cfg.use_motion_reference} reset_mode={cfg.reset_mode} "
            f"motion_dir={cfg.motion_dir} selection={cfg.motion_selection}",
            flush=True,
        )
        print(f"[train] log_dir={log_dir}", flush=True)
        runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=args.init_at_random_ep_len)
    finally:
        if wrapped_env is not None:
            wrapped_env.close()
        elif env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
