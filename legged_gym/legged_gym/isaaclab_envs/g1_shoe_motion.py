"""Lightweight stand-up motion bank for the HoST G1 shoe backend.

The motion bank intentionally stays independent from Isaac Lab so static
validation can inspect the dataset before the simulator is launched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from legged_gym.isaaclab_envs.g1_shoe_constants import (
    NUM_ACTIONS,
    URDF_G1_29DOF_JOINT_NAMES,
    XLoco_G1_29DOF_JOINT_NAMES,
)


REQUIRED_NPZ_KEYS = ("framerate", "joint_names", "joint_pos", "base_pos_w", "base_quat_w")


@dataclass(frozen=True)
class MotionSchemaSummary:
    motion_dir: Path
    selection_path: Path
    num_selected: int
    joint_order: list[str]
    policy_to_dataset_joint_order: list[int]
    min_frames: int
    max_frames: int
    min_duration_s: float
    max_duration_s: float
    low_start_min: int
    low_start_max: int
    low_start_total: int
    low_start_zero_files: int


@dataclass
class _MotionRecord:
    path: Path
    framerate: float
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    base_pos_w: torch.Tensor
    base_quat_w: torch.Tensor
    low_start_frames: torch.Tensor

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def duration_s(self) -> float:
        return max(0.0, (self.num_frames - 1) / self.framerate)


def selected_motion_paths(motion_dir: str | Path, selection: str | Path) -> tuple[Path, list[Path]]:
    """Resolve selected NPZ files from the stand-up YAML selection."""

    motion_dir = Path(motion_dir)
    selection_path = Path(selection)
    if not selection_path.is_absolute():
        selection_path = motion_dir / selection_path
    if not selection_path.is_file():
        raise FileNotFoundError(f"Missing motion selection YAML: {selection_path}")

    data = yaml.safe_load(selection_path.read_text()) or {}
    selected = data.get("selected_files")
    if not isinstance(selected, list) or not selected:
        raise ValueError(f"Motion selection must contain a non-empty selected_files list: {selection_path}")

    paths = []
    for item in selected:
        path = Path(item)
        if not path.is_absolute():
            path = motion_dir / path
        if not path.is_file():
            raise FileNotFoundError(f"Selected motion file does not exist: {path}")
        paths.append(path)
    return selection_path, paths


def validate_motion_dataset(
    motion_dir: str | Path,
    selection: str | Path,
    policy_joint_names: list[str] | tuple[str, ...] = XLoco_G1_29DOF_JOINT_NAMES,
    *,
    low_start_base_z: float = 0.55,
    low_start_uprightness: float = 0.50,
) -> MotionSchemaSummary:
    """Validate selection YAML and NPZ schema, returning remap metadata."""

    selection_path, paths = selected_motion_paths(motion_dir, selection)
    first_joint_order: list[str] | None = None
    min_frames = 10**9
    max_frames = 0
    min_duration_s = float("inf")
    max_duration_s = 0.0
    low_start_min = 10**9
    low_start_max = 0
    low_start_total = 0
    low_start_zero_files = 0

    for path in paths:
        with np.load(path, allow_pickle=True) as data:
            missing = [key for key in REQUIRED_NPZ_KEYS if key not in data.files]
            if missing:
                raise ValueError(f"{path} missing NPZ keys: {missing}")
            joint_names = [str(name) for name in data["joint_names"].tolist()]
            if len(joint_names) != NUM_ACTIONS:
                raise ValueError(f"{path} expected {NUM_ACTIONS} joint names, got {len(joint_names)}")
            if set(joint_names) != set(policy_joint_names):
                missing_joints = sorted(set(policy_joint_names) - set(joint_names))
                extra_joints = sorted(set(joint_names) - set(policy_joint_names))
                raise ValueError(f"{path} joint set mismatch; missing={missing_joints}, extra={extra_joints}")
            if first_joint_order is None:
                first_joint_order = joint_names
            elif joint_names != first_joint_order:
                raise ValueError(f"{path} joint order differs from first selected motion")

            joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
            base_pos = np.asarray(data["base_pos_w"], dtype=np.float32)
            base_quat = np.asarray(data["base_quat_w"], dtype=np.float32)
            framerate = float(np.asarray(data["framerate"]).reshape(-1)[0])
            if joint_pos.ndim != 2 or joint_pos.shape[1] != NUM_ACTIONS:
                raise ValueError(f"{path} joint_pos must have shape (T, {NUM_ACTIONS}), got {joint_pos.shape}")
            if base_pos.shape != (joint_pos.shape[0], 3):
                raise ValueError(f"{path} base_pos_w shape mismatch: {base_pos.shape}")
            if base_quat.shape != (joint_pos.shape[0], 4):
                raise ValueError(f"{path} base_quat_w shape mismatch: {base_quat.shape}")
            if framerate <= 0.0:
                raise ValueError(f"{path} invalid framerate: {framerate}")
            if not np.isfinite(joint_pos).all():
                raise ValueError(f"{path} joint_pos contains non-finite values")
            if not np.isfinite(base_pos).all():
                raise ValueError(f"{path} base_pos_w contains non-finite values")
            if not np.isfinite(base_quat).all():
                raise ValueError(f"{path} base_quat_w contains non-finite values")
            quat_norm = np.linalg.norm(base_quat, axis=-1)
            if np.min(quat_norm) < 0.5 or np.max(quat_norm) > 1.5:
                raise ValueError(f"{path} base_quat_w norms look invalid: min={quat_norm.min()}, max={quat_norm.max()}")
            uprightness = _uprightness_wxyz_np(_normalize_quat_np(base_quat))
            low_start_count = int(((base_pos[:, 2] <= low_start_base_z) | (uprightness <= low_start_uprightness)).sum())
            if low_start_count == 0:
                low_start_zero_files += 1
            duration_s = max(0.0, (joint_pos.shape[0] - 1) / framerate)
            min_frames = min(min_frames, joint_pos.shape[0])
            max_frames = max(max_frames, joint_pos.shape[0])
            min_duration_s = min(min_duration_s, duration_s)
            max_duration_s = max(max_duration_s, duration_s)
            low_start_min = min(low_start_min, low_start_count)
            low_start_max = max(low_start_max, low_start_count)
            low_start_total += low_start_count

    assert first_joint_order is not None
    remap = [first_joint_order.index(name) for name in policy_joint_names]
    return MotionSchemaSummary(
        motion_dir=Path(motion_dir),
        selection_path=selection_path,
        num_selected=len(paths),
        joint_order=first_joint_order,
        policy_to_dataset_joint_order=remap,
        min_frames=min_frames,
        max_frames=max_frames,
        min_duration_s=min_duration_s,
        max_duration_s=max_duration_s,
        low_start_min=low_start_min,
        low_start_max=low_start_max,
        low_start_total=low_start_total,
        low_start_zero_files=low_start_zero_files,
    )


class HostG1MotionBank:
    """Motion bank with per-env random start sampling and time interpolation."""

    def __init__(
        self,
        motion_dir: str | Path,
        selection: str | Path,
        *,
        device: torch.device | str,
        policy_joint_names: list[str] | tuple[str, ...] = XLoco_G1_29DOF_JOINT_NAMES,
        low_start_base_z: float = 0.55,
        low_start_uprightness: float = 0.50,
    ) -> None:
        self.device = torch.device(device)
        self.summary = validate_motion_dataset(motion_dir, selection, list(policy_joint_names))
        _, paths = selected_motion_paths(motion_dir, selection)
        dataset_order = self.summary.joint_order
        policy_to_dataset = [dataset_order.index(name) for name in policy_joint_names]

        self.records: list[_MotionRecord] = []
        for path in paths:
            data = np.load(path, allow_pickle=True)
            framerate = float(np.asarray(data["framerate"]).reshape(-1)[0])
            joint_pos_np = np.asarray(data["joint_pos"], dtype=np.float32)[:, policy_to_dataset]
            base_pos_np = np.asarray(data["base_pos_w"], dtype=np.float32)
            base_quat_np = _normalize_quat_np(np.asarray(data["base_quat_w"], dtype=np.float32))

            joint_pos = torch.as_tensor(joint_pos_np, dtype=torch.float32, device=self.device)
            base_pos = torch.as_tensor(base_pos_np, dtype=torch.float32, device=self.device)
            base_quat = torch.as_tensor(base_quat_np, dtype=torch.float32, device=self.device)
            joint_vel = _finite_difference(joint_pos, framerate)

            uprightness = torch.as_tensor(_uprightness_wxyz_np(base_quat_np), dtype=torch.float32, device=self.device)
            low = (base_pos[:, 2] <= low_start_base_z) | (uprightness <= low_start_uprightness)
            low_start_frames = torch.nonzero(low, as_tuple=False).squeeze(-1)
            if low_start_frames.numel() == 0:
                low_start_frames = torch.zeros(1, dtype=torch.long, device=self.device)

            self.records.append(
                _MotionRecord(
                    path=path,
                    framerate=framerate,
                    joint_pos=joint_pos,
                    joint_vel=joint_vel,
                    base_pos_w=base_pos,
                    base_quat_w=base_quat,
                    low_start_frames=low_start_frames,
                )
            )

        self.num_motions = len(self.records)
        self.max_duration_s = max(record.duration_s for record in self.records)
        self.urdf_to_policy_order = [XLoco_G1_29DOF_JOINT_NAMES.index(name) for name in URDF_G1_29DOF_JOINT_NAMES]

    def sample_starts(self, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample low-start motion ids, frame ids, and start times."""

        motion_ids = torch.randint(self.num_motions, (count,), dtype=torch.long, device=self.device)
        frame_ids = torch.zeros(count, dtype=torch.long, device=self.device)
        start_times = torch.zeros(count, dtype=torch.float32, device=self.device)
        for motion_id in range(self.num_motions):
            env_mask = motion_ids == motion_id
            num = int(env_mask.sum().item())
            if num == 0:
                continue
            candidates = self.records[motion_id].low_start_frames
            picks = torch.randint(candidates.numel(), (num,), dtype=torch.long, device=self.device)
            frames = candidates[picks]
            frame_ids[env_mask] = frames
            start_times[env_mask] = frames.float() / self.records[motion_id].framerate
        return motion_ids, frame_ids, start_times

    def state_at(self, motion_ids: torch.Tensor, motion_times: torch.Tensor) -> dict[str, torch.Tensor]:
        """Interpolate reference state for a batch of motion ids and times."""

        count = int(motion_ids.numel())
        joint_pos = torch.zeros(count, NUM_ACTIONS, dtype=torch.float32, device=self.device)
        joint_vel = torch.zeros_like(joint_pos)
        base_pos = torch.zeros(count, 3, dtype=torch.float32, device=self.device)
        base_quat = torch.zeros(count, 4, dtype=torch.float32, device=self.device)
        base_quat[:, 0] = 1.0
        exhausted = torch.ones(count, dtype=torch.bool, device=self.device)

        for motion_id in torch.unique(motion_ids):
            mid = int(motion_id.item())
            rows = torch.nonzero(motion_ids == mid, as_tuple=False).squeeze(-1)
            record = self.records[mid]
            scaled = motion_times[rows].clamp(min=0.0) * record.framerate
            lo = torch.floor(scaled).long().clamp(0, record.num_frames - 1)
            hi = torch.clamp(lo + 1, max=record.num_frames - 1)
            alpha = (scaled - lo.float()).clamp(0.0, 1.0).unsqueeze(-1)

            joint_pos[rows] = torch.lerp(record.joint_pos[lo], record.joint_pos[hi], alpha)
            joint_vel[rows] = torch.lerp(record.joint_vel[lo], record.joint_vel[hi], alpha)
            base_pos[rows] = torch.lerp(record.base_pos_w[lo], record.base_pos_w[hi], alpha)
            quat = torch.lerp(record.base_quat_w[lo], record.base_quat_w[hi], alpha)
            base_quat[rows] = torch.nn.functional.normalize(quat, dim=-1)
            exhausted[rows] = motion_times[rows] >= record.duration_s

        return {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "base_pos_w": base_pos,
            "base_quat_w": base_quat,
            "exhausted": exhausted,
        }


def _finite_difference(values: torch.Tensor, framerate: float) -> torch.Tensor:
    vel = torch.zeros_like(values)
    if values.shape[0] > 1:
        vel[:-1] = (values[1:] - values[:-1]) * framerate
        vel[-1] = vel[-2]
    return vel


def _normalize_quat_np(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat / np.maximum(norm, 1.0e-8)


def _uprightness_wxyz_np(quat: np.ndarray) -> np.ndarray:
    # IsaacLab quaternions use wxyz; uprightness is body z-axis dot world z-axis.
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    _ = w
    return 1.0 - 2.0 * (x * x + y * y)
