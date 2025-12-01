"""MuJoCo Franka Panda pick-and-place environment with safety checks."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import os
import numpy as np

# Configure osmesa rendering BEFORE importing mujoco (fixes EGL errors on headless servers)
os.environ['MUJOCO_GL'] = 'osmesa'

try:  # pragma: no cover - optional dependency for structured observations
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # pragma: no cover
    gym = None  # type: ignore
    spaces = None  # type: ignore

try:  # pragma: no cover - MuJoCo is required at runtime
    import mujoco
except Exception as exc:  # pragma: no cover
    mujoco = None  # type: ignore

try:  # pragma: no cover - viewer optional
    import mujoco.viewer as mujoco_viewer
except Exception:  # pragma: no cover
    mujoco_viewer = None  # type: ignore

from .controllers import infer_actuated_joints

DEFAULT_XML = "franka_emika_panda/task_scene.xml"
_HIDDEN_POSE = np.array([2.0, 2.0, -1.0, 1.0, 0.0, 0.0, 0.0])
_OBJECT_COLORS = ("red", "green", "blue", "yellow", "purple")


@dataclass(slots=True)
class StepResult:
    """Container returned by :meth:`FrankaPickPlaceEnv.step`."""

    observation: Dict[str, np.ndarray]
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, float]


class FrankaPickPlaceEnv:
    """Gym-like wrapper around a MuJoCo Franka Panda manipulation scene."""

    def __init__(
        self,
        asset_root: Path | str = Path("env/mujoco_assets"),
        *,
        gui: bool = False,
        width: int = 224,
        height: int = 224,
        seed: Optional[int] = 0,
        camera_name: str = "top",
        reward_type: str = "dense",  # "dense", "sparse", or "shaped"
    ) -> None:
        if mujoco is None:  # pragma: no cover - handled at runtime
            raise ImportError(
                "MuJoCo is required for FrankaPickPlaceEnv. Install 'mujoco' and place the"
                " Franka assets in env/mujoco_assets/."
            )

        # Convert to absolute path to avoid MuJoCo path resolution issues
        self.asset_root = Path(asset_root).resolve()
        self.gui = gui
        self.width = width
        self.height = height
        self.camera_name = camera_name
        self.rng = np.random.default_rng(seed)
        
        # Reward configuration
        if reward_type not in ["dense", "sparse", "shaped"]:
            raise ValueError(f"reward_type must be 'dense', 'sparse', or 'shaped', got '{reward_type}'")
        self.reward_type = reward_type

        xml_path = self.asset_root / DEFAULT_XML
        if not xml_path.exists():
            raise FileNotFoundError(
                f"MuJoCo XML '{xml_path}' not found. Refer to the README for asset setup instructions."
            )

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self._default_qpos = self.model.qpos0.copy()
        self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)

        self.step_dt = 0.04
        self.control_rate_hz = 1.0 / self.step_dt
        self.max_steps = 300  # Increased to allow keyframe controller to complete placement sequence
        self.success_height = 0.10  # Height threshold for considering object "placed"
        self.workspace_extent = np.array([0.25, 0.25])
        self.bin_position = np.array([0.55, 0.45, 0.08])  # Closer to robot for easier center placement
        self.bin_radius = 0.12  # Increased radius for easier placement

        self.viewer: Optional[object] = None
        if self.gui:
            if mujoco_viewer is None:  # pragma: no cover - viewer optional
                raise RuntimeError("MuJoCo viewer is unavailable; install mujoco>=2.3.5 or disable GUI mode.")
            self.viewer = mujoco_viewer.launch_passive(self.model, self.data)

        self._gripper_site_id = self._get_site_id("gripper")
        self._object_body_ids = {color: self._get_body_id(f"object_body_{color}") for color in _OBJECT_COLORS}
        self._object_site_ids = {color: self._get_site_id(f"object_site_{color}") for color in _OBJECT_COLORS}
        self._object_qpos_addrs = {color: self._free_joint_qpos_addr(body_id) for color, body_id in self._object_body_ids.items()}
        self._occluder_body = self._get_body_id("occluder_body")
        self._occluder_qpos_addr = self._free_joint_qpos_addr(self._occluder_body)
        self._light_id = self._get_light_id("top_light")

        joint_info = infer_actuated_joints(self.model)
        self._joint_ids = joint_info.joint_ids
        self._joint_qpos_indices = joint_info.qpos_indices
        self._joint_dof_indices = joint_info.dof_indices
        self._joint_limits = joint_info.limits
        # Forward-reaching configuration for better workspace coverage
        # This config starts lower (~30cm) and extends forward for pick-and-place
        self._home_configuration = np.array([0.0, 0.3, 0.0, -1.5, 0.0, 1.8, 0.79])

        self._active_objects: Tuple[str, ...] = tuple(_OBJECT_COLORS[:3])
        self._target_color = "red"
        self._instruction = ""
        self._hindered = False
        self._elapsed_steps = 0
        self._last_action = np.zeros(7, dtype=np.float64)
        self._max_abs_joint = 0.0
        self._max_abs_velocity = 0.0

        if spaces is not None:
            # Action space: 7 joint velocities [-0.5, 0.5] + 1 gripper position [0, 255]
            self.action_space = spaces.Box(
                low=np.array([-2.8973]*7 + [0.0]),  # Joint position limits + gripper closed
                high=np.array([2.8973]*7 + [0.04]),  # Joint position limits + gripper open
                dtype=np.float32
            )
            self.observation_space = spaces.Dict(
                {
                    "rgb_static": spaces.Box(low=0.0, high=1.0, shape=(self.height, self.width, 3), dtype=np.float32),
                    "proprio": spaces.Box(low=-np.inf, high=np.inf, shape=(len(self._joint_qpos_indices),), dtype=np.float32),
                }
            )

    # ------------------------------------------------------------------
    # Reset and stepping
    # ------------------------------------------------------------------
    def reset(self, *, hindered: bool = False) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self._default_qpos
        self.data.qvel[:] = 0.0

        self._hindered = hindered
        self._elapsed_steps = 0
        self._max_abs_joint = 0.0
        self._max_abs_velocity = 0.0

        self._randomize_robot_pose()
        self._active_objects = self._randomize_objects()
        self._target_color = self.rng.choice(self._active_objects)
        self._instruction = f"Pick up the {self._target_color} sphere and place it in the goal bin."
        self._apply_hindered_modifications()

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {
            "instruction": self._instruction,
            "target_color": self._target_color,
            "hindered": hindered,
            "control_dt": self.step_dt,
        }
        return obs, info

    def step(self, action: np.ndarray) -> StepResult:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (8,):
            raise ValueError("Action must be an 8-D vector: 7 joint positions + 1 gripper position.")
        if not np.all(np.isfinite(action)):
            raise ValueError("Action contains non-finite values.")

        # Split action into arm positions and gripper position
        arm_positions = action[:7]
        gripper_position = action[7]
        
        # Clip arm positions to joint limits
        self._last_action = np.clip(arm_positions, -2.8973, 2.8973)
        if self.model.nu >= 7:
            # Position control: send target positions directly
            self.data.ctrl[:7] = self._last_action
            # Apply gripper position (actuator 8 uses different range)
            if self.model.nu >= 8:
                self.data.ctrl[7] = np.clip(gripper_position, 0.0, 0.04)  # Gripper range is 0-0.04m
        else:  # pragma: no cover - fallback
            self.data.qpos[self._joint_qpos_indices] = self._last_action

        substeps = max(1, int(round(self.step_dt / self.model.opt.timestep)))
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)

        self._enforce_limits()
        mujoco.mj_forward(self.model, self.data)
        self._update_safety_stats()

        obs = self._get_obs()
        reward = self._compute_reward()
        success = self._check_success()

        self._elapsed_steps += 1
        terminated = success
        truncated = self._elapsed_steps >= self.max_steps
        info = {
            "success": float(success),
            "target_color": self._target_color,
            "hindered": float(self._hindered),
            "distance": float(self._target_distance()),
            "joint_pos_max_abs": self._max_abs_joint,
            "joint_vel_max_abs": self._max_abs_velocity,
        }
        return StepResult(obs, reward, terminated, truncated, info)

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get current observation dictionary.
        
        Returns:
            Dictionary containing:
                - "rgb_static": (H, W, 3) float32 RGB image normalized to [0, 1]
                - "proprio": (7,) float32 joint positions in radians
        """
        image = self.render(mode="rgb_array")
        if not np.all(np.isfinite(image)):
            raise RuntimeError("Rendered image contains non-finite values.")
        proprio = self.data.qpos[self._joint_qpos_indices].astype(np.float32).copy()
        return {"rgb_static": image, "proprio": proprio}

    # ------------------------------------------------------------------
    # Reward and termination utilities
    # ------------------------------------------------------------------
    def _target_distance(self) -> float:
        gripper_pos = self.data.site_xpos[self._gripper_site_id]
        target_pos = self.data.site_xpos[self._object_site_ids[self._target_color]]
        return float(np.linalg.norm(gripper_pos - target_pos))

    def _compute_reward(self) -> float:
        """Compute reward based on configured reward type.
        
        Reward types:
        - "dense": Negative distance to target (dense feedback, easier to learn)
        - "sparse": +1 for success, 0 otherwise (harder, but more realistic)
        - "shaped": Dense distance reward + bonus for success (best of both worlds)
        """
        if self.reward_type == "sparse":
            # Sparse reward: only reward success
            return 1.0 if self._check_success() else 0.0
        
        elif self.reward_type == "dense":
            # Dense reward: negative distance (original behavior)
            return -self._target_distance()
        
        elif self.reward_type == "shaped":
            # Shaped reward: dense feedback + success bonus
            distance_reward = -self._target_distance()
            success_bonus = 10.0 if self._check_success() else 0.0
            return distance_reward + success_bonus
        
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

    def _check_success(self) -> bool:
        """Check if the target object is successfully placed in the bin.
        
        Success requires:
        1. Object within bin_radius of the bin's horizontal position
        2. Object at VERY LOW height (< 0.08m) - actually in the bin, not just passing by
        3. Object velocity low (< 0.1 m/s) - at rest, not being dropped
        """
        site_id = self._object_site_ids[self._target_color]
        body_id = self._object_body_ids[self._target_color]
        obj_pos = self.data.site_xpos[site_id]
        
        # Check horizontal proximity to bin
        horizontal_dist = np.linalg.norm(obj_pos[:2] - self.bin_position[:2])
        if horizontal_dist >= self.bin_radius:
            return False
        
        # Check that object is IN the bin (very low height < 0.08m)
        # Bin floor is at ~0.02m, walls at 0.06m, so < 0.08m means actually in bin
        if obj_pos[2] >= 0.08:
            return False
        
        # Check that object is at rest (not being actively dropped)
        # Get object velocity from body velocities
        body_vel_adr = self.model.body_dofadr[body_id]
        if body_vel_adr >= 0:  # Body has DOFs
            body_vel = self.data.qvel[body_vel_adr:body_vel_adr+6]  # 6 DOF (3 pos + 3 rot)
            linear_vel = np.linalg.norm(body_vel[:3])
            if linear_vel > 0.1:  # Moving faster than 0.1 m/s
                return False
        
        return True

    # ------------------------------------------------------------------
    # Randomisation helpers
    # ------------------------------------------------------------------
    def _randomize_robot_pose(self) -> None:
        home = self._home_configuration
        noise = self.rng.uniform(-0.05, 0.05, size=home.shape)
        self.data.qpos[self._joint_qpos_indices] = home + noise
        self.data.qvel[self._joint_dof_indices] = 0.0

    def _randomize_objects(self) -> Tuple[str, ...]:
        count = int(self.rng.integers(3, len(_OBJECT_COLORS) + 1))
        active = tuple(self.rng.choice(_OBJECT_COLORS, size=count, replace=False))
        
        # Track placed ball positions to prevent overlap
        placed_positions = []
        min_separation = 0.07  # 7cm minimum distance (ball diameter ~6cm + 1cm buffer)
        
        for color in _OBJECT_COLORS:
            addr = self._object_qpos_addrs[color]
            if color in active:
                # Place objects in the working zone (0.45-0.55m forward, ±0.25m lateral)
                # This range has proven successful grasping with our keyframes
                # Try up to 50 times to find a non-overlapping position
                for attempt in range(50):
                    x = self.rng.uniform(0.45, 0.55)
                    # y = self.rng.uniform(-0.25, 0.25)
                    # MATCH DATASET DISTRIBUTION: Dataset has Y in [-0.25, 0.25]
                    y = self.rng.uniform(-0.25, 0.25) 
                    pos = np.array([x, y, 0.035], dtype=np.float64)
                    
                    # Check if this position overlaps with any existing balls
                    overlap = False
                    for existing_pos in placed_positions:
                        dist = np.linalg.norm(pos[:2] - existing_pos[:2])
                        if dist < min_separation:
                            overlap = True
                            break
                    
                    if not overlap:
                        # Valid position found
                        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
                        self._set_free_joint_pose(addr, pos, quat)
                        placed_positions.append(pos)
                        break
                else:
                    # Fallback: If we can't find non-overlapping position after 50 tries,
                    # place it anyway (rare edge case)
                    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
                    self._set_free_joint_pose(addr, pos, quat)
                    placed_positions.append(pos)
            else:
                self._set_free_joint_pose(addr, _HIDDEN_POSE[:3], _HIDDEN_POSE[3:])
        return active

    def _apply_hindered_modifications(self) -> None:
        # Reset lighting (light attributes are 3D RGB arrays)
        self.model.light_ambient[self._light_id] = np.array([0.4, 0.4, 0.4])
        self.model.light_diffuse[self._light_id] = np.array([0.6, 0.6, 0.6])

        # Always hide the occluder cube (optimization: no distracting objects)
        # self._set_free_joint_pose(self._occluder_qpos_addr, _HIDDEN_POSE[:3], _HIDDEN_POSE[3:])
        
        # Note: 'hindered' flag is still tracked for dataset labels, but no visual changes
        if not self._hindered:
            self._set_free_joint_pose(self._occluder_qpos_addr, _HIDDEN_POSE[:3], _HIDDEN_POSE[3:])
            return
        
        # Randomize lighting VERY aggressively for distinct tints
        # Generate a random saturated color
        color = self.rng.uniform(0.0, 1.0, size=3)
        # Boost saturation: make the max channel 1.0 and min channel close to 0.0
        color[np.argmax(color)] = 1.0
        color[np.argmin(color)] = self.rng.uniform(0.0, 0.2)
        
        self.model.light_diffuse[self._light_id] = color
        self.model.light_ambient[self._light_id] = color * 0.2 # Dim ambient to let shadows/diffuse pop
        
        print(f"DEBUG: Hindered Lighting - Color: {color}")

        # Perturb camera position slightly (simulating calibration error)
        # Note: This modifies the model, so it persists. We rely on reset() to restore if needed,
        # but since we don't store default, we just apply small random noise around the *current* position.
        # Ideally we should store default, but for now small jitter is fine as long as it doesn't drift too far.
        # Actually, let's be safe and only modify if we haven't drifted too far, or just accept small drift.
        # Better: The camera is static in XML. We can just add noise to the *view* matrix if we could,
        # but here we modify model.cam_pos.
        # Let's assume the drift is negligible over a few episodes or we restart.
        
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.camera_name)
        if cam_id >= 0:
            # Apply AGGRESSIVE jitter: +/- 10cm position
            pos_noise = self.rng.uniform(-0.1, 0.1, size=3)
            original_pos = self.model.cam_pos[cam_id].copy()
            self.model.cam_pos[cam_id] += pos_noise
            
            # Apply rotation jitter (small random rotation)
            # Create a small random rotation quaternion
            # Axis-angle: random axis, angle between -10 and 10 degrees
            axis = self.rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            angle = np.radians(self.rng.uniform(-10, 10))
            
            # Convert to quaternion [w, x, y, z]
            sin_half = np.sin(angle / 2)
            cos_half = np.cos(angle / 2)
            rot_quat = np.array([cos_half, axis[0]*sin_half, axis[1]*sin_half, axis[2]*sin_half])
            
            # Apply rotation: q_new = q_noise * q_old
            current_quat = self.model.cam_quat[cam_id]
            # Hamilton product
            w1, x1, y1, z1 = rot_quat
            w2, x2, y2, z2 = current_quat
            new_quat = np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ])
            self.model.cam_quat[cam_id] = new_quat / np.linalg.norm(new_quat)
            
            print(f"DEBUG: Camera Jitter - Pos Delta: {pos_noise}, New Pos: {self.model.cam_pos[cam_id]}")

        # Increase occluder size
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "occluder_geom")
        if geom_id >= 0:
            # Make it bigger: 8cm half-extent -> 16cm cube (vs original 5cm/10cm)
            self.model.geom_size[geom_id] = np.array([0.08, 0.08, 0.08])

        xy = self.rng.uniform(-self.workspace_extent * 0.5, self.workspace_extent * 0.5)
        pos = np.array([0.5 + xy[0], xy[1], 0.18], dtype=np.float64)
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._set_free_joint_pose(self._occluder_qpos_addr, pos, quat)

    # ------------------------------------------------------------------
    # Safety helpers
    # ------------------------------------------------------------------
    def _enforce_limits(self) -> None:
        for idx, (lower, upper) in zip(self._joint_qpos_indices, self._joint_limits):
            if np.isfinite(lower) and self.data.qpos[idx] < lower:
                self.data.qpos[idx] = lower
            if np.isfinite(upper) and self.data.qpos[idx] > upper:
                self.data.qpos[idx] = upper
        for addr in self._free_joint_addresses():
            quat = self.data.qpos[addr + 3 : addr + 7]
            self.data.qpos[addr + 3 : addr + 7] = self._normalize_quaternion(quat)

    def _update_safety_stats(self) -> None:
        joints = self.data.qpos[self._joint_qpos_indices]
        vels = self.data.qvel[self._joint_dof_indices]
        if not np.all(np.isfinite(joints)) or not np.all(np.isfinite(vels)):
            raise RuntimeError("Non-finite values encountered in joint states.")
        self._max_abs_joint = max(self._max_abs_joint, float(np.max(np.abs(joints))))
        self._max_abs_velocity = max(self._max_abs_velocity, float(np.max(np.abs(vels))))

    # ------------------------------------------------------------------
    # Rendering and cleanup
    # ------------------------------------------------------------------
    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode == "rgb_array":
            # Use existing renderer
            if self.renderer is None:
                 self.renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
            
            # CRITICAL: Don't pass camera parameter to update_scene - it causes frozen images!
            # Just use default camera (first camera in model, which is 'top')
            self.renderer.update_scene(self.data)
            rgb = self.renderer.render()
            
            # Convert from uint8 [0, 255] to float32 [0, 1]
            return (rgb / 255.0).astype(np.float32)
        if mode == "human":  # pragma: no cover - viewer only
            if self.viewer is None:
                raise RuntimeError("Viewer not initialised; construct environment with gui=True.")
            self.viewer.sync()
            return np.zeros((self.height, self.width, 3), dtype=np.float32)
        raise ValueError("Unsupported render mode; expected 'rgb_array' or 'human'.")

    def close(self) -> None:
        if self.viewer is not None:  # pragma: no cover
            self.viewer.close()
            self.viewer = None
        if hasattr(self.renderer, "free"):
            self.renderer.free()  # type: ignore[attr-defined]
        self.renderer = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _get_body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Body '{name}' not found in MuJoCo model.")
        return body_id

    def _get_site_id(self, name: str) -> int:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            raise ValueError(f"Site '{name}' not found in MuJoCo model.")
        return site_id

    def _get_light_id(self, name: str) -> int:
        light_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_LIGHT, name)
        if light_id < 0:
            raise ValueError(f"Light '{name}' not found in MuJoCo model.")
        return light_id

    def _free_joint_qpos_addr(self, body_id: int) -> int:
        joint_adr = self.model.body_jntadr[body_id]
        if joint_adr < 0:
            raise ValueError("Body does not have an associated joint for pose updates.")
        if self.model.jnt_type[joint_adr] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Expected a free joint for movable body.")
        return self.model.jnt_qposadr[joint_adr]

    def _set_free_joint_pose(self, addr: int, pos: Sequence[float], quat: Sequence[float]) -> None:
        self.data.qpos[addr : addr + 3] = np.asarray(pos, dtype=np.float64)
        self.data.qpos[addr + 3 : addr + 7] = self._normalize_quaternion(np.asarray(quat, dtype=np.float64))

    def _free_joint_addresses(self) -> Iterable[int]:
        for addr in self._object_qpos_addrs.values():
            yield addr
        yield self._occluder_qpos_addr

    @staticmethod
    def _normalize_quaternion(quat: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(quat)
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return quat / norm

    @property
    def instruction(self) -> str:
        return self._instruction

    @property
    def target_color(self) -> str:
        return self._target_color

    def __del__(self) -> None:  # pragma: no cover - destructor best-effort
        try:
            self.close()
        except Exception:
            pass


def _smoke_test(asset_root: Path) -> None:  # pragma: no cover - CLI helper
    from .controllers import KinematicsHelper, quat_to_axis_angle, quat_multiply, quat_conjugate
    
    env = FrankaPickPlaceEnv(asset_root=asset_root, gui=False)
    obs, info = env.reset()
    print("Reset observation keys:", obs.keys())
    print("Instruction:", info["instruction"])
    
    # Create kinematics helper for Jacobian-based control
    kin_helper = KinematicsHelper(env.model, site_name="gripper")
    
    # Extract the initial "good" downward orientation as reference
    initial_gripper_quat = np.zeros(4, dtype=np.float64)
    import mujoco
    mujoco.mju_mat2Quat(initial_gripper_quat, env.data.site_xmat[env._gripper_site_id])
    print(f"Initial gripper quaternion (good downward orientation): {initial_gripper_quat}")
    downward_quat_reference = initial_gripper_quat.copy()
    
    # Control frequency management - run Jacobian control at 50Hz instead of per-step
    CONTROL_HZ = 50  # Standard control frequency for Franka Cartesian control
    control_decimation = max(1, int(round(1.0 / (CONTROL_HZ * 0.02))))  # assuming 50Hz physics
    
    # Initialize cached action
    if not hasattr(env, '_last_jacobian_action'):
        env._last_jacobian_action = np.zeros(7)  # type: ignore
    
    for step_idx in range(300):
        # Multi-phase proportional controller with 6D control (position + orientation)
        target_site = env._object_site_ids[env._target_color]
        target_pos = env.data.site_xpos[target_site]
        gripper_pos = env.data.site_xpos[env._gripper_site_id]
        
        # Compute position error
        error = target_pos - gripper_pos
        horizontal_error = np.array([error[0], error[1], 0.0])
        horizontal_dist = np.linalg.norm(horizontal_error[:2])
        
        # Get bin position for placement phase
        bin_pos = env.bin_position
        bin_horizontal_dist = np.linalg.norm(target_pos[:2] - bin_pos[:2])
        
        gain = 2.0  # More aggressive for faster convergence (Step 4 from plan)
        
        # Track descent phase
        if not hasattr(env, '_descent_phase_active'):
            env._descent_phase_active = False  # type: ignore
            env._ik_recompute_counter = 0  # type: ignore
        
        # Multi-phase pick-and-place behavior with vertical descent
        # Use different thresholds for entering vs staying in descend phase (hysteresis)
        # All values in meters: 0.01 = 10mm, 0.02 = 20mm
        # Increased to 0.20 (200mm) to account for kinematic limits at lower heights
        descend_entry_threshold = 0.20  # Must be within 200mm before entering descent phase
        descend_exit_threshold = 0.70  # Much larger hysteresis to prevent phase bouncing
        orientation_threshold = 75.0  # Relaxed from 15 to 75 degrees - focus on position first
        
        # Compute gripper orientation before phase checks (needs to be before phase logic)
        current_mat_temp = env.data.site_xmat[env._gripper_site_id].reshape(3, 3)
        local_z_temp = current_mat_temp[:, 2]
        desired_axis_temp = np.array([0.0, 0.0, -1.0])
        cos_angle_temp = float(np.clip(np.dot(local_z_temp, desired_axis_temp), -1.0, 1.0))
        angle_deg = float(np.arccos(cos_angle_temp) * 180.0 / np.pi)
        
        # Check if we can descend: use position only, orientation will naturally improve during descent
        can_descend = horizontal_dist <= descend_entry_threshold
        
        # Track whether we've actually started descending (state-based, not just distance threshold)
        if not hasattr(env, '_is_in_descent_phase'):
            env._is_in_descent_phase = False  # type: ignore
        
        # Enter descent phase ONLY when can_descend is True
        if can_descend:
            env._is_in_descent_phase = True  # type: ignore
        
        # Exit descent phase when far away (hysteresis)
        if horizontal_dist > descend_exit_threshold:
            env._is_in_descent_phase = False  # type: ignore
        
        is_descending = env._is_in_descent_phase  # type: ignore
        
        # Don't use separate ORIENT phase - let orientation improve naturally
        orientation_good = True  # Always proceed with positioning
        
        # Get current joint configuration (needed for both IK and Jacobian control)
        current_q = env.data.qpos[env._joint_qpos_indices].copy()
        
        # Step 1: Control frequency management - only recompute Jacobian control every N steps
        recompute_control = (step_idx % control_decimation == 0)
        
        if recompute_control and target_pos[2] < 0.1 and not can_descend:
            # Phase 1: POSITION - IK-based planning with joint-space tracking
            env._descent_phase_active = False  # type: ignore
            
            safe_height = 0.15  # Increased from 0.10 for better workspace
            # CRITICAL: Keep current height until horizontally converged
            horizontal_dist_2d = np.linalg.norm(target_pos[:2] - gripper_pos[:2])
            if horizontal_dist_2d > 0.08:  # Far horizontally: stay at current height
                above_ball_pos = np.array([target_pos[0], target_pos[1], gripper_pos[2]])
            else:  # Close horizontally: can descend to safe height
                above_ball_pos = np.array([target_pos[0], target_pos[1], safe_height])
            
            # Compute distance-based IK weighting
            horizontal_dist_3d = np.linalg.norm(above_ball_pos[:2] - gripper_pos[:2])
            
            # Adaptive weighting: relax orientation when far/high
            if gripper_pos[2] > 0.20:  # High Z: prioritize position heavily
                pos_weight, ori_weight = 10.0, 0.1
            elif horizontal_dist_3d > 0.15:  # Far: moderate position priority
                pos_weight, ori_weight = 5.0, 0.5
            elif horizontal_dist_3d > 0.08:  # Medium: balanced
                pos_weight, ori_weight = 2.0, 1.0
            else:  # Close: fully balanced
                pos_weight, ori_weight = 1.0, 1.0
            
            # APPROACH 2: Use IK to compute target joint configuration, then track it
            try:
                target_q = kin_helper.inverse_kinematics(
                    target_pos=above_ball_pos,
                    target_quat=downward_quat_reference,
                    initial_q=current_q,
                    max_iters=150,  # Increased for better convergence
                    position_weight=pos_weight,
                    orientation_weight=ori_weight,
                    damping=0.01,
                    step_size=0.5,
                )
                
                # Simple joint-space PD control to track the target configuration
                joint_error = target_q - current_q
                joint_vel = 0.5 * joint_error  # Proportional gain of 0.5
                
                # Convert to position command
                action = joint_vel * 0.02
                action = np.clip(action, -0.05, 0.05)
                
            except RuntimeError:
                # IK failed to converge - fall back to current position
                action = np.zeros(7)
            
            # Diagnostic logging
            if step_idx % 20 == 0:
                ik_success = "IK_OK" if 'target_q' in locals() else "IK_FAIL"
                print(f"  [POSITION-IK] horiz_dist={horizontal_dist*1000:.1f}mm "
                      f"pos_w={pos_weight:.1f} ori_w={ori_weight:.1f} "
                      f"{ik_success} "
                      f"gripper_z={gripper_pos[2]*100:.1f}cm "
                      f"angle_deg={angle_deg:.1f}°")
            
            env._last_jacobian_action = action  # Cache for next steps
        
        elif recompute_control and target_pos[2] < 0.1 and (can_descend or is_descending):
            # Phase 2: DESCEND - IK-based planning with joint-space tracking
            
            horizontal_error_val = np.linalg.norm(horizontal_error[:2])
            
            # Staged descent: fix XY first, then Z
            if horizontal_error_val > 0.060:
                target_offset = np.array([target_pos[0], target_pos[1], gripper_pos[2]])
            else:
                target_offset = np.array([target_pos[0], target_pos[1], 0.04])
            
            # Balanced weighting during descent
            pos_weight, ori_weight = 2.0, 1.0
            
            # IK-based planning
            try:
                target_q = kin_helper.inverse_kinematics(
                    target_pos=target_offset,
                    target_quat=downward_quat_reference,
                    initial_q=current_q,
                    max_iters=150,  # Increased for better convergence
                    position_weight=pos_weight,
                    orientation_weight=ori_weight,
                    damping=0.01,
                    step_size=0.5,
                )
                
                # Joint-space PD tracking
                joint_error = target_q - current_q
                joint_vel = 0.5 * joint_error
                
                action = joint_vel * 0.02
                action = np.clip(action, -0.05, 0.05)
                
            except RuntimeError:
                action = np.zeros(7)
            
            if step_idx % 10 == 0:
                ik_success = "IK_OK" if 'target_q' in locals() else "IK_FAIL"
                print(f"  [DESCEND-IK] horiz_err={horizontal_error_val*1000:.1f}mm "
                      f"{ik_success} "
                      f"gripper_z={gripper_pos[2]*100:.1f}cm "
                      f"angle_deg={angle_deg:.1f}°")
            
            env._last_jacobian_action = action  # Cache for next steps
        
        elif recompute_control:
            # Phases 3-5: Use Jacobian control for lift, transport, place
            # Reset descent phase tracking when in other phases
            env._descent_phase_active = False  # type: ignore
            
            if target_pos[2] >= 0.1 and target_pos[2] < 0.25:
                # Phase 3: Lift - Object grasped, lift it up to safe carrying height
                lift_height = 0.3
                desired_pos_vel = gain * np.array([0.0, 0.0, lift_height - target_pos[2]])
            elif bin_horizontal_dist > 0.1:
                # Phase 4: Transport - Carry object to bin horizontally at safe height
                bin_horizontal_error = bin_pos[:2] - target_pos[:2]
                desired_pos_vel = gain * np.array([bin_horizontal_error[0], bin_horizontal_error[1], 0.0])
            else:
                # Phase 5: Place - Above bin, descend to place object
                bin_error = bin_pos - target_pos
                desired_pos_vel = gain * bin_error
            
            # Cross-product orientation control for later phases
            current_mat = env.data.site_xmat[env._gripper_site_id].reshape(3, 3)
            local_z_in_world = current_mat[:, 2]
            desired_axis = np.array([0.0, 0.0, -1.0])
            axis_err = np.cross(local_z_in_world, desired_axis)
            sin_angle = np.linalg.norm(axis_err)
            if sin_angle < 1e-6:
                desired_ang_vel = np.zeros(3)
            else:
                axis_err /= sin_angle
                cos_angle = float(np.clip(np.dot(local_z_in_world, desired_axis), -1.0, 1.0))
                angle = float(np.arccos(cos_angle))
                orientation_gain = 2.0 * gain
                desired_ang_vel = orientation_gain * angle * axis_err
                desired_ang_vel = np.clip(desired_ang_vel, -0.5, 0.5)
            
            # Combine position and orientation
            desired_cart_vel = np.concatenate([desired_pos_vel, desired_ang_vel])
            
            # Jacobian-based control
            jac = kin_helper.analytic_jacobian(current_q)
            lambda_damping = 0.05  # Standard damping
            jac_damped = jac.T @ np.linalg.inv(jac @ jac.T + lambda_damping * np.eye(6))
            
            # Compute joint-space command
            joint_command = jac_damped @ desired_cart_vel
            
            # Null-space control
            forward_home = np.array([0.0, 0.3, 0.0, -1.5, 0.0, 1.8, 0.79])
            null_space_proj = np.eye(7) - jac_damped @ jac
            home_error = forward_home - current_q
            null_space_gains = np.array([0.3, 0.5, 0.3, 0.4, 0.3, 0.3, 0.2])
            null_space_command = null_space_proj @ (null_space_gains * home_error)
            joint_command += null_space_command
            
            # Scale and clip for position control at 50Hz
            action = joint_command * 0.02
            action = np.clip(action, -0.1, 0.1)  # Allow more motion for lift/transport
            env._last_jacobian_action = action  # Cache for next steps
        
        else:
            # Not recomputing - reuse cached action from previous control step
            action = env._last_jacobian_action
        
        # Add gripper control
        dist_to_target = np.linalg.norm(gripper_pos - target_pos)
        
        if target_pos[2] < 0.1 and not can_descend:
            # Phase 1: Positioning above - gripper open
            gripper_cmd = 0.04
        elif target_pos[2] < 0.1 and (can_descend or is_descending):
            # Phase 2: Descending - close gripper when within millimeters for precision grasp
            # Wait until very close for reliable contact
            if dist_to_target < 0.025:  # Close when within 25mm of ball center
                gripper_cmd = 0.0
            else:
                gripper_cmd = 0.04  # Keep open while approaching
        elif target_pos[2] >= 0.1 and bin_horizontal_dist > 0.08:
            # Phase 3 & 4: Lifting and transporting - gripper closed
            gripper_cmd = 0.0
        else:
            # Phase 5: Placing - gripper open to release
            gripper_cmd = 0.04
        
        # For position control, compute target joint positions
        # All phases use Jacobian control now
        target_joint_positions = current_q + action
        
        # Clip to joint limits for safety
        target_joint_positions = np.clip(target_joint_positions, -2.8973, 2.8973)
        
        action = np.concatenate([target_joint_positions, [gripper_cmd]])
        
        result = env.step(action)
        
        # Sync the viewer to display the updated state
        if env.viewer is not None:
            env.viewer.sync()
        
        if step_idx % 40 == 0:
            obj_height = target_pos[2]
            bin_dist = bin_horizontal_dist
            # Determine phase
            if target_pos[2] < 0.1 and not can_descend:
                phase = "POSITION"
            elif target_pos[2] < 0.1 and (can_descend or is_descending):
                phase = "DESCEND"
            elif target_pos[2] >= 0.1 and target_pos[2] < 0.25:
                phase = "LIFT"
            elif bin_horizontal_dist > 0.1:
                phase = "TRANSPORT"
            else:
                phase = "PLACE"
            
        print(
                f"step {step_idx:3d} | {phase:9s} | success={result.info['success']} "
                f"reward={result.reward:+.3f} | dist={result.info['distance']:.3f} "
                f"horiz_dist={horizontal_dist:.3f} angle_deg={angle_deg:.1f} can_desc={can_descend} "
                f"obj_z={obj_height:.3f} bin_dist={bin_dist:.3f} gripper={gripper_cmd:.3f}"
        )
        if result.terminated or result.truncated:
            print(f"\nEpisode finished at step {step_idx}! Success: {result.info['success']}")
            break
    env.close()


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test for FrankaPickPlaceEnv")
    parser.add_argument("--asset-root", type=Path, default=Path("env/mujoco_assets"))
    args = parser.parse_args()
    if mujoco is None:
        raise SystemExit("MuJoCo not installed")
    _smoke_test(args.asset_root)
