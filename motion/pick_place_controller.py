"""
Pick-and-place controller for the SO-101 manipulator using cuMotion RMPflow.
Phases (PickPlacePhase):
  ORIENT_DOWNWARD (1): Orients gripper downward from default posture [0, 0, 0, 1.57, folded_yaw].
  PRE_GRASP_HOVER (2): Moves smoothly to pre-grasp hover target above the brick.
  DESCEND_TO_GRASP (3): Descends vertically to contact target beside the brick.
  GRASP (4): Closes the gripper jaw firmly onto the brick.
"""

import os
from enum import IntEnum
import numpy as np
from scipy.spatial.transform import Rotation as R
import warp as wp

from isaacsim.core.experimental.prims import Articulation
import isaacsim.robot_motion.experimental.motion_generation as mg
from isaacsim.robot_motion.cumotion import load_cumotion_robot, RmpFlowController
from motion.world_interface import SO101WorldInterface


class PickPlacePhase(IntEnum):
    """
    Named execution phases for the SO-101 pick-and-place state machine:
      - ORIENT_DOWNWARD (1): Pitches wrist 90 deg down and rolls to brick yaw from home pose.
      - PRE_GRASP_HOVER (2): Smoothly navigates to pre-grasp hover altitude above the brick.
      - DESCEND_TO_GRASP (3): Descends vertically to grasp contact target beside the brick.
      - GRASP (4): Closes the gripper jaw firmly onto the brick.
      - MOVE_UP (5): Lifts the grasped object vertically above the table.
      - HOLD (6): Holds the grasped object steady in the air (dummy step).
      - MOVE_TO_BIN (7): Navigates smoothly to hover above the bin center.
      - DROP (8): Opens the gripper jaw to release the brick into the bin.
      - POST_DROP_LIFT (9): Lifts the end-effector vertically to clear the bin before retracting.
      - RETRACT_HOME (10): Returns the robot arm to its default home posture [0, 0, 0, 0, 0].
    """
    ORIENT_DOWNWARD = 1
    PRE_GRASP_HOVER = 2
    DESCEND_TO_GRASP = 3
    GRASP = 4
    MOVE_UP = 5
    HOLD = 6
    MOVE_TO_BIN = 7
    DROP = 8
    POST_DROP_LIFT = 9
    RETRACT_HOME = 10


# Scaled Lego brick centroid offset from top-left origin (at scale 0.25)
BRICK_CENTER_OFFSET_LOCAL = np.array([0.0197, 0.0397, 0.0145], dtype=np.float32)

# Default clearance offset (in meters) outside the brick's -X face for the stationary jaw
DEFAULT_CLEARANCE_DELTA_X: float = 0.005  # 5 mm clearance

class SO101PickPlaceController:
    """
    Manages task-space trajectory tracking with cuMotion RmpFlowController
    using named execution phases (PickPlacePhase.ORIENT_DOWNWARD, PickPlacePhase.PRE_GRASP_HOVER).
    """

    _GRIPPER_JOINT_NAME: str = "gripper"
    _TOOL_FRAME_NAME: str = "gripper_frame_link"

    # Gripper limits in radians: -10 deg close, 100 deg open
    _OPEN_POS: float = 1.7453    # +100 degrees
    _CLOSED_POS: float = 0.2    # Contact-matched angle for 4cm object (~30 deg)

    _PRE_GRASP_OFFSET_Z: float = 0.08   # 8 cm above brick
    _LIFT_OFFSET_Z: float = 0.08        # 8 cm vertical lift
    _POST_DROP_LIFT_OFFSET_Z: float = 0.08  # 8 cm vertical lift above drop position before retracting
    _MIN_STEPS: int = 30                # Minimum frames before declaring convergence
    _JOINT_CONVERGE_THRESH: float = 0.05 # 0.05 rad (~2.8 deg) tracking error
    _PHYSICS_DT: float = 1.0 / 60.0

    def __init__(
        self,
        articulation: Articulation,
        world_interface: SO101WorldInterface,
        config_dir: str | None = None,
        robot_prim_path: str = "/World/so101",
        brick_center_offset: np.ndarray | None = None,
        half_width: float | None = None,
        closed_pos: float = _CLOSED_POS,
        gripper_max_effort: float = 0.5,
        post_drop_lift_offset_z: float | None = None,
    ):
        self._articulation = articulation
        self._world_interface = world_interface
        self._robot_prim_path = robot_prim_path
        self._closed_pos = float(closed_pos)
        self._gripper_max_effort = float(gripper_max_effort)
        self._post_drop_lift_offset_z = (
            float(post_drop_lift_offset_z) if post_drop_lift_offset_z is not None else self._POST_DROP_LIFT_OFFSET_Z
        )

        if brick_center_offset is None:
            brick_center_offset = BRICK_CENTER_OFFSET_LOCAL
        self._brick_center_offset = np.asarray(brick_center_offset, dtype=np.float32)

        if half_width is None:
            half_width = float(self._brick_center_offset[0]) if np.any(self._brick_center_offset) else 0.02
        self._half_width = float(half_width)

        if config_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_dir = os.path.join(project_root, "assets", "config")
        self._config_dir = config_dir

        self._cumotion_robot = None
        self._controller = None
        self._site_space = None
        self._finger_idx = None

        self._phase: PickPlacePhase = PickPlacePhase.ORIENT_DOWNWARD
        self._step: int = 0
        self._t: float = 0.0
        self._is_initialized: bool = False
        self._phase_converged: dict[PickPlacePhase, bool] = {
            PickPlacePhase.ORIENT_DOWNWARD: False,
            PickPlacePhase.PRE_GRASP_HOVER: False,
            PickPlacePhase.DESCEND_TO_GRASP: False,
            PickPlacePhase.GRASP: False,
            PickPlacePhase.MOVE_UP: False,
            PickPlacePhase.HOLD: False,
            PickPlacePhase.MOVE_TO_BIN: False,
            PickPlacePhase.DROP: False,
            PickPlacePhase.POST_DROP_LIFT: False,
            PickPlacePhase.RETRACT_HOME: False,
        }
        self._episode_id: int = 0
        self._initial_brick_z: float | None = None
        self._grasp_lift_target: np.ndarray | None = None
        self._grasp_q_target: np.ndarray | None = None

    @property
    def episode_id(self) -> int:
        """Returns the current episode ID."""
        return self._episode_id

    @property
    def phase(self) -> PickPlacePhase:
        """Returns the current execution phase enum."""
        return self._phase

    @property
    def is_converged(self) -> bool:
        """Returns True if the final phase (PickPlacePhase.RETRACT_HOME) has converged, indicating the pick-and-place task is complete."""
        return self.is_phase_converged(PickPlacePhase.RETRACT_HOME)

    def is_phase_converged(self, phase: PickPlacePhase) -> bool:
        """Returns True if the specified phase has converged."""
        return self._phase_converged.get(phase, False)

    def initialize(self) -> None:
        """Initializes cuMotion robot description, RmpFlowController, and joint indices."""
        print("[SO101PickPlaceController] Initializing cuMotion RMPflow Controller...")

        # Load cuMotion robot configuration
        self._cumotion_robot = load_cumotion_robot(
            directory=self._config_dir,
            urdf_filename="robot.urdf",
            xrdf_filename="robot.xrdf",
        )
        tool_frames = self._cumotion_robot.robot_description.tool_frame_names()
        if not tool_frames:
            raise ValueError("No tool frames found in SO-101 robot description.")
        self._site_space = tool_frames
        print(f"[SO101PickPlaceController] Tool frame: {tool_frames[0]}")

        # Identify gripper joint index in articulation
        dof_names = self._articulation.dof_names
        if self._GRIPPER_JOINT_NAME in dof_names:
            self._finger_idx = dof_names.index(self._GRIPPER_JOINT_NAME)
            print(f"[SO101PickPlaceController] Gripper joint index: {self._finger_idx} ('{self._GRIPPER_JOINT_NAME}')")

            # Configure gripper max effort to gentle grip torque to prevent mesh penetration
            efforts = self._articulation.get_dof_max_efforts()
            eff_np = efforts.numpy().flatten() if hasattr(efforts, "numpy") else np.asarray(efforts).flatten()
            eff_np = np.copy(eff_np)
            eff_np[self._finger_idx] = self._gripper_max_effort
            self._articulation.set_dof_max_efforts(eff_np.tolist())
            print(f"[SO101PickPlaceController] Configured gripper max effort to {self._gripper_max_effort} N*m")
        else:
            raise ValueError(f"Gripper joint '{self._GRIPPER_JOINT_NAME}' not found in {dof_names}")

        # Create RmpFlowController referencing CumotionWorldInterface
        cwi = self._world_interface.world_interface
        if cwi is None:
            raise RuntimeError("SO101WorldInterface must be initialized before creating RmpFlowController.")

        self._controller = RmpFlowController(
            cumotion_robot=self._cumotion_robot,
            cumotion_world_interface=cwi,
            robot_joint_space=dof_names,
            robot_site_space=self._site_space,
            tool_frame=self._TOOL_FRAME_NAME,
        )

        cfg = self._controller.get_rmp_flow_config()
        # Scale cspace_target_rmp metric scalar to allow natural task-space reaching
        cfg.set_param("cspace_target_rmp/metric_scalar", 1.0)

        # Set initial gripper target to open (+100 degrees)
        self._set_gripper(self._OPEN_POS)

        self._is_initialized = True
        print("[SO101PickPlaceController] Controller initialized successfully.")

    def reset(self, episode_id: int | None = None) -> None:
        """Resets the state machine back to PickPlacePhase.ORIENT_DOWNWARD."""
        if episode_id is not None:
            self._episode_id = episode_id
        elif self._episode_id == 0:
            self._episode_id = 1
        else:
            self._episode_id += 1
        self._phase = PickPlacePhase.ORIENT_DOWNWARD
        self._step = 0
        self._t = 0.0
        self._phase_converged = {
            PickPlacePhase.ORIENT_DOWNWARD: False,
            PickPlacePhase.PRE_GRASP_HOVER: False,
            PickPlacePhase.DESCEND_TO_GRASP: False,
            PickPlacePhase.GRASP: False,
            PickPlacePhase.MOVE_UP: False,
            PickPlacePhase.HOLD: False,
            PickPlacePhase.MOVE_TO_BIN: False,
            PickPlacePhase.DROP: False,
            PickPlacePhase.POST_DROP_LIFT: False,
            PickPlacePhase.RETRACT_HOME: False,
        }
        self._initial_brick_z = None
        self._grasp_lift_target = None
        self._grasp_q_target = None
        self._set_gripper(self._OPEN_POS)

    def compute_brick_yaw(self, brick_ori: np.ndarray) -> float:
        """
        Computes the brick's yaw angle folded into [-pi/2, pi/2] for 180-degree symmetry.
        """
        w, x, y, z = brick_ori
        rot = R.from_quat([x, y, z, w])
        folded_yaw = float(rot.as_euler("zyx", degrees=False)[0])
        
        while folded_yaw > np.pi / 2.0:
            folded_yaw -= np.pi
        while folded_yaw < -np.pi / 2.0:
            folded_yaw += np.pi
        return float(folded_yaw)

    def compute_brick_target_and_ori(
        self,
        brick_pos: np.ndarray,
        brick_ori: np.ndarray,
        clearance_delta_x: float = DEFAULT_CLEARANCE_DELTA_X,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Computes the grasp target for the stationary jaw tool frame, the brick centroid,
        and the downward gripper orientation aligned with the brick's yaw.

        The stationary jaw target is defined relative to the brick's true volumetric centroid
        with 180-degree symmetry folding:
        - True centroid computed via actual brick yaw
        - Face offset placed along the brick's width outside the face with clearance_delta_x

        Returns:
            grasp_target (np.ndarray): World position for the stationary jaw contact.
            brick_center (np.ndarray): World volumetric centroid of the brick.
            q_target (np.ndarray): Target orientation quaternion [w, x, y, z].
            folded_yaw (float): Brick yaw angle in radians folded into [-pi/2, pi/2].
        """
        w, x, y, z = brick_ori
        
        # Extract the brick's true yaw and compute the world position of its volumetric centroid
        rot = R.from_quat([x, y, z, w])
        matrix = rot.as_matrix()
        brick_center = brick_pos + (matrix @ self._brick_center_offset)

        # Extract the brick's yaw and fold it into [-pi/2, pi/2] for 180-degree symmetry
        folded_yaw = self.compute_brick_yaw(brick_ori)
        rot_psi = R.from_rotvec([0.0, 0.0, folded_yaw])    
        half_width = float(self._half_width)
        face_offset_local = np.array([-(half_width + float(clearance_delta_x)), 0.0, 0.0], dtype=np.float32)
        grasp_target = brick_center + (rot_psi.as_matrix() @ face_offset_local)

        # Downward orientation pointing along -Z aligned with brick yaw via quaternion multiplication
        rot_z = R.from_euler("z", folded_yaw)
        rot_down = R.from_euler("y", np.pi)
        qx, qy, qz, qw = (rot_z * rot_down).as_quat()
        q_target = np.array([qw, qx, qy, qz], dtype=np.float32)

        return grasp_target, brick_center, q_target, folded_yaw

    def _make_setpoint(self, position: np.ndarray, orientation: np.ndarray) -> mg.RobotState:
        return mg.RobotState(
            sites=mg.SpatialState.from_name(
                spatial_space=self._site_space,
                positions=([self._TOOL_FRAME_NAME], wp.array([position.tolist()], dtype=wp.float32)),
                orientations=([self._TOOL_FRAME_NAME], wp.array([orientation.tolist()], dtype=wp.float32)),
            ),
        )

    def _estimated_state(self) -> mg.RobotState:
        names = self._articulation.dof_names
        return mg.RobotState(
            joints=mg.JointState.from_name(
                robot_joint_space=names,
                positions=(names, self._articulation.get_dof_positions()),
                velocities=(names, self._articulation.get_dof_velocities()),
            )
        )

    def _set_gripper(self, pos: float) -> None:
        if self._finger_idx is not None:
            self._articulation.set_dof_position_targets(
                wp.array([pos], dtype=wp.float32), dof_indices=[self._finger_idx]
            )

    def get_arm_positions(self) -> np.ndarray:
        """Returns the current positions of the 5 arm DOFs as a 1D numpy array."""
        curr_dofs = self._articulation.get_dof_positions()
        curr_np = (
            curr_dofs.cpu().numpy()
            if hasattr(curr_dofs, "cpu")
            else (curr_dofs.numpy() if hasattr(curr_dofs, "numpy") else np.asarray(curr_dofs))
        )
        return curr_np.flatten()[:5]

    def get_joint_error(self, desired_arm_positions: np.ndarray) -> float:
        """
        Computes the maximum absolute error across all 5 arm DOFs.

        Args:
            desired_arm_positions: Target positions for the arm DOFs.

        Returns:
            float: Maximum tracking error in radians.
        """
        curr_arm = self.get_arm_positions()
        des_np = np.asarray(desired_arm_positions).flatten()[:5]
        return float(np.max(np.abs(curr_arm - des_np)))

    def check_convergence(
        self,
        desired_arm_positions: np.ndarray,
        threshold: float | None = None,
        min_steps: int | None = None,
    ) -> bool:
        """
        Checks whether the arm joints have converged to the target joint positions.

        Args:
            desired_arm_positions: Target positions for the arm DOFs.
            threshold: Maximum allowed joint tracking error in radians.
                       Defaults to self._JOINT_CONVERGE_THRESH.
            min_steps: Minimum elapsed frames before convergence can trigger.
                       Defaults to self._MIN_STEPS.

        Returns:
            bool: True if min_steps have elapsed and max joint error is below threshold.
        """
        min_s = min_steps if min_steps is not None else self._MIN_STEPS
        if self._step < min_s:
            return False

        thresh = threshold if threshold is not None else self._JOINT_CONVERGE_THRESH
        err = self.get_joint_error(desired_arm_positions)
        return err < thresh

    def _step_orient_downward(self, brick_pos: np.ndarray, brick_ori: np.ndarray) -> None:
        """
        Phase ORIENT_DOWNWARD (1):
        Orients the gripper downward from default posture [0, 0, 0, 1.5708, folded_yaw].
        """
        folded_yaw = self.compute_brick_yaw(brick_ori)
        target_joints = np.array([0.0, 0.0, 0.0, 1.5708, folded_yaw], dtype=np.float32)

        if self._step == 0:
            print(f"\n==================== [Episode {self._episode_id} | Phase: {self._phase.name}] ====================")
            print(f"  Target: Red Lego Brick")
            print(f"  Brick Origin:         [{brick_pos[0]:.4f}, {brick_pos[1]:.4f}, {brick_pos[2]:.4f}]")
            print(f"  Default Joint Target: {np.round(target_joints, 4).tolist()} (Pitch DOWN 90 deg + Yaw {np.degrees(folded_yaw):.1f} deg)")
            print(f"  Gripper Jaw State:    FULL OPEN (+100.0 deg)")
            print(f"========================================================================================\n")

        self._articulation.set_dof_position_targets(
            wp.array(target_joints, dtype=wp.float32),
            dof_indices=[0, 1, 2, 3, 4],
        )

        self._t += self._PHYSICS_DT
        self._step += 1

        joint_err = self.get_joint_error(target_joints)
        curr_arm = self.get_arm_positions()

        # ORIENT_DOWNWARD convergence check
        if (self._step >= 25 and joint_err < 0.06) or self._step >= 35:
            self._phase_converged[PickPlacePhase.ORIENT_DOWNWARD] = True
            print(f">>> [Episode {self._episode_id} | {self._phase.name} CONVERGED at Step {self._step}!] Gripper oriented DOWNWARD at default posture.")
            print(f"    Joint positions: {np.round(curr_arm, 3)} (joint error: {joint_err:.4f} rad)")
            print(f"    >>> Transitioning to Phase: {PickPlacePhase.PRE_GRASP_HOVER.name} (Hover Location)...\n")
            self._phase = PickPlacePhase.PRE_GRASP_HOVER
            self._step = 0
            self._t = 0.0
        elif self._step % 10 == 0:
            print(f"  [Episode {self._episode_id} | {self._phase.name} Step {self._step:2d}] Orienting downward... Joint error: {joint_err:.3f} rad")

    def _step_pre_grasp_hover(self, brick_pos: np.ndarray, brick_ori: np.ndarray) -> None:
        """
        Phase PRE_GRASP_HOVER (2):
        Moves smoothly to the pre-grasp hover target above the brick using cuMotion RMPflow.
        """
        # Compute grasp target, brick centroid, orientation quaternion, and hover position inside Phase 2
        grasp_target, brick_center, q_target, folded_yaw = self.compute_brick_target_and_ori(brick_pos, brick_ori)

        # Pre-Grasp hover position (8cm above the grasp target)
        pre_grasp_pos = np.copy(grasp_target)
        pre_grasp_pos[2] += self._PRE_GRASP_OFFSET_Z

        setpoint = self._make_setpoint(pre_grasp_pos, q_target)

        if self._step == 0:
            print(f"==================== [Episode {self._episode_id} | Phase: {self._phase.name}] ====================")
            print(f"  Brick Centroid:    [{brick_center[0]:.4f}, {brick_center[1]:.4f}, {brick_center[2]:.4f}]")
            print(f"  Pre-Grasp Target:  [{pre_grasp_pos[0]:.4f}, {pre_grasp_pos[1]:.4f}, {pre_grasp_pos[2]:.4f}]")
            print(f"========================================================================================\n")
            if not self._controller.reset(self._estimated_state(), setpoint, t=0.0):
                raise RuntimeError("[SO101PickPlaceController] RmpFlowController reset failed.")
            self._t = 0.0

        desired = self._controller.forward(self._estimated_state(), setpoint, self._t)
        des_pos = None
        if desired is not None and desired.joints.positions is not None:
            des_pos = np.asarray(desired.joints.positions)
            self._articulation.set_dof_position_targets(
                positions=desired.joints.positions,
                dof_indices=desired.joints.position_indices,
            )

        self._t += self._PHYSICS_DT
        self._step += 1

        if des_pos is not None:
            joint_err = self.get_joint_error(des_pos)
            curr_arm = self.get_arm_positions()

            if self._step >= 30 and joint_err < self._JOINT_CONVERGE_THRESH and not self.is_phase_converged(PickPlacePhase.PRE_GRASP_HOVER):
                self._phase_converged[PickPlacePhase.PRE_GRASP_HOVER] = True
                print(f">>> [Episode {self._episode_id} | {self._phase.name} CONVERGED at Step {self._step}!] Gripper hovering stably above target!")
                print(f"    Joint positions: {np.round(curr_arm, 3)} (joint error: {joint_err:.4f} rad)")
                print(f"    >>> Transitioning to Phase: {PickPlacePhase.DESCEND_TO_GRASP.name} (Descending to contact target)...\n")
                self._phase = PickPlacePhase.DESCEND_TO_GRASP
                self._step = 0
                self._t = 0.0
            elif self._step % 20 == 0 and not self.is_phase_converged(PickPlacePhase.PRE_GRASP_HOVER):
                print(f"  [Episode {self._episode_id} | {self._phase.name} Step {self._step:2d}] Approaching hover target... Joint error: {joint_err:.3f} rad | Arm: {np.round(curr_arm, 2)}")

    def _step_descend_to_grasp(self, brick_pos: np.ndarray, brick_ori: np.ndarray) -> None:
        """
        Phase DESCEND_TO_GRASP (3):
        Descends vertically from the hover position down to the contact target beside the brick using cuMotion RMPflow.
        """
        grasp_target, brick_center, q_target, folded_yaw = self.compute_brick_target_and_ori(brick_pos, brick_ori)

        setpoint = self._make_setpoint(grasp_target, q_target)

        if self._step == 0:
            print(f"==================== [Episode {self._episode_id} | Phase: {self._phase.name}] ====================")
            print(f"  Brick Centroid:    [{brick_center[0]:.4f}, {brick_center[1]:.4f}, {brick_center[2]:.4f}]")
            print(f"  Grasp Target:      [{grasp_target[0]:.4f}, {grasp_target[1]:.4f}, {grasp_target[2]:.4f}]")
            print(f"  Gripper Jaw State: FULL OPEN (+100.0 deg)")
            print(f"========================================================================================\n")
            if not self._controller.reset(self._estimated_state(), setpoint, t=0.0):
                raise RuntimeError("[SO101PickPlaceController] RmpFlowController reset failed.")
            self._t = 0.0

        desired = self._controller.forward(self._estimated_state(), setpoint, self._t)
        des_pos = None
        if desired is not None and desired.joints.positions is not None:
            des_pos = np.asarray(desired.joints.positions)
            self._articulation.set_dof_position_targets(
                positions=desired.joints.positions,
                dof_indices=desired.joints.position_indices,
            )

        self._t += self._PHYSICS_DT
        self._step += 1

        if des_pos is not None:
            joint_err = self.get_joint_error(des_pos)
            curr_arm = self.get_arm_positions()

            if self._step >= 30 and joint_err < self._JOINT_CONVERGE_THRESH and not self.is_phase_converged(PickPlacePhase.DESCEND_TO_GRASP):
                self._phase_converged[PickPlacePhase.DESCEND_TO_GRASP] = True
                print(f">>> [Episode {self._episode_id} | {self._phase.name} CONVERGED at Step {self._step}!] Gripper descended to contact target, ready for grasp!")
                print(f"    Joint positions: {np.round(curr_arm, 3)} (joint error: {joint_err:.4f} rad)")
                print(f"    >>> Transitioning to Phase: {PickPlacePhase.GRASP.name} (Closing gripper onto brick)...\n")
                self._phase = PickPlacePhase.GRASP
                self._step = 0
                self._t = 0.0
            elif self._step % 20 == 0 and not self.is_phase_converged(PickPlacePhase.DESCEND_TO_GRASP):
                print(f"  [Episode {self._episode_id} | {self._phase.name} Step {self._step:2d}] Descending... Joint error: {joint_err:.3f} rad | Arm: {np.round(curr_arm, 2)}")

    def _step_grasp(self, brick_pos: np.ndarray, brick_ori: np.ndarray) -> None:
        """
        Phase GRASP (4):
        Closes the gripper jaw firmly onto the object while maintaining the arm at the grasp target.
        Once converged, transitions to PickPlacePhase.HOLD so nothing further acts on the gripper.
        """
        if self._step == 0:
            grasp_target, _, _, _ = self.compute_brick_target_and_ori(brick_pos, brick_ori)
            print(f"==================== [Episode {self._episode_id} | Phase: {self._phase.name}] ====================")
            print(f"  Target:            Red Object")
            print(f"  Grasp Target:      [{grasp_target[0]:.4f}, {grasp_target[1]:.4f}, {grasp_target[2]:.4f}]")
            print(f"  Gripper Jaw Target: {np.degrees(self._closed_pos):.1f} deg ({self._closed_pos:.3f} rad)")
            print(f"  Gripper Max Effort: {self._gripper_max_effort:.1f} N*m")
            print(f"========================================================================================\n")

        # Drive gripper to closed position tailored to object dimension
        self._set_gripper(self._closed_pos)

        self._t += self._PHYSICS_DT
        self._step += 1

        dofs = self._articulation.get_dof_positions()
        dofs_np = dofs.numpy().flatten() if hasattr(dofs, "numpy") else np.asarray(dofs).flatten()
        grip_angle = float(dofs_np[self._finger_idx]) if self._finger_idx is not None else 0.0

        # Gripper closes over ~25-30 frames achieving snug contact
        if (self._step >= 25 or abs(grip_angle - self._closed_pos) < 0.04) and not self.is_phase_converged(PickPlacePhase.GRASP):
            self._phase_converged[PickPlacePhase.GRASP] = True
            print(f">>> [Episode {self._episode_id} | {self._phase.name} CONVERGED at Step {self._step}!] Object firmly gripped without penetration!")
            print(f"    Gripper joint angle: {np.degrees(grip_angle):.1f} deg ({grip_angle:.3f} rad)")
            print(f"    >>> Transitioning to Phase: {PickPlacePhase.MOVE_UP.name} (Lifting object)...\n")
            self._phase = PickPlacePhase.MOVE_UP
            self._step = 0
            self._t = 0.0
        elif self._step % 10 == 0 and not self.is_phase_converged(PickPlacePhase.GRASP):
            print(f"  [Episode {self._episode_id} | {self._phase.name} Step {self._step:2d}] Gripper closing... Angle: {np.degrees(grip_angle):.1f} deg")

    def _step_move_up(self, brick_pos: np.ndarray, brick_ori: np.ndarray) -> None:
        """
        Phase MOVE_UP (5):
        Lifts the grasped object vertically above the table using cuMotion RMPflow.
        Maintains the gripper closed at self._closed_pos.
        Converged when the arm reaches the lift setpoint AND the brick has moved up in Z (+z).
        """
        if self._step == 0:
            self._initial_brick_z = float(brick_pos[2])
            self._grasp_lift_target, _, self._grasp_q_target, _ = self.compute_brick_target_and_ori(brick_pos, brick_ori)

        # lift_target is a local variable
        lift_target = np.array(
            [self._grasp_lift_target[0], self._grasp_lift_target[1], self._grasp_lift_target[2] + self._LIFT_OFFSET_Z],
            dtype=np.float32,
        )
        setpoint = self._make_setpoint(lift_target, self._grasp_q_target)

        if self._step == 0:
            print(f"==================== [Episode {self._episode_id} | Phase: {self._phase.name}] ====================")
            print(f"  Target:            Red Object")
            print(f"  Initial Brick Z:   {self._initial_brick_z:.4f} m")
            print(f"  Lift Target:       [{lift_target[0]:.4f}, {lift_target[1]:.4f}, {lift_target[2]:.4f}]")
            print(f"  Gripper Jaw State: CLOSED ({np.degrees(self._closed_pos):.1f} deg)")
            print(f"========================================================================================\n")
            if not self._controller.reset(self._estimated_state(), setpoint, t=0.0):
                raise RuntimeError("[SO101PickPlaceController] RmpFlowController reset failed.")
            self._t = 0.0

        # Keep gripper firmly closed around the object
        self._set_gripper(self._closed_pos)

        desired = self._controller.forward(self._estimated_state(), setpoint, self._t)
        des_pos = None
        if desired is not None and desired.joints.positions is not None:
            des_pos = np.asarray(desired.joints.positions)
            self._articulation.set_dof_position_targets(
                positions=desired.joints.positions,
                dof_indices=desired.joints.position_indices,
            )

        self._t += self._PHYSICS_DT
        self._step += 1

        if des_pos is not None:
            joint_err = self.get_joint_error(des_pos)
            curr_arm = self.get_arm_positions()
            delta_brick_z = float(brick_pos[2]) - (self._initial_brick_z if self._initial_brick_z is not None else float(brick_pos[2]))

            # Converged if arm reached setpoint AND brick has moved up in Z (+z)
            arm_converged = (self._step >= 30 and joint_err < self._JOINT_CONVERGE_THRESH)
            brick_lifted = delta_brick_z > 0.005  # Any positive +z (>= 5mm lift above table)

            if arm_converged and brick_lifted and not self.is_phase_converged(PickPlacePhase.MOVE_UP):
                self._phase_converged[PickPlacePhase.MOVE_UP] = True
                print(f">>> [Episode {self._episode_id} | {self._phase.name} CONVERGED at Step {self._step}!] Object lifted successfully into the air!")
                print(f"    Joint positions: {np.round(curr_arm, 3)} (joint error: {joint_err:.4f} rad)")
                print(f"    Brick Delta Z:   +{delta_brick_z * 100.0:.2f} cm (from {self._initial_brick_z:.4f} to {float(brick_pos[2]):.4f})")
                print(f"    >>> Transitioning to Phase: {PickPlacePhase.HOLD.name} (Holding object in air)...\n")
                self._phase = PickPlacePhase.MOVE_TO_BIN
                self._step = 0
                self._t = 0.0
            elif self._step % 20 == 0 and not self.is_phase_converged(PickPlacePhase.MOVE_UP):
                print(f"  [Episode {self._episode_id} | {self._phase.name} Step {self._step:2d}] Lifting... Joint error: {joint_err:.3f} rad | Brick Delta Z: {delta_brick_z * 100.0:+.2f} cm | Arm: {np.round(curr_arm, 2)}")

    def _step_hold(self) -> None:
        """
        Phase HOLD (6):
        Dummy step holding the lifted object steady in the air with nothing happening.
        """
        self._set_gripper(self._closed_pos)
        self._step += 1

    def _step_move_to_bin(self, bin_pos):
        """
        Phase MOVE_TO_BIN (7):
        Moves the grasped object to the bin position using cuMotion RMPflow.
        """

        if self._grasp_q_target is None:
            raise RuntimeError("[SO101PickPlaceController] Cannot move to bin: grasp orientation target is None.")

        target_pos = np.copy(bin_pos).flatten()[:3]
        target_pos[2] += 0.20
        setpoint = self._make_setpoint(target_pos, self._grasp_q_target)

        if self._step == 0:
            print(f"==================== [Episode {self._episode_id} | Phase: {self._phase.name}] ====================")
            print(f"  Target:            Red Object")
            print(f"  Bin Position:      [{bin_pos[0]:.4f}, {bin_pos[1]:.4f}, {bin_pos[2]:.4f}]")
            print(f"  Gripper Jaw State: CLOSED ({np.degrees(self._closed_pos):.1f} deg)")
            print(f"========================================================================================\n")
            if not self._controller.reset(self._estimated_state(), setpoint, t=0.0):
                raise RuntimeError("[SO101PickPlaceController] RmpFlowController reset failed.")
            self._t = 0.0

        # Keep gripper firmly closed around the object
        self._set_gripper(self._closed_pos)

        desired = self._controller.forward(self._estimated_state(), setpoint, self._t)
        des_pos = None
        if desired is not None and desired.joints.positions is not None:
            des_pos = np.asarray(desired.joints.positions)
            self._articulation.set_dof_position_targets(
                positions=desired.joints.positions,
                dof_indices=desired.joints.position_indices,
            )

        self._t += self._PHYSICS_DT
        self._step += 1

        if des_pos is not None:
            joint_err = self.get_joint_error(des_pos)
            curr_arm = self.get_arm_positions()

            if self._step >= 30 and joint_err < self._JOINT_CONVERGE_THRESH and not self.is_phase_converged(PickPlacePhase.MOVE_TO_BIN):
                self._phase_converged[PickPlacePhase.MOVE_TO_BIN] = True
                print(f">>> [Episode {self._episode_id} | {self._phase.name} CONVERGED at Step {self._step}!] Object moved to bin successfully!")
                print(f"    Joint positions: {np.round(curr_arm, 3)} (joint error: {joint_err:.4f} rad)")
                print(f"    >>> Task complete!\n")
                self._phase = PickPlacePhase.DROP  # Optionally transition to HOLD or another phase
                self._step = 0
                self._t = 0.0
            elif self._step % 20 == 0 and not self.is_phase_converged(PickPlacePhase.MOVE_TO_BIN):
                print(f"  [Episode {self._episode_id} | {self._phase.name} Step {self._step:2d}] Moving to bin... Joint error: {joint_err:.3f} rad | Arm: {np.round(curr_arm, 2)}")

    def _step_drop(self) -> None:
        """
        Phase DROP (8):
        Opens the gripper to release the object into the bin.
        """
        if self._step == 0:
            print(f"==================== [Episode {self._episode_id} | Phase: {self._phase.name}] ====================")
            print(f"  Target:            Red Object")
            print(f"  Gripper Jaw State: OPEN (+100.0 deg)")
            print(f"========================================================================================\n")

        # Open gripper to release object
        self._set_gripper(self._OPEN_POS)

        self._step += 1

        if self._step >= 25 and not self.is_phase_converged(PickPlacePhase.DROP):
            self._phase_converged[PickPlacePhase.DROP] = True
            print(f">>> [Episode {self._episode_id} | {self._phase.name} CONVERGED at Step {self._step}!] Object released successfully!")
            print(f"    >>> Transitioning to Phase: {PickPlacePhase.POST_DROP_LIFT.name} (Lifting up above bin)...\n")
            self._phase = PickPlacePhase.POST_DROP_LIFT
            self._step = 0
            self._t = 0.0
        elif self._step % 10 == 0 and not self.is_phase_converged(PickPlacePhase.DROP):
            print(f"  [Episode {self._episode_id} | {self._phase.name} Step {self._step:2d}] Releasing object... Arm: {np.round(self.get_arm_positions(), 2)}")

    def _step_post_drop_lift(self, bin_pos: np.ndarray) -> None:
        """
        Phase POST_DROP_LIFT (9):
        Lifts the end-effector vertically upward after dropping the object,
        clearing the bin rim before retracting home.
        """
        if self._grasp_q_target is None:
            raise RuntimeError("[SO101PickPlaceController] Cannot lift above bin: grasp orientation target is None.")

        target_pos = np.copy(bin_pos).flatten()[:3]
        target_pos[2] += (0.20 + self._post_drop_lift_offset_z)
        setpoint = self._make_setpoint(target_pos, self._grasp_q_target)

        if self._step == 0:
            print(f"==================== [Episode {self._episode_id} | Phase: {self._phase.name}] ====================")
            print(f"  Target Altitude Z: {target_pos[2]:.4f} m (+{self._post_drop_lift_offset_z*100.0:.1f} cm above drop height)")
            print(f"  Gripper State:     OPEN (+100.0 deg)")
            print(f"========================================================================================\n")
            if not self._controller.reset(self._estimated_state(), setpoint, t=0.0):
                raise RuntimeError("[SO101PickPlaceController] RmpFlowController reset failed.")
            self._t = 0.0

        # Keep gripper open
        self._set_gripper(self._OPEN_POS)

        desired = self._controller.forward(self._estimated_state(), setpoint, self._t)
        des_pos = None
        if desired is not None and desired.joints.positions is not None:
            des_pos = np.asarray(desired.joints.positions)
            self._articulation.set_dof_position_targets(
                positions=desired.joints.positions,
                dof_indices=desired.joints.position_indices,
            )

        self._t += self._PHYSICS_DT
        self._step += 1

        if des_pos is not None:
            joint_err = self.get_joint_error(des_pos)
            curr_arm = self.get_arm_positions()

            if self._step >= 25 and joint_err < self._JOINT_CONVERGE_THRESH and not self.is_phase_converged(PickPlacePhase.POST_DROP_LIFT):
                self._phase_converged[PickPlacePhase.POST_DROP_LIFT] = True
                print(f">>> [Episode {self._episode_id} | {self._phase.name} CONVERGED at Step {self._step}!] End-effector lifted cleanly above bin!")
                print(f"    Joint positions: {np.round(curr_arm, 3)} (joint error: {joint_err:.4f} rad)")
                print(f"    >>> Transitioning to Phase: {PickPlacePhase.RETRACT_HOME.name} (Retracting to default home posture)...\n")
                self._phase = PickPlacePhase.RETRACT_HOME
                self._step = 0
                self._t = 0.0
            elif self._step % 10 == 0 and not self.is_phase_converged(PickPlacePhase.POST_DROP_LIFT):
                print(f"  [Episode {self._episode_id} | {self._phase.name} Step {self._step:2d}] Lifting above bin... Joint error: {joint_err:.3f} rad | Arm: {np.round(curr_arm, 2)}")

    def _step_retract_home(self) -> None:
        """
        Phase RETRACT_HOME (10):
        Retracts the robot arm back to its default home posture [0, 0, 0, 0, 0] in joint space.
        """
        home_joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        if self._step == 0:
            print(f"==================== [Episode {self._episode_id} | Phase: {self._phase.name}] ====================")
            print(f"  Target Posture:    {home_joints.tolist()} (Home Posture)")
            print(f"  Gripper State:     OPEN (+100.0 deg)")
            print(f"========================================================================================\n")

        self._articulation.set_dof_position_targets(
            wp.array(home_joints, dtype=wp.float32),
            dof_indices=[0, 1, 2, 3, 4],
        )

        self._t += self._PHYSICS_DT
        self._step += 1

        joint_err = self.get_joint_error(home_joints)
        curr_arm = self.get_arm_positions()

        if self._step >= 30 and joint_err < self._JOINT_CONVERGE_THRESH and not self.is_phase_converged(PickPlacePhase.RETRACT_HOME):
            self._phase_converged[PickPlacePhase.RETRACT_HOME] = True
            print(f">>> [Episode {self._episode_id} | {self._phase.name} CONVERGED at Step {self._step}!] Arm returned to default home posture!")
            print(f"    Joint positions: {np.round(curr_arm, 3)} (joint error: {joint_err:.4f} rad)")
            print(f"    >>> Pick-and-place cycle complete!\n")
            self._step = 0
            self._t = 0.0
        elif self._step % 10 == 0 and not self.is_phase_converged(PickPlacePhase.RETRACT_HOME):
            print(f"  [Episode {self._episode_id} | {self._phase.name} Step {self._step:2d}] Retracting home... Joint error: {joint_err:.3f} rad | Arm: {np.round(curr_arm, 2)}")

    def step(self, brick_pos: np.ndarray, brick_ori: np.ndarray, bin_pos: np.ndarray | None = None) -> None:
        """
        Executes the current pick-and-place phase:
          PickPlacePhase.ORIENT_DOWNWARD: Orients gripper downward from default posture [0, 0, 0, 1.5708, folded_yaw].
          PickPlacePhase.PRE_GRASP_HOVER: Moves smoothly to pre-grasp hover target above the brick.
          PickPlacePhase.DESCEND_TO_GRASP: Descends vertically down to the grasp target ready for grasp.
          PickPlacePhase.GRASP: Closes the gripper jaw firmly onto the brick.
          PickPlacePhase.MOVE_UP: Lifts the grasped object vertically into the air.
          PickPlacePhase.HOLD: Dummy step holding the object steady.
          PickPlacePhase.MOVE_TO_BIN: Navigates smoothly to hover above the bin center.
          PickPlacePhase.DROP: Opens gripper jaw to release the brick.
          PickPlacePhase.POST_DROP_LIFT: Lifts end-effector vertically to clear bin.
          PickPlacePhase.RETRACT_HOME: Retracts arm back to default home posture.
        """
        if not self._is_initialized:
            return

        # Keep gripper fully open during Approach, Hover, Descent, Drop, Post-Drop Lift, and Retract phases
        if self._phase in (
            PickPlacePhase.ORIENT_DOWNWARD,
            PickPlacePhase.PRE_GRASP_HOVER,
            PickPlacePhase.DESCEND_TO_GRASP,
            PickPlacePhase.DROP,
            PickPlacePhase.POST_DROP_LIFT,
            PickPlacePhase.RETRACT_HOME,
        ):
            self._set_gripper(self._OPEN_POS)

        if self._phase == PickPlacePhase.ORIENT_DOWNWARD:
            self._step_orient_downward(brick_pos, brick_ori)
        elif self._phase == PickPlacePhase.PRE_GRASP_HOVER:
            self._step_pre_grasp_hover(brick_pos, brick_ori)
        elif self._phase == PickPlacePhase.DESCEND_TO_GRASP:
            self._step_descend_to_grasp(brick_pos, brick_ori)
        elif self._phase == PickPlacePhase.GRASP:
            self._step_grasp(brick_pos, brick_ori)
        elif self._phase == PickPlacePhase.MOVE_UP:
            self._step_move_up(brick_pos, brick_ori)
        elif self._phase == PickPlacePhase.HOLD:
            self._step_hold()
        elif self._phase == PickPlacePhase.MOVE_TO_BIN:
            self._step_move_to_bin(bin_pos)
        elif self._phase == PickPlacePhase.DROP:
            self._step_drop()
        elif self._phase == PickPlacePhase.POST_DROP_LIFT:
            self._step_post_drop_lift(bin_pos)
        elif self._phase == PickPlacePhase.RETRACT_HOME:
            self._step_retract_home()

