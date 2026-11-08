"""
Pick-and-place controller for the SO-101 manipulator using cuMotion RMPflow.
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
from envs.constants import LEGO_BRICK_HALF_EXTENTS


class PickPlacePhase(IntEnum):
    """
    Named execution phases for the SO-101 pick-and-place state machine:
      - INITIAL_POSITION (1): Pitches wrist downward to 90 deg [0, 0, 0, 1.5708, 0].
      - MOVE_ON_TOP_OF_BRICK (2): Moves directly on top of the brick (+25cm hover).
      - ORIENT_TO_BRICK_POS (3): Orients gripper to grasp target pose beside the brick.
      - LOWER_DOWNWARD_TO_GRASP (4): Lowers gripper vertically down to brick center height.
      - GRASP (5): Closes gripper jaw firmly onto the brick.
      - MOVE_TO_BIN (6): Navigates smoothly to hover on top of the bin center (+Z).
      - HOLD (7): Holds the grasped object steady above the bin center for stability before release.
      - DROP (8): Opens gripper jaw to release the brick into the bin.
      - RETRACT_HOME (9): Returns the robot arm to its default home posture [0, 0, 0, 0, 0].
    """
    INITIAL_POSITION = 1
    MOVE_ON_TOP_OF_BRICK = 2
    ORIENT_TO_BRICK_POS = 3
    LOWER_DOWNWARD_TO_GRASP = 4
    GRASP = 5
    MOVE_TO_BIN = 6
    HOLD = 7
    DROP = 8
    RETRACT_HOME = 9


class SO101PickPlaceController:
    _GRIPPER_JOINT_NAME: str = "gripper"
    _TOOL_FRAME_NAME: str = "gripper_frame_link"

    # Gripper limits in radians: -10 deg close, 100 deg open
    _OPEN_POS: float = 1.2
    _CLOSED_POS: float = -0.05

    _PRE_GRASP_OFFSET_Z: float = 0.08   # 8 cm above brick
    _BIN_OFFSET_Z: float = 0.20         # 20 cm vertical offset (+Z) on top of bin center
    _MIN_STEPS: int = 30                # Minimum frames before declaring convergence
    _JOINT_CONVERGE_THRESH: float = 0.05 # 0.05 rad (~2.8 deg) tracking error
    _PHYSICS_DT: float = 1.0 / 60.0

    def __init__(
        self,
        articulation: Articulation,
        world_interface: SO101WorldInterface,
        config_dir: str | None = None,
        robot_prim_path: str = "/World/so101",
        half_width: float | None = None,
        closed_pos: float = _CLOSED_POS,
        gripper_max_effort: float = 0.5,
        bin_offset_z: float | None = None,
    ):
        self._articulation = articulation
        self._world_interface = world_interface
        self._robot_prim_path = robot_prim_path
        self._closed_pos = float(closed_pos)
        self._gripper_max_effort = float(gripper_max_effort)
        self._bin_offset_z = (
            float(bin_offset_z) if bin_offset_z is not None else self._BIN_OFFSET_Z
        )

        if config_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_dir = os.path.join(project_root, "assets", "config")
        self._config_dir = config_dir

        self._cumotion_robot = None
        self._controller = None
        self._site_space = None
        self._finger_idx = None

        self._phase: PickPlacePhase = PickPlacePhase.INITIAL_POSITION
        self._step: int = 0
        self._t: float = 0.0
        self._is_initialized: bool = False
        self._phase_converged: dict[PickPlacePhase, bool] = {
            PickPlacePhase.INITIAL_POSITION: False,
            PickPlacePhase.MOVE_ON_TOP_OF_BRICK: False,
            PickPlacePhase.ORIENT_TO_BRICK_POS: False,
            PickPlacePhase.LOWER_DOWNWARD_TO_GRASP: False,
            PickPlacePhase.GRASP: False,
            PickPlacePhase.MOVE_TO_BIN: False,
            PickPlacePhase.HOLD: False,
            PickPlacePhase.DROP: False,
            PickPlacePhase.RETRACT_HOME: False,
        }
        self._episode_id: int = 0
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
        """Resets the state machine back to PickPlacePhase.INITIAL_POSITION."""
        if episode_id is not None:
            self._episode_id = episode_id
        elif self._episode_id == 0:
            self._episode_id = 1
        else:
            self._episode_id += 1
        self._phase = PickPlacePhase.INITIAL_POSITION
        self._step = 0
        self._t = 0.0
        self._phase_converged = {
            PickPlacePhase.INITIAL_POSITION: False,
            PickPlacePhase.MOVE_ON_TOP_OF_BRICK: False,
            PickPlacePhase.ORIENT_TO_BRICK_POS: False,
            PickPlacePhase.LOWER_DOWNWARD_TO_GRASP: False,
            PickPlacePhase.GRASP: False,
            PickPlacePhase.MOVE_TO_BIN: False,
            PickPlacePhase.HOLD: False,
            PickPlacePhase.DROP: False,
            PickPlacePhase.RETRACT_HOME: False,
        }
        self._grasp_lift_target = None
        self._grasp_q_target = None
        self._set_gripper(self._OPEN_POS)

    def compute_gripper_pos_ori_for_grasp(
        self,
        brick_pos: np.ndarray,
        brick_ori: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        w, x, y, z = brick_ori
        rot_matrix = R.from_quat([x, y, z, w]).as_matrix()
        b_x = rot_matrix[:, 0]
        b_y = rot_matrix[:, 1]
        b_z = rot_matrix[:, 2]

        x_gripper = 0.0
        y_gripper = 0.0
        z_gripper = 0.0

        # ----------------------- Calculate gripper orientation -----------------------

        # Check if the brick's Z face is facing downward
        robot_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        if abs(np.dot(b_z, robot_z)) > 0.90:
            # If the brick's Z-axis is mostly aligned with the robot's Z-axis, use it as the primary axis
            primary_axis, ref_axis = b_z, b_x
        elif abs(np.dot(b_y, robot_z)) > 0.90:
            # If the brick's Y-axis is mostly aligned with the robot's Z-axis, use it as the primary axis
            primary_axis, ref_axis = b_y, b_z
        else:
            # Otherwise, use the brick's X-axis as the primary axis
            primary_axis, ref_axis = b_x, b_z

        v_desired = -robot_z
        
        z_gripper = primary_axis if np.dot(primary_axis, v_desired) > 0.0 else -primary_axis
        ref_x = -ref_axis if np.dot(ref_axis, [-1.0, 0.0, 0.0]) < 0.0 else ref_axis
        
        y_gripper = np.cross(z_gripper, ref_x)
        y_gripper /= np.linalg.norm(y_gripper)
        x_gripper = np.cross(y_gripper, z_gripper)
        x_gripper /= np.linalg.norm(x_gripper)

        rot_gripper_mat = np.column_stack((x_gripper, y_gripper, z_gripper))
        qx, qy, qz, qw = R.from_matrix(rot_gripper_mat).as_quat()

        q_target = np.array([qw, qx, qy, qz], dtype=np.float32)

        # ----------------------- Calculate gripper position -----------------------

        # Pinch thickness (half-extent of the front/side face along x_gripper):
        # Find which brick local axis is aligned with x_gripper:
        pinch_half_extent = float(np.abs(rot_matrix.T @ x_gripper) @ LEGO_BRICK_HALF_EXTENTS)

        # GRASP TARGET (Stationary jaw resting at the front face at brick mid-height):
        clearance = 0.008
        grasp_pos = brick_pos + (pinch_half_extent + clearance) * x_gripper

        # HOVER / INITIAL TARGET (Same X, Y at the front face, but lifted safely above the brick in Z):
        top_face_extent = float(np.abs(rot_matrix.T @ z_gripper) @ LEGO_BRICK_HALF_EXTENTS)
        hover_height = top_face_extent + 0.05
        target_pos = grasp_pos - hover_height * z_gripper

        return np.array(target_pos, dtype=np.float32), q_target

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

    def _step_initial_position(self, brick_pos: np.ndarray, brick_ori: np.ndarray) -> None:
        """
        Phase INITIAL_POSITION (1):
        Pitches the gripper down [0, 0, 0, 1.5708, 0].
        """
        target_joints = np.array([0.0, 0.0, 0.0, 1.5708, 0.0], dtype=np.float32)

        if self._step == 0:
            print(f"\n==================== [Episode {self._episode_id} | Phase: {self._phase.name}] ====================")
            print(f"  Target Joints (Downward): {np.round(target_joints, 4).tolist()} (Pitch DOWN 90 deg)")
            print(f"  Gripper Jaw State:        FULL OPEN (+100.0 deg)")
            print(f"========================================================================================\n")

        self._articulation.set_dof_position_targets(
            wp.array(target_joints, dtype=wp.float32),
            dof_indices=[0, 1, 2, 3, 4],
        )

        self._t += self._PHYSICS_DT
        self._step += 1

        joint_err = self.get_joint_error(target_joints)
        curr_arm = self.get_arm_positions()

        if (self._step >= 25 and joint_err < 0.06) or self._step >= 35:
            self._phase_converged[PickPlacePhase.INITIAL_POSITION] = True
            print(f">>> [Episode {self._episode_id} | {self._phase.name} CONVERGED at Step {self._step}!] Gripper oriented downward at initial position.")
            print(f"    Joint positions: {np.round(curr_arm, 3)} (joint error: {joint_err:.4f} rad)")
            print(f"    >>> Transitioning to Phase: {PickPlacePhase.MOVE_ON_TOP_OF_BRICK.name} (Moving on top of brick)...\n")
            self._phase = PickPlacePhase.MOVE_ON_TOP_OF_BRICK
            self._step = 0
            self._t = 0.0

    def _step_move_on_top_of_brick(self, brick_pos: np.ndarray, brick_ori: np.ndarray) -> None:
        """
        Phase MOVE_ON_TOP_OF_BRICK (2):
        Moves the gripper directly on top of the brick with height, oriented downward.
        """
        high_z_offset = 0.25  # meters above the brick
        target_pos = np.array([brick_pos[0], brick_pos[1], brick_pos[2] + high_z_offset], dtype=np.float32)

        rot_down = R.from_euler("y", np.pi)  # 180 deg pitch points gripper straight down (-Z)
        qx, qy, qz, qw = rot_down.as_quat()
        q_target = np.array([qw, qx, qy, qz], dtype=np.float32)

        if self._step == 0:
            print(f"\n==================== [Episode {self._episode_id} | Phase: {self._phase.name}] ====================")
            print(f"  Target Pose (On Top of Brick): [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}] (Height +{high_z_offset*100:.0f}cm)")
            print(f"  Gripper Jaw State:             FULL OPEN (+100.0 deg)")
            print(f"========================================================================================\n")

            self._current_setpoint = self._make_setpoint(target_pos, q_target)

            if not self._controller.reset(self._estimated_state(), self._current_setpoint, t=0.0):
                raise RuntimeError("[SO101PickPlaceController] RmpFlowController reset failed.")
            self._t = 0.0

        desired = self._controller.forward(self._estimated_state(), self._current_setpoint, self._t)
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
            if self._step >= 25 and joint_err < self._JOINT_CONVERGE_THRESH:
                self._phase_converged[PickPlacePhase.MOVE_ON_TOP_OF_BRICK] = True
                print(f">>> [Episode {self._episode_id} | {self._phase.name} CONVERGED at Step {self._step}!] Gripper reached top of brick.")
                print(f"    >>> Transitioning to Phase: {PickPlacePhase.ORIENT_TO_BRICK_POS.name} (Orienting to brick position)...\n")
                self._phase = PickPlacePhase.ORIENT_TO_BRICK_POS
                self._step = 0
                self._t = 0.0

    def _step_orient_to_brick_pos(self, brick_pos: np.ndarray, brick_ori: np.ndarray) -> None:
        """
        Phase ORIENT_TO_BRICK_POS (3):
        Orients the gripper to the grasp target pose beside the brick.
        """
        
        if self._step == 0:
            
            self._grasp_lift_target, self._grasp_q_target = self.compute_gripper_pos_ori_for_grasp(brick_pos, brick_ori)

            self._current_setpoint = self._make_setpoint(self._grasp_lift_target, self._grasp_q_target)
            
            if not self._controller.reset(self._estimated_state(), self._current_setpoint, t=0.0):
                raise RuntimeError("[SO101PickPlaceController] RmpFlowController reset failed.")
            
            self._t = 0.0

            print(f"==================== [Episode {self._episode_id} | Phase: {self._phase.name}] ====================")
            print(f"  Target Pose:           [{self._grasp_lift_target[0]:.4f}, {self._grasp_lift_target[1]:.4f}, {self._grasp_lift_target[2]:.4f}]")
            print(f"  Gripper Jaw State:     FULL OPEN (+100.0 deg)")
            print(f"========================================================================================\n")

        desired = self._controller.forward(self._estimated_state(), self._current_setpoint, self._t)
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

            if self._step >= 25 and joint_err < self._JOINT_CONVERGE_THRESH and not self.is_phase_converged(PickPlacePhase.ORIENT_TO_BRICK_POS):
                self._phase_converged[PickPlacePhase.ORIENT_TO_BRICK_POS] = True
                print(f">>> [Episode {self._episode_id} | {self._phase.name} CONVERGED at Step {self._step}!] Gripper oriented to brick grasp target!")
                print(f"    Joint positions: {np.round(curr_arm, 3)} (joint error: {joint_err:.4f} rad)")
                print(f"    >>> Transitioning to Phase: {PickPlacePhase.LOWER_DOWNWARD_TO_GRASP.name} (Lowering downward to grasp)...\n")
                self._phase = PickPlacePhase.LOWER_DOWNWARD_TO_GRASP
                self._step = 0
                self._t = 0.0
            elif self._step % 10 == 0 and not self.is_phase_converged(PickPlacePhase.ORIENT_TO_BRICK_POS):
                print(f"  [Episode {self._episode_id} | {self._phase.name} Step {self._step:2d}] Orienting to brick position... Joint error: {joint_err:.3f} rad | Arm: {np.round(curr_arm, 2)}")

    def _step_lower_downward_to_grasp(self, brick_pos: np.ndarray, brick_ori: np.ndarray) -> None:
        """
        Phase LOWER_DOWNWARD_TO_GRASP (4):
        Lowers the gripper to center from the previous position.
        """

        if self._step == 0:
            self._grasp_lift_target[2] = brick_pos[2]  # Update grasp lift target to center Z for lowering 

            self._current_setpoint = self._make_setpoint(self._grasp_lift_target, self._grasp_q_target)
            
            if not self._controller.reset(self._estimated_state(), self._current_setpoint, t=0.0):
                raise RuntimeError("[SO101PickPlaceController] RmpFlowController reset failed.")
            
            self._t = 0.0

            print(f"==================== [Episode {self._episode_id} | Phase: {self._phase.name}] ====================")
            print(f"  Target Pose (Center):  [{self._grasp_lift_target[0]:.4f}, {self._grasp_lift_target[1]:.4f}, {self._grasp_lift_target[2]:.4f}]")
            print(f"  Gripper Jaw State:     FULL OPEN (+100.0 deg)")
            print(f"========================================================================================\n")

        desired = self._controller.forward(self._estimated_state(), self._current_setpoint, self._t)
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

            if self._step >= 25 and joint_err < self._JOINT_CONVERGE_THRESH and not self.is_phase_converged(PickPlacePhase.LOWER_DOWNWARD_TO_GRASP):
                self._phase_converged[PickPlacePhase.LOWER_DOWNWARD_TO_GRASP] = True
                print(f">>> [Episode {self._episode_id} | {self._phase.name} CONVERGED at Step {self._step}!] Gripper lowered to center position!")
                print(f"    Joint positions: {np.round(curr_arm, 3)} (joint error: {joint_err:.4f} rad)")
                print(f"    >>> Transitioning to Phase: {PickPlacePhase.GRASP.name} (Closing gripper onto brick)...\n")
                self._phase = PickPlacePhase.GRASP
                self._step = 0
                self._t = 0.0
            elif self._step % 10 == 0 and not self.is_phase_converged(PickPlacePhase.LOWER_DOWNWARD_TO_GRASP):
                print(f"  [Episode {self._episode_id} | {self._phase.name} Step {self._step:2d}] Lowering to center... Joint error: {joint_err:.3f} rad | Arm: {np.round(curr_arm, 2)}")

    def _step_grasp(self, brick_pos: np.ndarray, brick_ori: np.ndarray) -> None:
        """
        Phase GRASP (5):
        Closes the gripper jaw firmly onto the object while maintaining the arm at the grasp target.
        """
        if self._step == 0:
            print(f"==================== [Episode {self._episode_id} | Phase: {self._phase.name}] ====================")
            print(f"  Target:             Red Lego Brick")
            print(f"  Gripper Jaw Target: {np.degrees(self._closed_pos):.1f} deg ({self._closed_pos:.3f} rad)")
            print(f"  Gripper Max Effort: {self._gripper_max_effort:.1f} N*m")
            print(f"========================================================================================\n")

        # Drive gripper to closed position tailored to object dimension
        self._set_gripper(self._closed_pos)

        self._t += self._PHYSICS_DT
        self._step += 1

        dofs = self._articulation.get_dof_positions()
        dofs_np = (
            dofs.cpu().numpy()
            if hasattr(dofs, "cpu")
            else (dofs.numpy() if hasattr(dofs, "numpy") else np.asarray(dofs))
        )
        grip_angle = float(dofs_np.flatten()[self._finger_idx]) if self._finger_idx is not None else 0.0

        # Gripper closes over ~20-25 frames achieving snug contact
        if (self._step >= 25 or abs(grip_angle - self._closed_pos) < 0.03) and not self.is_phase_converged(PickPlacePhase.GRASP):
            self._phase_converged[PickPlacePhase.GRASP] = True
            print(f">>> [Episode {self._episode_id} | {self._phase.name} CONVERGED at Step {self._step}!] Object firmly gripped!")
            print(f"    Gripper joint angle: {np.degrees(grip_angle):.1f} deg ({grip_angle:.3f} rad)")
            print(f"    >>> Transitioning to Phase: {PickPlacePhase.MOVE_TO_BIN.name} (Moving on top of bin center)...\n")
            self._phase = PickPlacePhase.MOVE_TO_BIN
            self._step = 0
            self._t = 0.0
        elif self._step % 10 == 0 and not self.is_phase_converged(PickPlacePhase.GRASP):
            print(f"  [Episode {self._episode_id} | {self._phase.name} Step {self._step:2d}] Gripper closing... Angle: {np.degrees(grip_angle):.1f} deg")

    def _step_hold(self) -> None:
        """
        Phase HOLD (7):
        Holds the grasped object steady above the bin center for stability before release.
        """
        if self._step == 0:
            print(f"==================== [Episode {self._episode_id} | Phase: {self._phase.name}] ====================")
            print(f"  Target:            Holding steady above bin center")
            print(f"  Gripper Jaw State: CLOSED ({np.degrees(self._closed_pos):.1f} deg)")
            print(f"========================================================================================\n")

        # Keep gripper firmly closed around the object
        self._set_gripper(self._closed_pos)

        # Actively maintain position setpoint with controller
        if self._current_setpoint is not None:
            desired = self._controller.forward(self._estimated_state(), self._current_setpoint, self._t)
            if desired is not None and desired.joints.positions is not None:
                self._articulation.set_dof_position_targets(
                    positions=desired.joints.positions,
                    dof_indices=desired.joints.position_indices,
                )

        self._t += self._PHYSICS_DT
        self._step += 1

        if self._step >= 30 and not self.is_phase_converged(PickPlacePhase.HOLD):
            self._phase_converged[PickPlacePhase.HOLD] = True
            print(f">>> [Episode {self._episode_id} | {self._phase.name} CONVERGED at Step {self._step}!] Held steady above bin.")
            print(f"    >>> Transitioning to Phase: {PickPlacePhase.DROP.name} (Releasing object into bin)...\n")
            self._phase = PickPlacePhase.DROP
            self._step = 0
            self._t = 0.0
        elif self._step % 10 == 0 and not self.is_phase_converged(PickPlacePhase.HOLD):
            print(f"  [Episode {self._episode_id} | {self._phase.name} Step {self._step:2d}] Holding object above bin... Arm: {np.round(self.get_arm_positions(), 2)}")

    def _step_move_to_bin(self, bin_pos: np.ndarray | None = None) -> None:
        """
        Phase MOVE_TO_BIN (6):
        Moves the grasped object with a +Z offset on top of the bin center using cuMotion RMPflow.
        """
        if bin_pos is None:
            bin_pos = np.array([0.25, 0.12, 0.05], dtype=np.float32)

        bin_center = np.asarray(bin_pos, dtype=np.float32).flatten()[:3]
        target_pos = np.array(
            [bin_center[0], bin_center[1], bin_center[2] + self._bin_offset_z],
            dtype=np.float32,
        )

        if self._grasp_q_target is not None:
            q_target = self._grasp_q_target
        else:
            rot_down = R.from_euler("y", np.pi)
            qx, qy, qz, qw = rot_down.as_quat()
            q_target = np.array([qw, qx, qy, qz], dtype=np.float32)

        if self._step == 0:
            print(f"==================== [Episode {self._episode_id} | Phase: {self._phase.name}] ====================")
            print(f"  Target:            Bin Center (+Z offset: +{self._bin_offset_z*100.0:.1f} cm)")
            print(f"  Bin Center Pos:    [{bin_center[0]:.4f}, {bin_center[1]:.4f}, {bin_center[2]:.4f}]")
            print(f"  Target Pose:       [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
            print(f"  Gripper Jaw State: CLOSED ({np.degrees(self._closed_pos):.1f} deg)")
            print(f"========================================================================================\n")

            self._current_setpoint = self._make_setpoint(target_pos, q_target)

            if not self._controller.reset(self._estimated_state(), self._current_setpoint, t=0.0):
                raise RuntimeError("[SO101PickPlaceController] RmpFlowController reset failed.")
            self._t = 0.0

        # Keep gripper firmly closed around the object
        self._set_gripper(self._closed_pos)

        desired = self._controller.forward(self._estimated_state(), self._current_setpoint, self._t)
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

            if self._step >= 25 and joint_err < self._JOINT_CONVERGE_THRESH and not self.is_phase_converged(PickPlacePhase.MOVE_TO_BIN):
                self._phase_converged[PickPlacePhase.MOVE_TO_BIN] = True
                print(f">>> [Episode {self._episode_id} | {self._phase.name} CONVERGED at Step {self._step}!] End-effector moved to bin center successfully!")
                print(f"    Joint positions: {np.round(curr_arm, 3)} (joint error: {joint_err:.4f} rad)")
                print(f"    >>> Transitioning to Phase: {PickPlacePhase.HOLD.name} (Holding object steady above bin)...\n")
                self._phase = PickPlacePhase.HOLD
                self._step = 0
                self._t = 0.0
            elif self._step % 10 == 0 and not self.is_phase_converged(PickPlacePhase.MOVE_TO_BIN):
                print(f"  [Episode {self._episode_id} | {self._phase.name} Step {self._step:2d}] Moving on top of bin center... Joint error: {joint_err:.3f} rad | Arm: {np.round(curr_arm, 2)}")

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
            print(f"    >>> Transitioning to Phase: {PickPlacePhase.RETRACT_HOME.name} (Retracting to default home posture)...\n")
            self._phase = PickPlacePhase.RETRACT_HOME
            self._step = 0
            self._t = 0.0
        elif self._step % 10 == 0 and not self.is_phase_converged(PickPlacePhase.DROP):
            print(f"  [Episode {self._episode_id} | {self._phase.name} Step {self._step:2d}] Releasing object... Arm: {np.round(self.get_arm_positions(), 2)}")

    def _step_retract_home(self) -> None:
        """
        Phase RETRACT_HOME (9):
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
        if not self._is_initialized:
            return

        # Keep gripper fully open during Initial Position, Move On Top, Orient To Brick Pos, Lower Downward To Grasp, Drop, and Retract phases
        if self._phase in (
            PickPlacePhase.INITIAL_POSITION,
            PickPlacePhase.MOVE_ON_TOP_OF_BRICK,
            PickPlacePhase.ORIENT_TO_BRICK_POS,
            PickPlacePhase.LOWER_DOWNWARD_TO_GRASP,
            PickPlacePhase.DROP,
            PickPlacePhase.RETRACT_HOME,
        ):
            self._set_gripper(self._OPEN_POS)

        if self._phase == PickPlacePhase.INITIAL_POSITION:
            self._step_initial_position(brick_pos, brick_ori)
        elif self._phase == PickPlacePhase.MOVE_ON_TOP_OF_BRICK:
            self._step_move_on_top_of_brick(brick_pos, brick_ori)
        elif self._phase == PickPlacePhase.ORIENT_TO_BRICK_POS:
            self._step_orient_to_brick_pos(brick_pos, brick_ori)
        elif self._phase == PickPlacePhase.LOWER_DOWNWARD_TO_GRASP:
            self._step_lower_downward_to_grasp(brick_pos, brick_ori)
        elif self._phase == PickPlacePhase.GRASP:
            self._step_grasp(brick_pos, brick_ori)
        elif self._phase == PickPlacePhase.MOVE_TO_BIN:
            self._step_move_to_bin(bin_pos)
        elif self._phase == PickPlacePhase.HOLD:
            self._step_hold()
        elif self._phase == PickPlacePhase.DROP:
            self._step_drop()
        elif self._phase == PickPlacePhase.RETRACT_HOME:
            self._step_retract_home()

