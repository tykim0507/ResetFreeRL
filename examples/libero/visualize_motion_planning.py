"""
Visualize CuRobo motion planning in LIBERO environment.

This script:
1. Creates a LIBERO environment with rendering using JOINT_POSITION controller
2. Sets up CuRobo motion planner
3. Plans a motion to a simple end-effector goal
4. Executes trajectory using env.step() with delta joint position actions
5. Saves a video of the execution

Note: The JOINT_POSITION controller expects delta actions (changes in joint positions).
This script converts absolute joint positions from CuRobo into delta actions.
"""

import dataclasses
import logging
import pathlib

import imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import ControlEnv
import numpy as np
import torch
import tqdm
import tyro
import pickle

from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState

import libero_curobo_utils
import mesh_utils
from curobo.geom.types import Cuboid
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.geom.sphere_fit import SphereFitType


LIBERO_DUMMY_ACTION = [0.0] * 8  # 7 arm joints + 1 gripper
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_spatial"  # Task suite
    task_id: int = 0  # Which task to visualize (0-9 for libero_spatial)
    #################################################################################################################
    # Execution parameters
    #################################################################################################################
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize
    execution_speed: float = 1.0  # Speed multiplier for trajectory execution (1.0 = normal)
    
    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 7  # Random seed
    

def visualize_motion_planning(args: Args) -> None:
    """Visualize motion planning in LIBERO environment."""
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Initialize LIBERO task suite
    logging.info(f"Loading LIBERO task suite: {args.task_suite_name}")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    
    # Get specific task
    task = task_suite.get_task(args.task_id)
    task_description = task.language
    logging.info(f"Task {args.task_id}: {task_description}")
    
    # Create LIBERO environment with rendering
    logging.info("Creating LIBERO environment with rendering...")
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "controller": "JOINT_POSITION",
        "camera_heights": LIBERO_ENV_RESOLUTION,
        "camera_widths": LIBERO_ENV_RESOLUTION,
        "has_renderer": True,  # Enable rendering for visualization
        "ignore_done": True,
    }
    env = ControlEnv(**env_args)
    env.seed(args.seed)
    
    # Reset environment
    logging.info("Resetting environment...")
    obs = env.reset()
    
    # Wait for objects to stabilize
    logging.info(f"Waiting {args.num_steps_wait} steps for objects to stabilize...")
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        env.env.render()
    
    # =========================================================================
    # Setup CuRobo Motion Planner
    # =========================================================================
    logging.info("\nSetting up CuRobo motion planner...")
    tensor_args = TensorDeviceType(device=torch.device("cuda:0"))
    
    motion_gen, robot_config, world_config, base_transform = libero_curobo_utils.setup_libero_curobo_motion_gen(
        env,
        robot_config_base="franka.yml",
        tensor_args=tensor_args,
        exclude_table=False,  # Include table in collision checking
        include_meshes=True,  # Include mesh objects
        # Motion planning parameters
        # trajopt_tsteps=100,
        interpolation_steps=10000,  # Increased from default 5000 to get more trajectory points
        num_ik_seeds=100,
        # num_trajopt_seeds=6,
        # grad_trajopt_iters=500,
        # trajopt_dt=0.5,
        interpolation_dt=0.0005,  # Reduced from 0.02 to get more trajectory points
        evaluate_interpolated_trajectory=True,
        use_cuda_graph=True,
        collision_activation_distance=0.005,  
    )
    
    logging.info("CuRobo setup complete!")
    
    # =========================================================================
    # Get Current Robot State
    # =========================================================================
    logging.info("\nGetting current robot state...")
    current_qpos = libero_curobo_utils.get_current_joint_state(env)
    current_ee_pos, current_ee_quat = libero_curobo_utils.get_ee_pose(env)

    logging.info(f"Current EE position (world): {current_ee_pos}")
    logging.info(f"Current EE quaternion (world): {current_ee_quat}")
    logging.info(f"Current joint positions: {current_qpos[:7]}")
    
    # =========================================================================
    # Define Motion Planning Problem
    # =========================================================================
    logging.info("\nDefining motion planning problem...")
    
    # Start state: current joint positions
    q_start = JointState.from_position(
        tensor_args.to_device([current_qpos[:7].tolist()]),
        joint_names=[
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ],
    )
    
    pose_dir = pathlib.Path("examples/libero/robot_poses")
    grab_pose_dict = pickle.load(open(pose_dir / f"grab_pose_0.pkl", "rb"))
    grab_position_world = grab_pose_dict["pos"]
    grab_quaternion_world = grab_pose_dict["quat"]
    
    done_pose_dict = pickle.load(open(pose_dir / f"done_pose.pkl", "rb"))
    done_position_world = done_pose_dict["pos"]
    done_quaternion_world = done_pose_dict["quat"]
    
    # set goal pose to grab pose
    goal_position_world = grab_position_world
    goal_quaternion_world = grab_quaternion_world

    
    logging.info(f"Goal EE position (world): {goal_position_world}")
    logging.info(f"Goal EE quaternion (world): {goal_quaternion_world}")
    
    # IMPORTANT: Transform goal pose to robot base frame
    goal_position_base, goal_quaternion_base = base_transform.transform_pose_to_base_frame(
        goal_position_world, goal_quaternion_world
    )
    
    logging.info(f"Goal EE position (robot base): {goal_position_base}")
    logging.info(f"Goal EE quaternion (robot base): {goal_quaternion_base}")
    
    goal_pose = Pose(
        position=tensor_args.to_device([goal_position_base.tolist()]),
        quaternion=tensor_args.to_device([goal_quaternion_base.tolist()]),
    )
    
    # =========================================================================
    # Plan Motion
    # =========================================================================
    logging.info("\nPlanning motion...")
    grab_plan_config = MotionGenPlanConfig(
        partial_ik_opt = False, # default value, add more arguments if needed
    )
    result = motion_gen.plan_single(q_start, goal_pose, plan_config=grab_plan_config)
    
    # Get interpolated trajectory
    interpolated_solution = result.get_interpolated_plan()
    trajectory_positions = interpolated_solution.position.cpu().numpy()
    
    logging.info(f"Trajectory length: {len(trajectory_positions)} waypoints")
    logging.info(f"Interpolation dt: {result.interpolation_dt}")
    
    # =========================================================================
    # Execute Trajectory in LIBERO
    # =========================================================================
    logging.info("\nExecuting trajectory in LIBERO...")
    
    replay_images = []
    
    # Sample waypoints based on execution speed
    step_size = max(1, int(1.0 / args.execution_speed))
    sampled_trajectory = trajectory_positions[::step_size]
    logging.info(f"Executing {len(sampled_trajectory)} waypoints (step_size={step_size})...")
    
    for i, joint_pos in enumerate(tqdm.tqdm(sampled_trajectory)):
        # Get current joint state from the environment
        current_qpos_env = libero_curobo_utils.get_current_joint_state(env)
        current_joint_pos = current_qpos_env[:7].copy()
        
        # Convert from absolute joint positions to delta actions for JOINT_POSITION controller
        # The controller expects delta joint positions: new_pos = old_pos + action
        target_joints = np.array(joint_pos)  # 7 joint positions from cuRobo
        delta_joint_actions = target_joints - current_joint_pos
        
        # JOINT_POSITION controller expects 8 actions: 7 arm deltas + 1 gripper delta
        # For gripper, we keep the current state (no change) by using 0.0
        # Create action for JOINT_POSITION controller: [7 arm deltas, 1 gripper delta]
        action = delta_joint_actions.tolist() + [0.0]  # Add gripper delta (0.0 means no change)
        
        _, _, _, _ = env.step(action)
        env.env.render()
    
    # =========================================================================
    # Grasp the Object
    # =========================================================================
    logging.info("\nAttempting to grasp the object...")

    for _ in range(20):
        _, _, _, _ = env.step([0.0] * 7 + [0.001])
        env.env.render()
    # =========================================================================
    # Attach Grasped Object to Robot
    # =========================================================================
    logging.info("\nAttaching grasped object boxes to robot...")
    
    # Get current joint state after grasping
    current_qpos = libero_curobo_utils.get_current_joint_state(env)
    current_joint_state = JointState.from_position(
        tensor_args.to_device([current_qpos[:7].tolist()]),
        joint_names=[
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7",
        ],
    )
    
    # Re-extract scene info to get current object poses (objects may have moved)
    scene_info = mesh_utils.get_mesh_info_and_poses(env)
    
    # Filter for akita_black_bowl_1 related boxes (excluding _g0 since it is style mesh)
    bowl_boxes = []
    for geom_name, geom_data in scene_info['geom_info'].items():
        # Check if this is an akita_black_bowl_1 related box, excluding _g0
        if 'akita_black_bowl_1' in geom_name.lower() and '_g0' not in geom_name.lower():
            # Only include box geometries
            if geom_data.get('type') == 'box':
                # Get current pose
                position = geom_data['position']
                rotation_matrix = geom_data['rotation_matrix']
                quat = libero_curobo_utils.rotation_matrix_to_quaternion(rotation_matrix)
                
                # Transform pose to robot base frame
                position_base, quat_base = base_transform.transform_pose_to_base_frame(position, quat)
                
                # Get dimensions (size is [half_x, half_y, half_z], cuRobo expects full dimensions)
                size = geom_data['size']
                dims = [float(size[0] * 2), float(size[1] * 2), float(size[2] * 2)]
                
                pose = [
                    float(position_base[0]), float(position_base[1]), float(position_base[2]),
                    float(quat_base[0]), float(quat_base[1]), float(quat_base[2]), float(quat_base[3])
                ]
                
                cuboid_obstacle = Cuboid(
                    name=geom_name,
                    pose=pose,
                    dims=dims
                )
                bowl_boxes.append(cuboid_obstacle)
                logging.info(f"Found bowl box: {geom_name}")
    
    if len(bowl_boxes) > 0:
        logging.info(f"Attaching {len(bowl_boxes)} bowl box(es) to robot...")
        success = motion_gen.attach_external_objects_to_robot(
            joint_state=current_joint_state,
            external_objects=bowl_boxes,
            link_name="attached_object",
            world_objects_pose_offset=Pose.from_list([0.0, 0.0, 0.01, 1.0, 0.0, 0.0, 0.0], tensor_args=tensor_args)
        )
        if success:
            logging.info("✓ Successfully attached bowl boxes to robot!")
        else:
            logging.warning("⚠ Failed to attach bowl boxes to robot")
    else:
        logging.warning("⚠ No akita_black_bowl_1 boxes found (excluding _g0)")
    # =========================================================================
    # Move to Done Pose
    # =========================================================================
    logging.info("\nMoving to done pose...")
    
    # Use the done pose from the pickle file
    logging.info(f"Target EE position (done pose): {done_position_world}")
    
    # Transform done pose to robot base frame
    done_position_base, done_quaternion_base = base_transform.transform_pose_to_base_frame(
        done_position_world, done_quaternion_world
    )
    
    logging.info(f"Done pose position (robot base): {done_position_base}")
    logging.info(f"Done pose quaternion (robot base): {done_quaternion_base}")
    
    done_pose = Pose(
        position=tensor_args.to_device([done_position_base.tolist()]),
        quaternion=tensor_args.to_device([done_quaternion_base.tolist()]),
    )
    
    # Get current joint state for planning
    current_qpos = libero_curobo_utils.get_current_joint_state(env)
    q_start = JointState.from_position(
        tensor_args.to_device([current_qpos[:7].tolist()]),
        joint_names=[
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7",
        ],
    )

    # Reset motion planner state before planning to done pose
    logging.info("Resetting motion planner state...")
    motion_gen.reset(reset_seed=False)
    
    # Plan motion to done pose
    logging.info("Starting motion planning to done pose...")
    logging.info(f"Start state: {q_start.position.cpu().numpy()}")
    logging.info(f"Goal pose position: {done_pose.position.cpu().numpy()}")
    logging.info(f"Goal pose quaternion: {done_pose.quaternion.cpu().numpy()}")
    
    done_plan_config = MotionGenPlanConfig(
        partial_ik_opt = True,
    )
    
    
    done_result = motion_gen.plan_single(q_start, done_pose, plan_config=done_plan_config)
    
    # Execute the planned trajectory to done pose
    interpolated_solution = done_result.get_interpolated_plan()
    trajectory_positions = interpolated_solution.position.cpu().numpy()
    
    logging.info(f"Executing {len(trajectory_positions)} waypoints to done pose...")
    
    # Sample waypoints for execution
    step_size = max(1, int(1.0 / args.execution_speed))
    sampled_trajectory = trajectory_positions[::step_size]
    
    for i, joint_pos in enumerate(tqdm.tqdm(sampled_trajectory, desc="Moving to done pose")):
        # Get current joint state from the environment
        current_qpos_env = libero_curobo_utils.get_current_joint_state(env)
        current_joint_pos = current_qpos_env[:7].copy()
        target_joints = np.array(joint_pos)
        delta_joint_actions = target_joints - current_joint_pos
        action = delta_joint_actions.tolist() + [0.0]
        _, _, _, _ = env.step(action)
        env.env.render()
            
    
    logging.info("✓ Successfully moved to done pose!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    tyro.cli(visualize_motion_planning)
