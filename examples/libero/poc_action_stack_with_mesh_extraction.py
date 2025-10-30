"""
Modified version of poc_action_stack.py that extracts mesh and pose information.

This demonstrates how to integrate mesh extraction into your existing evaluation loop.
The mesh extraction happens once per task before the episode loop begins.
"""

import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv, ControlEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

# Import mesh extraction utilities
import mesh_utils

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 2  # Number of rollouts per task
    disable_gravity: bool = False  # Set to True to disable gravity in the simulation

    #################################################################################################################
    # Mesh extraction parameters
    #################################################################################################################
    extract_meshes: bool = True  # Extract mesh files and poses
    mesh_output_dir: str = "data/libero/scene_meshes"  # Where to save mesh data

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed, args.disable_gravity)
        
        # ============================================================================
        # MESH EXTRACTION: Extract mesh files and poses once per task
        # ============================================================================
        if args.extract_meshes:
            logging.info(f"\n{'='*80}")
            logging.info(f"Extracting mesh and pose information for task {task_id}")
            logging.info(f"{'='*80}")
            
            # Reset environment to initialize scene
            env.reset()
            
            # Extract scene information
            scene_info = mesh_utils.get_mesh_info_and_poses(env)
            
            # Print summary
            mesh_utils.print_scene_summary(scene_info)
            
            # Save to files
            task_name_clean = task_description.replace(" ", "_").replace(",", "")
            
            # Save JSON with all scene data
            json_path = f"{args.mesh_output_dir}/task_{task_id}_{task_name_clean}_scene.json"
            mesh_utils.save_scene_to_json(scene_info, json_path)
            
            # Export meshes to OBJ files
            meshes_dir = f"{args.mesh_output_dir}/task_{task_id}_{task_name_clean}_meshes"
            mesh_utils.export_all_meshes(scene_info, meshes_dir)
            
            logging.info(f"{'='*80}\n")
        # ============================================================================
        
        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()
            action_history = []  # Stack to store executed actions for reverse playback
            
            # Apply gravity settings AFTER reset/init_state (in case they get reset)
            if args.disable_gravity:
                env.env.sim.model.opt.gravity[:] = [0, 0, 0]
                
                # Zero out all velocities to prevent objects from floating away
                state = env.env.sim.get_state()
                state.qvel[:] = 0.0  # Set all joint/object velocities to zero
                env.env.sim.set_state(state)
                env.env.sim.forward()  # Important: propagate the changes through the simulation
                
                logging.info(f"Gravity disabled and all velocities zeroed: {env.env.sim.model.opt.gravity}")
            
            # Save initial robot state (qpos and qvel)
            # For Panda robot: 7 arm joints + 2 gripper joints = 9 DOF
            sim_state = env.env.sim.get_state()
            initial_robot_qpos = sim_state.qpos[:9].copy()
            initial_robot_qvel = sim_state.qvel[:9].copy()

            # Setup
            t = 0
            replay_images = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall (or stabilize if gravity is off)
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        
                        # If gravity is disabled, continuously zero velocities to keep objects still
                        if args.disable_gravity:
                            state = env.env.sim.get_state()
                            state.qvel[:] = 0.0  # Keep all velocities at zero
                            env.env.sim.set_state(state)
                            env.env.sim.forward()
                        
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    
                    # Store action for reverse playback (before executing it)
                    action_history.append(action.copy())

                    # Execute action in environment
                    env.env.render()
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        # Clear action plan for the reverse task
                        action_plan.clear()
                        
                        logging.info(f"Forward task complete! Replaying {len(action_history)} actions in reverse...")
                        
                        # Reverse the action history to play back in reverse order
                        reversed_actions = action_history[::-1]
                        
                        for reverse_idx, forward_action in enumerate(reversed_actions):
                            # Reverse the process by negating delta actions
                            # LIBERO actions: [dx, dy, dz, drotx, droty, drotz, gripper]
                            # - First 6 dims are deltas (position + orientation) -> negate to reverse
                            # - Last dim is gripper (absolute position) -> keep as-is or invert
                            reverse_action = forward_action.copy()
                            reverse_action[:6] = -forward_action[:6]  # Negate position and orientation deltas
                            # Keep gripper action the same (or you could invert: 1.0 - forward_action[6])
                            
                            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                            img = image_tools.convert_to_uint8(
                                image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                            )
                            wrist_img = image_tools.convert_to_uint8(
                                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                            )

                            # Save preprocessed image for replay video
                            replay_images.append(img)

                            # Execute reversed action in environment
                            env.env.render()
                            obs, reward, _, info = env.step(reverse_action.tolist())
                            
                            if reverse_idx % 10 == 0:
                                logging.info(f"Reverse progress: {reverse_idx + 1}/{len(reversed_actions)}")
                        
                        logging.info("Reverse playback complete!")
                        task_successes += 1
                        total_successes += 1
                        break  # Exit the timestep loop
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed, disable_gravity=False):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution, "has_renderer": True}
    env = ControlEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    
    # Disable gravity if requested
    if disable_gravity:
        # Access the MuJoCo simulation and set gravity to zero
        # Default gravity is [0, 0, -9.81] in MuJoCo
        env.env.sim.model.opt.gravity[:] = [0, 0, 0]
        
        # Increase damping to help stabilize objects (prevents floating)
        # This adds resistance to motion in all directions
        env.env.sim.model.dof_damping[:] = 10.0  # Increase damping for all degrees of freedom
        
        env.env.sim.forward()  # Propagate changes through simulation
        logging.info(f"Gravity initially disabled: {env.env.sim.model.opt.gravity}")
        logging.info(f"Damping increased to prevent floating")
    
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)


