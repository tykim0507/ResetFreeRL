import collections
import dataclasses
import logging
import math
import pathlib
import pickle

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv, ControlEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

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
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 1  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 0  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    poses_dir = pathlib.Path("examples/libero/robot_poses")
    poses_dir.mkdir(parents=True, exist_ok=True)

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
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            
            # Track gripper state for detecting grab/release events
            prev_gripper_qpos = None
            grab_pose = None
            release_pose = None

            grab_count = 0
            release_count = 0
            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
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

                    # Track gripper state to detect grab/release events
                    current_gripper_qpos = obs["robot0_gripper_qpos"][0] if len(obs["robot0_gripper_qpos"]) > 0 else obs["robot0_gripper_qpos"]
                    
                    
                    # Detect gripper state changes (grab: close to open, release: open to close)
                    # In LIBERO, gripper_qpos is typically 0 when open and negative when closed
                    if prev_gripper_qpos is not None:
                        gripper_threshold = 0.01  # Threshold to determine if gripper is open or closed
                        
                        # Detect closing (grab event)
                        if prev_gripper_qpos > gripper_threshold and current_gripper_qpos < gripper_threshold:
                            # Gripper closed - grab event
                            grab_pose = {
                                "pos": env.env.sim.data.body_xpos[env.env.sim.model.body_name2id("robot0_link7")].copy(),
                                "quat": env.env.sim.data.body_xquat[env.env.sim.model.body_name2id("robot0_link7")].copy(),
                                "gripper_qpos": current_gripper_qpos.copy(),
                                "qpos": env.env.sim.data.qpos[:9].copy(),
                                "timestamp": t
                            }
                            logging.info(f"Grab detected at step {t}")
                            pickle.dump(grab_pose, open(poses_dir / f"grab_pose_{grab_count}.pkl", "wb"))
                            grab_count += 1
                        # Detect opening (release event)
                        elif prev_gripper_qpos < gripper_threshold and current_gripper_qpos > gripper_threshold:
                            # Gripper opened - release event
                            release_pose = {
                                "pos": env.env.sim.data.body_xpos[env.env.sim.model.body_name2id("robot0_link7")].copy(),
                                "quat": env.env.sim.data.body_xquat[env.env.sim.model.body_name2id("robot0_link7")].copy(),
                                "gripper_qpos": current_gripper_qpos.copy(),
                                "qpos": env.env.sim.data.qpos[:9].copy(),
                                "timestamp": t
                            }
                            logging.info(f"Release detected at step {t}")
                            pickle.dump(release_pose, open(poses_dir / f"release_pose_{release_count}.pkl", "wb"))
                            release_count += 1
                    
                    prev_gripper_qpos = current_gripper_qpos

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

                    # Execute action in environment
                    env.env.render()
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        #save the eef pose at the end of the episode
                        done_pose = {
                            "pos": env.env.sim.data.body_xpos[env.env.sim.model.body_name2id("robot0_link7")].copy(),
                            "quat": env.env.sim.data.body_xquat[env.env.sim.model.body_name2id("robot0_link7")].copy(),
                            "gripper_qpos": obs["robot0_gripper_qpos"][0] if len(obs["robot0_gripper_qpos"]) > 0 else obs["robot0_gripper_qpos"],
                            "qpos": env.env.sim.data.qpos[:9].copy(),
                            "timestamp": t 
                        }
                        pickle.dump(done_pose, open(poses_dir / f"done_pose.pkl", "wb"))
                        breakpoint()
                        task_successes += 1
                        total_successes += 1
                        break
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


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution, "has_renderer": True}
    env = ControlEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
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
