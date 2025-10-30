"""
Example: Extract mesh files and poses from a LIBERO environment.

This example shows how to use mesh_utils.py to extract scene information
from any LIBERO task.

Usage:
    python example_extract_meshes.py
"""

import logging
import pathlib

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import ControlEnv
import mesh_utils


def main():
    # Configuration
    task_suite_name = "libero_spatial"
    task_id = 0
    resolution = 256
    seed = 7
    output_dir = "data/libero/scene_meshes"
    
    # Initialize task suite
    logging.info(f"Loading task suite: {task_suite_name}")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()
    
    # Get specific task
    task = task_suite.get_task(task_id)
    task_description = task.language
    logging.info(f"Task: {task_description}")
    
    # Initialize environment
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "has_renderer": True
    }
    env = ControlEnv(**env_args)
    env.seed(seed)
    
    # Reset environment to initialize the scene
    logging.info("Resetting environment...")
    env.reset()
    
    # Extract mesh and pose information
    logging.info("\nExtracting scene information...")
    scene_info = mesh_utils.get_mesh_info_and_poses(env)
    
    # Print summary
    mesh_utils.print_scene_summary(scene_info)
    
    # Save to JSON
    task_name_clean = task_description.replace(" ", "_").replace(",", "")
    json_path = f"{output_dir}/{task_name_clean}_scene.json"
    mesh_utils.save_scene_to_json(scene_info, json_path)
    
    # Export all meshes to OBJ files
    meshes_dir = f"{output_dir}/{task_name_clean}_meshes"
    logging.info(f"\nExporting meshes to: {meshes_dir}")
    mesh_utils.export_all_meshes(scene_info, meshes_dir)
    
    # Get simplified object mesh poses (useful for motion planning)
    object_poses = mesh_utils.get_object_mesh_poses(scene_info)
    logging.info(f"\nFound {len(object_poses)} objects with meshes:")
    for obj_name, obj_data in object_poses.items():
        pos = obj_data['position']
        logging.info(f"  {obj_name}: mesh={obj_data['mesh_name']}, pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    logging.info(f"\nAll data saved to: {output_dir}")
    
    # Example: Access specific mesh data
    print("\n" + "="*80)
    print("EXAMPLE: Accessing specific mesh data")
    print("="*80)
    for mesh_name, mesh_data in list(scene_info['meshes'].items())[:3]:  # Show first 3 meshes
        print(f"\nMesh: {mesh_name}")
        print(f"  Vertices shape: {mesh_data['vertices'].shape}")
        print(f"  Faces shape: {mesh_data['faces'].shape}")
        print(f"  First vertex: {mesh_data['vertices'][0]}")
        print(f"  First face: {mesh_data['faces'][0]}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    main()


