"""
Save LIBERO scene meshes to OBJ files for viewing in MeshLab or other 3D viewers.

This script extracts meshes from a LIBERO task and saves them as OBJ files.

Usage:
    python save_meshes_for_viewing.py --task_suite libero_spatial --task_id 0
"""

import dataclasses
import logging
import pathlib

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import ControlEnv
import tyro

import mesh_utils


@dataclasses.dataclass
class Args:
    task_suite: str = "libero_spatial"  # Task suite name
    task_id: int = 0  # Task ID within the suite
    output_dir: str = "output/meshes"  # Output directory for mesh files
    save_all: bool = True  # Save all meshes (robot + objects)
    save_objects_only: bool = True  # Save only object meshes (no robot)
    apply_transform: bool = True  # Apply world transform to mesh vertices (True = world coords, False = local coords)
    include_primitives: bool = True  # Include primitive geometries (box, cylinder, plane) - includes table, walls, floor
    seed: int = 7  # Random seed


def save_meshes(args: Args):
    """Extract and save meshes from LIBERO environment."""
    
    # Initialize task
    logging.info(f"Loading task suite: {args.task_suite}")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    task = task_suite.get_task(args.task_id)
    task_description = task.language
    
    logging.info(f"Task {args.task_id}: {task_description}")
    
    # Create environment with retry logic for randomization errors
    logging.info("Creating environment...")
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
        "has_renderer": False  # No rendering needed for mesh extraction
    }
    
    # Try multiple seeds if placement fails
    max_retries = 10
    env = None
    for attempt in range(max_retries):
        try:
            current_seed = args.seed + attempt
            logging.info(f"Attempt {attempt + 1}/{max_retries} with seed {current_seed}...")
            env = ControlEnv(**env_args)
            env.seed(current_seed)
            env.reset()
            logging.info(f"✓ Environment created successfully with seed {current_seed}")
            break
        except Exception as e:
            if "Cannot place all objects" in str(e):
                logging.warning(f"Placement failed with seed {current_seed}, retrying...")
                continue
            else:
                raise  # Re-raise if it's a different error
    
    if env is None:
        raise RuntimeError(f"Failed to create environment after {max_retries} attempts")
    
    # Extract scene information
    logging.info("\nExtracting scene information...")
    scene_info = mesh_utils.get_mesh_info_and_poses(env)
    
    # Print summary
    mesh_utils.print_scene_summary(scene_info)
    
    # Prepare output directory
    task_name_clean = task_description.replace(" ", "_").replace(",", "")
    output_path = pathlib.Path(args.output_dir) / f"task_{args.task_id}_{task_name_clean}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save meshes
    logging.info(f"\n{'='*80}")
    logging.info(f"Saving meshes to: {output_path}")
    logging.info(f"{'='*80}\n")
    
    if args.save_objects_only:
        # Save only object meshes (no robot parts)
        mode_str = "OBJECTS + PRIMITIVES" if args.include_primitives else "OBJECT MESHES ONLY"
        logging.info(f"Saving {mode_str} (excluding robot)...")
        
        saved_count = 0
        for geom_name, geom_data in scene_info['geom_info'].items():
            # Skip robot parts
            if any(x in geom_name.lower() for x in ['robot', 'gripper', 'mount']):
                continue
            
            position = geom_data['position']
            rotation_matrix = geom_data['rotation_matrix']
            geom_type = geom_data['type']
            
            # Handle mesh geometries
            if geom_data['mesh_id'] is not None:
                mesh_name = geom_data['mesh_name']
                mesh_data = scene_info['meshes'][mesh_name]
                
                obj_file = output_path / f"{geom_name}.obj"
                mesh_utils.export_mesh_to_obj(
                    geom_name,
                    mesh_data['vertices'],
                    mesh_data['faces'],
                    str(obj_file),
                    position=position,
                    rotation_matrix=rotation_matrix,
                    apply_transform=args.apply_transform
                )
                saved_count += 1
            
            # Handle primitives if requested
            elif args.include_primitives and geom_type in ['box', 'cylinder', 'sphere', 'capsule', 'plane']:
                size = geom_data['size']
                prim_mesh = mesh_utils.create_primitive_mesh(geom_type, size)
                
                if prim_mesh is not None:
                    vertices, faces = prim_mesh
                    obj_file = output_path / f"{geom_name}.obj"
                    mesh_utils.export_mesh_to_obj(
                        geom_name,
                        vertices,
                        faces,
                        str(obj_file),
                        position=position,
                        rotation_matrix=rotation_matrix,
                        apply_transform=args.apply_transform
                    )
                    saved_count += 1
        
        logging.info(f"\nSaved {saved_count} geometries")
    
    elif args.save_all:
        # Save all meshes
        mode_str = "ALL GEOMETRIES (meshes + primitives)" if args.include_primitives else "ALL MESHES"
        logging.info(f"Saving {mode_str}...")
        saved_count = mesh_utils.export_all_meshes(
            scene_info, 
            str(output_path), 
            apply_transform=args.apply_transform,
            include_primitives=args.include_primitives
        )
        logging.info(f"\nSaved {saved_count} geometries")
    
    # Also save scene info JSON
    json_file = output_path / "scene_info.json"
    mesh_utils.save_scene_to_json(scene_info, str(json_file))
    
    # Save a separate file with object poses for easy reference
    object_poses = mesh_utils.get_object_mesh_poses(scene_info)
    with open(output_path / "object_list.txt", 'w') as f:
        f.write(f"Task: {task_description}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Objects with meshes ({len(object_poses)} total):\n\n")
        
        # Separate robot and scene objects
        robot_objects = []
        scene_objects = []
        
        for geom_name, obj_data in object_poses.items():
            pos = obj_data['position']
            info = f"  {geom_name}\n"
            info += f"    Mesh: {obj_data['mesh_name']}\n"
            info += f"    Body: {obj_data['body_name']}\n"
            info += f"    Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]\n"
            info += f"    File: {geom_name}.obj\n\n"
            
            if any(x in geom_name.lower() for x in ['robot', 'gripper', 'mount']):
                robot_objects.append(info)
            else:
                scene_objects.append(info)
        
        f.write(f"SCENE OBJECTS ({len(scene_objects)}):\n")
        f.write("-" * 80 + "\n")
        for obj in scene_objects:
            f.write(obj)
        
        f.write(f"\nROBOT PARTS ({len(robot_objects)}):\n")
        f.write("-" * 80 + "\n")
        for obj in robot_objects:
            f.write(obj)
    
    logging.info(f"\n{'='*80}")
    logging.info(f"✓ All files saved to: {output_path}")
    logging.info(f"{'='*80}\n")
    
    # Print instructions
    print("\n" + "="*80)
    print("VIEWING MESHES IN MESHLAB")
    print("="*80)
    print(f"\n1. Open MeshLab")
    print(f"2. File → Import Mesh...")
    print(f"3. Navigate to: {output_path.absolute()}")
    print(f"4. Select one or more .obj files to view")
    print(f"\nGeometry Types:")
    if args.include_primitives:
        print(f"  ✓ Meshes + Primitives (box, cylinder, sphere, plane)")
        print(f"    - Includes table, walls, floor, and collision boxes")
    else:
        print(f"  ✓ Meshes only (primitives like table/floor not included)")
    
    print(f"\nCoordinate System:")
    if args.apply_transform:
        print(f"  ✓ WORLD COORDINATES - Vertices transformed using position + rotation")
        print(f"    Import multiple files and they'll be correctly positioned!")
    else:
        print(f"  ✓ LOCAL COORDINATES - Vertices in object's local frame")
        print(f"    Pose info is in OBJ file comments (open in text editor to see)")
    print(f"\nTip: To view multiple objects together:")
    print(f"  - Import first mesh: File → Import Mesh")
    print(f"  - Add more meshes: File → Import Mesh (again)")
    if args.apply_transform:
        print(f"  - All meshes will appear in correct world positions!")
    print(f"\nFiles saved:")
    print(f"  - Mesh files: {output_path.absolute()}/*.obj")
    print(f"  - Object list: {(output_path / 'object_list.txt').absolute()}")
    print(f"  - Scene data: {(output_path / 'scene_info.json').absolute()}")
    print("="*80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    args = tyro.cli(Args)
    save_meshes(args)

