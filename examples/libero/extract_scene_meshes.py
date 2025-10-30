"""
Script to extract mesh files and their poses from LIBERO environments.

This script provides utilities to:
1. Extract all meshes from a LIBERO environment
2. Get poses (position + orientation) of all bodies and geometries
3. Export meshes to OBJ files
4. Save scene information to JSON
5. Visualize the scene structure

Usage:
    python extract_scene_meshes.py --task_suite_name libero_spatial --task_id 0 --output_dir output/scene_data
"""

import dataclasses
import json
import logging
import pathlib
from typing import Dict, Any, Optional

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import ControlEnv
import numpy as np
import tyro


@dataclasses.dataclass
class Args:
    task_suite_name: str = "libero_spatial"  # Task suite name
    task_id: int = 0  # Task ID within the suite
    output_dir: str = "output/scene_data"  # Output directory for extracted data
    export_meshes: bool = True  # Whether to export meshes to OBJ files
    export_json: bool = True  # Whether to export scene info to JSON
    resolution: int = 256  # Environment resolution
    seed: int = 7  # Random seed
    

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)


def get_libero_env(task, resolution, seed):
    """Initialize and return LIBERO environment."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "has_renderer": True
    }
    env = ControlEnv(**env_args)
    env.seed(seed)
    return env, task_description


def extract_mesh_info(env) -> Dict[str, Any]:
    """
    Extract all mesh information from the environment.
    
    Args:
        env: LIBERO environment
        
    Returns:
        Dictionary with mesh data including vertices and faces
    """
    sim = env.env.sim
    model = sim.model
    
    mesh_info = {}
    
    logging.info(f"Extracting {model.nmesh} meshes...")
    for i in range(model.nmesh):
        mesh_name = model.mesh_id2name(i)
        
        # Get mesh vertex data
        mesh_vertadr = model.mesh_vertadr[i]
        mesh_vertnum = model.mesh_vertnum[i]
        vertices = model.mesh_vert[mesh_vertadr:mesh_vertadr + mesh_vertnum].copy()
        
        # Get mesh face data
        mesh_faceadr = model.mesh_faceadr[i]
        mesh_facenum = model.mesh_facenum[i]
        faces = model.mesh_face[mesh_faceadr:mesh_faceadr + mesh_facenum].copy()
        
        mesh_info[mesh_name] = {
            'mesh_id': i,
            'vertices': vertices,
            'faces': faces,
            'num_vertices': mesh_vertnum,
            'num_faces': mesh_facenum
        }
        
        logging.info(f"  Mesh {i}: '{mesh_name}' - {mesh_vertnum} vertices, {mesh_facenum} faces")
    
    return mesh_info


def extract_body_poses(env) -> Dict[str, Any]:
    """
    Extract poses of all bodies in the environment.
    
    Args:
        env: LIBERO environment
        
    Returns:
        Dictionary with body pose data (position, quaternion, rotation matrix)
    """
    sim = env.env.sim
    model = sim.model
    data = sim.data
    
    body_poses = {}
    
    logging.info(f"Extracting poses for {model.nbody} bodies...")
    for i in range(model.nbody):
        body_name = model.body_id2name(i)
        
        # Get body position and orientation
        body_pos = data.body_xpos[i].copy()
        body_quat = data.body_xquat[i].copy()  # (w, x, y, z) format
        body_rotmat = data.body_xmat[i].copy().reshape(3, 3)
        
        body_poses[body_name] = {
            'body_id': i,
            'position': body_pos,
            'quaternion': body_quat,
            'rotation_matrix': body_rotmat
        }
        
        logging.info(f"  Body {i}: '{body_name}' at {body_pos}")
    
    return body_poses


def extract_geom_info(env) -> Dict[str, Any]:
    """
    Extract geometry information including mesh associations and poses.
    
    Args:
        env: LIBERO environment
        
    Returns:
        Dictionary with geometry data
    """
    sim = env.env.sim
    model = sim.model
    data = sim.data
    
    # Geometry type mapping
    GEOM_TYPES = {
        0: 'plane',
        1: 'hfield',
        2: 'sphere',
        3: 'capsule',
        4: 'ellipsoid',
        5: 'cylinder',
        6: 'box',
        7: 'mesh'
    }
    
    geom_info = {}
    
    logging.info(f"Extracting {model.ngeom} geometries...")
    for i in range(model.ngeom):
        geom_name = model.geom_id2name(i)
        geom_type = model.geom_type[i]
        geom_type_str = GEOM_TYPES.get(geom_type, 'unknown')
        geom_dataid = model.geom_dataid[i]
        geom_bodyid = model.geom_bodyid[i]
        
        # Get geom pose
        geom_pos = data.geom_xpos[i].copy()
        geom_rotmat = data.geom_xmat[i].copy().reshape(3, 3)
        
        # Get geom size
        geom_size = model.geom_size[i].copy()
        
        geom_data = {
            'geom_id': i,
            'type': geom_type_str,
            'type_code': int(geom_type),
            'body_id': int(geom_bodyid),
            'body_name': model.body_id2name(geom_bodyid),
            'position': geom_pos,
            'rotation_matrix': geom_rotmat,
            'size': geom_size
        }
        
        # If it's a mesh geom, add mesh information
        if geom_type == 7:  # mesh type
            mesh_name = model.mesh_id2name(geom_dataid)
            geom_data['mesh_id'] = int(geom_dataid)
            geom_data['mesh_name'] = mesh_name
            logging.info(f"  Geom {i}: '{geom_name}' ({geom_type_str}) - mesh: '{mesh_name}' at {geom_pos}")
        else:
            geom_data['mesh_id'] = None
            geom_data['mesh_name'] = None
            logging.info(f"  Geom {i}: '{geom_name}' ({geom_type_str}) at {geom_pos}")
        
        geom_info[geom_name] = geom_data
    
    return geom_info


def export_meshes_to_obj(mesh_info: Dict[str, Any], output_dir: pathlib.Path):
    """
    Export meshes to OBJ files.
    
    Args:
        mesh_info: Dictionary containing mesh data
        output_dir: Directory to save OBJ files
    """
    try:
        import trimesh
    except ImportError:
        logging.warning("trimesh not installed. Skipping mesh export. Install with: pip install trimesh")
        return
    
    mesh_dir = output_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Exporting meshes to {mesh_dir}...")
    for mesh_name, mesh_data in mesh_info.items():
        try:
            # Create trimesh object
            mesh = trimesh.Trimesh(
                vertices=mesh_data['vertices'],
                faces=mesh_data['faces']
            )
            
            # Export to OBJ file
            output_file = mesh_dir / f"{mesh_name}.obj"
            mesh.export(str(output_file))
            logging.info(f"  Exported: {output_file}")
        except Exception as e:
            logging.error(f"  Failed to export mesh '{mesh_name}': {e}")


def save_scene_info_json(scene_info: Dict[str, Any], output_file: pathlib.Path):
    """
    Save scene information to JSON file.
    
    Args:
        scene_info: Dictionary containing scene data
        output_file: Path to output JSON file
    """
    logging.info(f"Saving scene info to {output_file}...")
    
    # Create a JSON-serializable version
    json_data = {
        'meshes': {},
        'bodies': {},
        'geoms': {}
    }
    
    # Convert mesh info
    for mesh_name, mesh_data in scene_info['meshes'].items():
        json_data['meshes'][mesh_name] = {
            'mesh_id': mesh_data['mesh_id'],
            'num_vertices': mesh_data['num_vertices'],
            'num_faces': mesh_data['num_faces'],
            'vertices': mesh_data['vertices'].tolist(),
            'faces': mesh_data['faces'].tolist()
        }
    
    # Convert body poses
    for body_name, pose_data in scene_info['bodies'].items():
        json_data['bodies'][body_name] = {
            'body_id': pose_data['body_id'],
            'position': pose_data['position'].tolist(),
            'quaternion': pose_data['quaternion'].tolist(),
            'rotation_matrix': pose_data['rotation_matrix'].tolist()
        }
    
    # Convert geom info
    json_data['geoms'] = scene_info['geoms']
    
    # Save to file
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=2, cls=NumpyEncoder)
    
    logging.info(f"  Saved successfully!")


def print_scene_summary(scene_info: Dict[str, Any], task_description: str):
    """Print a summary of the scene structure."""
    print("\n" + "="*80)
    print(f"SCENE SUMMARY: {task_description}")
    print("="*80)
    
    print(f"\nTotal meshes: {len(scene_info['meshes'])}")
    print(f"Total bodies: {len(scene_info['bodies'])}")
    print(f"Total geoms: {len(scene_info['geoms'])}")
    
    # Find mesh geoms
    mesh_geoms = {k: v for k, v in scene_info['geoms'].items() if v['mesh_id'] is not None}
    print(f"\nMesh geometries: {len(mesh_geoms)}")
    for geom_name, geom_data in mesh_geoms.items():
        print(f"  - {geom_name}")
        print(f"      Body: {geom_data['body_name']}")
        print(f"      Mesh: {geom_data['mesh_name']}")
        print(f"      Position: [{geom_data['position'][0]:.3f}, {geom_data['position'][1]:.3f}, {geom_data['position'][2]:.3f}]")
    
    print("\n" + "="*80 + "\n")


def extract_scene(args: Args):
    """Main function to extract scene information."""
    # Initialize task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    
    # Get task
    task = task_suite.get_task(args.task_id)
    
    # Initialize environment
    logging.info(f"Initializing environment for task: {task.language}")
    env, task_description = get_libero_env(task, args.resolution, args.seed)
    
    # Reset environment to initialize scene
    env.reset()
    
    # Extract all scene information
    scene_info = {
        'task_description': task_description,
        'task_suite': args.task_suite_name,
        'task_id': args.task_id,
        'meshes': extract_mesh_info(env),
        'bodies': extract_body_poses(env),
        'geoms': extract_geom_info(env)
    }
    
    # Print summary
    print_scene_summary(scene_info, task_description)
    
    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export meshes to OBJ files
    if args.export_meshes:
        export_meshes_to_obj(scene_info['meshes'], output_dir)
    
    # Save scene info to JSON
    if args.export_json:
        task_name_clean = task_description.replace(" ", "_").replace(",", "")
        json_file = output_dir / f"scene_info_{task_name_clean}.json"
        save_scene_info_json(scene_info, json_file)
    
    logging.info(f"\nAll data saved to: {output_dir}")
    
    return scene_info


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    args = tyro.cli(Args)
    extract_scene(args)


