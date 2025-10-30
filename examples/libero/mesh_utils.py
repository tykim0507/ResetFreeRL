"""
Utility functions for extracting mesh and pose information from LIBERO environments.

Can be imported and used in other scripts like poc_action_stack.py
"""

import logging
import json
import pathlib
from typing import Dict, Any, Optional

import numpy as np

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


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


def create_primitive_mesh(geom_type: str, size: np.ndarray):
    """
    Create a mesh for a primitive geometry.
    
    Args:
        geom_type: Type of geometry ('box', 'cylinder', 'sphere', 'plane', etc.)
        size: Size parameters for the geometry
        
    Returns:
        tuple: (vertices, faces) or None if geometry type not supported
    """
    if not TRIMESH_AVAILABLE:
        logging.warning("trimesh not available, cannot create primitive meshes")
        return None
    
    try:
        if geom_type == 'box':
            # size: [half_x, half_y, half_z]
            extents = size * 2  # trimesh uses full extents
            mesh = trimesh.creation.box(extents=extents)
            return mesh.vertices, mesh.faces
            
        elif geom_type == 'cylinder':
            # size: [radius, half_height]
            radius = size[0]
            height = size[1] * 2
            mesh = trimesh.creation.cylinder(radius=radius, height=height)
            return mesh.vertices, mesh.faces
            
        elif geom_type == 'sphere':
            # size: [radius, _, _]
            radius = size[0]
            mesh = trimesh.creation.icosphere(subdivisions=3, radius=radius)
            return mesh.vertices, mesh.faces
            
        elif geom_type == 'capsule':
            # size: [radius, half_height]
            radius = size[0]
            height = size[1] * 2
            mesh = trimesh.creation.capsule(radius=radius, height=height)
            return mesh.vertices, mesh.faces
            
        elif geom_type == 'plane':
            # Create a large plane for visualization
            # Planes are usually very large or infinite, so we create a reasonable size
            size_xy = 5.0 if size[0] == 0 else size[0] * 2
            mesh = trimesh.creation.box(extents=[size_xy, size_xy, 0.01])
            return mesh.vertices, mesh.faces
            
        else:
            logging.debug(f"Primitive type '{geom_type}' not supported for mesh generation")
            return None
            
    except Exception as e:
        logging.error(f"Failed to create mesh for {geom_type}: {e}")
        return None


def get_mesh_info_and_poses(env) -> Dict[str, Any]:
    """
    Extract mesh files and their poses from the LIBERO environment.
    
    Args:
        env: LIBERO environment (ControlEnv or OffScreenRenderEnv)
        
    Returns:
        Dictionary containing:
            - meshes: mesh data (vertices, faces)
            - body_poses: body positions and orientations
            - geom_info: geometry information linking meshes to bodies
    """
    sim = env.env.sim
    model = sim.model
    data = sim.data
    
    # Extract mesh information
    mesh_info = {}
    logging.info(f"Number of meshes: {model.nmesh}")
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
        
        logging.info(f"  Mesh {i}: {mesh_name} - {mesh_vertnum} vertices, {mesh_facenum} faces")
    
    # Extract all body poses
    body_poses = {}
    logging.info(f"\nNumber of bodies: {model.nbody}")
    for i in range(model.nbody):
        body_name = model.body_id2name(i)
        
        # Get body position and orientation (quaternion)
        body_pos = data.body_xpos[i].copy()  # 3D position
        body_quat = data.body_xquat[i].copy()  # Quaternion (w, x, y, z)
        body_rotmat = data.body_xmat[i].copy().reshape(3, 3)  # Rotation matrix
        
        body_poses[body_name] = {
            'body_id': i,
            'position': body_pos,
            'quaternion': body_quat,
            'rotation_matrix': body_rotmat
        }
        
        logging.info(f"  Body {i}: {body_name} at position {body_pos}")
    
    # Extract geom (geometry) information - connects meshes to bodies
    GEOM_TYPES = {
        0: 'plane', 1: 'hfield', 2: 'sphere', 3: 'capsule',
        4: 'ellipsoid', 5: 'cylinder', 6: 'box', 7: 'mesh'
    }
    
    geom_info = {}
    logging.info(f"\nNumber of geoms: {model.ngeom}")
    for i in range(model.ngeom):
        geom_name = model.geom_id2name(i)
        geom_type = model.geom_type[i]
        geom_type_str = GEOM_TYPES.get(geom_type, 'unknown')
        geom_dataid = model.geom_dataid[i]  # mesh id if geom_type==7
        geom_bodyid = model.geom_bodyid[i]  # which body this geom is attached to
        
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
        
        if geom_type == 7:  # If it's a mesh
            mesh_name = model.mesh_id2name(geom_dataid)
            geom_data['mesh_id'] = int(geom_dataid)
            geom_data['mesh_name'] = mesh_name
            logging.info(f"  Geom {i}: {geom_name} ({geom_type_str}) - mesh: {mesh_name}, body: {model.body_id2name(geom_bodyid)}")
        else:
            geom_data['mesh_id'] = None
            geom_data['mesh_name'] = None
            logging.info(f"  Geom {i}: {geom_name} ({geom_type_str})")
        
        geom_info[geom_name] = geom_data
    return {
        'meshes': mesh_info,
        'body_poses': body_poses,
        'geom_info': geom_info,
    }


def save_scene_to_json(scene_info: Dict[str, Any], output_path: str):
    """
    Save scene information to a JSON file.
    
    Args:
        scene_info: Dictionary from get_mesh_info_and_poses()
        output_path: Path to output JSON file
    """
    output_file = pathlib.Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(scene_info, f, indent=2, cls=NumpyEncoder)
    
    logging.info(f"Scene info saved to: {output_file}")


def export_mesh_to_obj(mesh_name: str, vertices: np.ndarray, faces: np.ndarray, output_path: str, 
                       position: np.ndarray = None, rotation_matrix: np.ndarray = None,
                       apply_transform: bool = False):
    """
    Export a single mesh to OBJ file.
    
    Args:
        mesh_name: Name of the mesh
        vertices: Nx3 array of vertex positions (in local coordinates)
        faces: Mx3 array of face indices
        output_path: Path to output OBJ file
        position: 3D position (translation) in world coordinates
        rotation_matrix: 3x3 rotation matrix
        apply_transform: If True, transform vertices to world coordinates
    """
    # Apply transformation if requested
    if apply_transform and position is not None and rotation_matrix is not None:
        # Transform vertices: v_world = R * v_local + t
        vertices_transformed = (rotation_matrix @ vertices.T).T + position
    else:
        vertices_transformed = vertices
    
    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=faces)
        
        # Add pose metadata as comments if available
        if position is not None and rotation_matrix is not None:
            header_comments = (
                f"# Mesh: {mesh_name}\n"
                f"# Position: [{position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f}]\n"
                f"# Rotation Matrix:\n"
                f"#   [{rotation_matrix[0,0]:.6f}, {rotation_matrix[0,1]:.6f}, {rotation_matrix[0,2]:.6f}]\n"
                f"#   [{rotation_matrix[1,0]:.6f}, {rotation_matrix[1,1]:.6f}, {rotation_matrix[1,2]:.6f}]\n"
                f"#   [{rotation_matrix[2,0]:.6f}, {rotation_matrix[2,1]:.6f}, {rotation_matrix[2,2]:.6f}]\n"
                f"# Transform Applied: {apply_transform}\n"
            )
            # Save with custom header
            with open(output_path, 'w') as f:
                f.write(header_comments)
                f.write("\n")
            # Append mesh data
            mesh.export(output_path, file_type='obj', include_color=False)
        else:
            mesh.export(output_path, file_type='obj', include_color=False)
        
        logging.info(f"Exported mesh '{mesh_name}' to: {output_path}")
    except ImportError:
        logging.warning("trimesh not installed. Install with: pip install trimesh")
        # Fallback: write simple OBJ file manually
        output_file = pathlib.Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(f"# Mesh: {mesh_name}\n")
            f.write(f"# Vertices: {len(vertices_transformed)}\n")
            f.write(f"# Faces: {len(faces)}\n")
            
            # Add pose metadata if available
            if position is not None and rotation_matrix is not None:
                f.write(f"# Position: [{position[0]:.6f}, {position[1]:.6f}, {position[2]:.6f}]\n")
                f.write(f"# Rotation Matrix:\n")
                f.write(f"#   [{rotation_matrix[0,0]:.6f}, {rotation_matrix[0,1]:.6f}, {rotation_matrix[0,2]:.6f}]\n")
                f.write(f"#   [{rotation_matrix[1,0]:.6f}, {rotation_matrix[1,1]:.6f}, {rotation_matrix[1,2]:.6f}]\n")
                f.write(f"#   [{rotation_matrix[2,0]:.6f}, {rotation_matrix[2,1]:.6f}, {rotation_matrix[2,2]:.6f}]\n")
                f.write(f"# Transform Applied: {apply_transform}\n")
            
            f.write("\n")
            
            # Write vertices
            for v in vertices_transformed:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write faces (OBJ uses 1-based indexing)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        
        logging.info(f"Exported mesh '{mesh_name}' to: {output_path} (manual OBJ)")


def export_all_meshes(scene_info: Dict[str, Any], output_dir: str = "meshes", apply_transform: bool = True, include_primitives: bool = False):
    """
    Export all meshes from scene_info to OBJ files.
    
    Args:
        scene_info: Dictionary from get_mesh_info_and_poses()
        output_dir: Directory to save mesh files
        apply_transform: If True, transform vertices to world coordinates using geom poses
        include_primitives: If True, also export primitive geometries (box, cylinder, etc.) as meshes
    """
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    exported_count = 0
    
    # Export meshes with their world poses from geom_info
    for geom_name, geom_data in scene_info['geom_info'].items():
        position = geom_data['position']
        rotation_matrix = geom_data['rotation_matrix']
        geom_type = geom_data['type']
        
        # Handle mesh geometries
        if geom_data['mesh_id'] is not None:
            mesh_name = geom_data['mesh_name']
            mesh_data = scene_info['meshes'][mesh_name]
            
            obj_file = output_path / f"{geom_name}.obj"
            export_mesh_to_obj(
                geom_name,
                mesh_data['vertices'],
                mesh_data['faces'],
                str(obj_file),
                position=position,
                rotation_matrix=rotation_matrix,
                apply_transform=apply_transform
            )
            exported_count += 1
            
        # Handle primitive geometries if requested
        elif include_primitives and geom_type in ['box', 'cylinder', 'sphere', 'capsule', 'plane']:
            size = geom_data['size']
            prim_mesh = create_primitive_mesh(geom_type, size)
            
            if prim_mesh is not None:
                vertices, faces = prim_mesh
                obj_file = output_path / f"{geom_name}.obj"
                export_mesh_to_obj(
                    geom_name,
                    vertices,
                    faces,
                    str(obj_file),
                    position=position,
                    rotation_matrix=rotation_matrix,
                    apply_transform=apply_transform
                )
                exported_count += 1
    
    logging.info(f"Exported {exported_count} geometries")
    return exported_count


def print_scene_summary(scene_info: Dict[str, Any]):
    """Print a summary of the scene."""
    print("\n" + "="*80)
    print("SCENE SUMMARY")
    print("="*80)
    
    print(f"\nMeshes: {len(scene_info['meshes'])}")
    for mesh_name, mesh_data in scene_info['meshes'].items():
        print(f"  - {mesh_name}: {mesh_data['num_vertices']} vertices, {mesh_data['num_faces']} faces")
    
    print(f"\nBodies: {len(scene_info['body_poses'])}")
    for body_name, pose in scene_info['body_poses'].items():
        pos = pose['position']
        print(f"  - {body_name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    print(f"\nMesh Geometries:")
    mesh_geoms = {k: v for k, v in scene_info['geom_info'].items() if v['mesh_id'] is not None}
    for geom_name, geom_data in mesh_geoms.items():
        pos = geom_data['position']
        print(f"  - {geom_name} (body: {geom_data['body_name']}, mesh: {geom_data['mesh_name']})")
        print(f"      Position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    
    print("\n" + "="*80 + "\n")


def get_object_mesh_poses(scene_info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Get a simplified view of object meshes with their world poses.
    
    Returns a dictionary mapping object names to their mesh and pose information.
    Useful for collision checking, motion planning, etc.
    """
    object_info = {}
    
    # Find all mesh geometries
    for geom_name, geom_data in scene_info['geom_info'].items():
        if geom_data['mesh_id'] is not None:
            mesh_name = geom_data['mesh_name']
            mesh_data = scene_info['meshes'].get(mesh_name, {})
            
            object_info[geom_name] = {
                'mesh_name': mesh_name,
                'body_name': geom_data['body_name'],
                'position': geom_data['position'],
                'rotation_matrix': geom_data['rotation_matrix'],
                'vertices': mesh_data.get('vertices'),
                'faces': mesh_data.get('faces'),
            }
    
    return object_info

