"""
Utilities for integrating LIBERO environments with cuRobo motion planning.

This module helps bridge the gap between LIBERO's mounted Panda robot
and cuRobo's motion planning by:
1. Extracting robot base transform from LIBERO
2. Converting LIBERO obstacles to cuRobo WorldConfig
3. Providing coordinate transformations
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import yaml

import mesh_utils

try:
    from curobo.geom.types import Cuboid, Cylinder, Sphere, Mesh, WorldConfig
    from curobo.types.math import Pose
    CUROBO_AVAILABLE = True
except ImportError:
    CUROBO_AVAILABLE = False
    logging.warning("cuRobo not available. Install to use motion planning features.")


def get_robot_base_transform(env) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the robot base position and orientation from LIBERO environment.
    
    Args:
        env: LIBERO environment
        
    Returns:
        tuple: (position [x, y, z], quaternion [w, x, y, z])
    """
    sim = env.env.sim
    data = sim.data
    model = sim.model
    
    # Find the robot base body
    base_body_id = model.body_name2id("robot0_link0")
    # Get position and orientation
    position = data.body_xpos[base_body_id].copy()
    quaternion = data.body_xquat[base_body_id].copy()
    logging.info(f"Robot base position: {position}")
    logging.info(f"Robot base quaternion (w,x,y,z): {quaternion}")
    
    return position, quaternion


class RobotBaseTransform:
    """
    Stores robot base transform and provides coordinate transformation utilities.
    
    CuRobo assumes the robot base is at the origin. This class helps transform
    poses between LIBERO's world frame and the robot's base frame.
    """
    def __init__(self, position: np.ndarray, quaternion: np.ndarray):
        """
        Args:
            position: Robot base position in world frame [x, y, z]
            quaternion: Robot base quaternion in world frame [w, x, y, z]
        """
        self.position = position
        self.quaternion = quaternion  # [w, x, y, z]
        
        # Create transformation matrix from world to robot base
        self.world_to_base_matrix = self._compute_world_to_base_transform()
        
    def _compute_world_to_base_transform(self) -> np.ndarray:
        """Compute 4x4 transformation matrix from world frame to robot base frame."""
        # Convert quaternion to rotation matrix
        w, x, y, z = self.quaternion
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        
        # Create 4x4 homogeneous transformation matrix (base in world frame)
        T_world_base = np.eye(4)
        T_world_base[:3, :3] = R
        T_world_base[:3, 3] = self.position
        
        # Invert to get world-to-base transform
        T_base_world = np.linalg.inv(T_world_base)
        
        return T_base_world
    
    def transform_pose_to_base_frame(self, position: np.ndarray, quaternion: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform a pose from world frame to robot base frame.
        
        Args:
            position: Position in world frame [x, y, z]
            quaternion: Quaternion in world frame [w, x, y, z]
            
        Returns:
            tuple: (position_in_base, quaternion_in_base) both as np.ndarray
        """
        # Convert quaternion to rotation matrix
        w, x, y, z = quaternion
        R_world = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        
        # Create 4x4 homogeneous transformation
        T_world_pose = np.eye(4)
        T_world_pose[:3, :3] = R_world
        T_world_pose[:3, 3] = position
        
        # Transform to base frame
        T_base_pose = self.world_to_base_matrix @ T_world_pose
        
        # Extract position and rotation
        position_base = T_base_pose[:3, 3]
        R_base = T_base_pose[:3, :3]
        quaternion_base = rotation_matrix_to_quaternion(R_base)
        
        return position_base, quaternion_base
    
    def transform_position_to_base_frame(self, position: np.ndarray) -> np.ndarray:
        """
        Transform a position from world frame to robot base frame.
        
        Args:
            position: Position in world frame [x, y, z]
            
        Returns:
            np.ndarray: Position in robot base frame
        """
        pos_homogeneous = np.append(position, 1.0)
        pos_base_homogeneous = self.world_to_base_matrix @ pos_homogeneous
        return pos_base_homogeneous[:3]


def convert_libero_obstacles_to_curobo(
    scene_info: Dict[str, Any],
    base_transform: Optional[RobotBaseTransform] = None,
    exclude_robot: bool = True,
    include_primitives: bool = True,
    include_meshes: bool = True,
    table_as_collision: bool = True
) -> WorldConfig:
    """
    Convert LIBERO scene collision obstacles to cuRobo WorldConfig.
    
    Only extracts collision geometries (boxes, cylinders, spheres) and excludes visual meshes.
    Visual geometries (g0, _vis, _visual) are filtered out as they are for rendering only.
    
    All obstacle poses will be transformed to the robot's base frame if base_transform is provided.
    
    Args:
        scene_info: Scene information from mesh_utils.get_mesh_info_and_poses()
        base_transform: RobotBaseTransform for coordinate transformation (optional)
        exclude_robot: If True, exclude robot geometries
        include_primitives: If True, include primitive collision geometries (table, walls)
        include_meshes: If True, include collision mesh geometries (non-visual meshes)
        table_as_collision: If True, include table as collision object
        
    Returns:
        WorldConfig: cuRobo world configuration with collision obstacles in robot base frame
    """
    if not CUROBO_AVAILABLE:
        raise ImportError("cuRobo not available. Please install cuRobo.")
    
    cuboids = []
    cylinders = []
    spheres = []
    meshes = []

    for geom_name, geom_data in scene_info['geom_info'].items():
        # Skip robot parts if requested
        if exclude_robot and any(x in geom_name.lower() for x in ['robot', 'gripper', 'mount']):
            continue
        
        # Skip walls and floor if not including primitives
        if not include_primitives and any(x in geom_name.lower() for x in ['wall', 'floor']):
            continue
        
        # Optionally skip table
        if not table_as_collision and 'table' in geom_name.lower():
            continue
        # Skip objects that might interfere with grasping
        if any(x in geom_name.lower() for x in [
            'akita_black_bowl_1',  # Target object
            # 'akita_black_bowl_2',  # Other bowl
            # 'wall'
            # 'cookies_1',           # Cookies
            # 'glazed_rim_porcelain_ramekin_1',  # Ramekin
            # 'plate_1'              # Plate
            'table'
        ]):
            position = geom_data['position']
            logging.info(f"Skipping object: {geom_name} at position: {position}")
            continue
        
        # Only include collision geometries, skip visual geometries
        # Visual geometries are typically named with _g0 suffix or contain 'vis' in name
        if (geom_name.endswith('_g0') or 
            'vis' in geom_name.lower() or 
            geom_name.endswith('_visual')):
            # Skip visual geometries - these are for rendering only
            continue
        
        position = geom_data['position']
        rotation_matrix = geom_data['rotation_matrix']
        size = geom_data['size']
        geom_type = geom_data['type']
        
        # Convert rotation matrix to quaternion
        quat = rotation_matrix_to_quaternion(rotation_matrix)
        # Transform pose to robot base frame if transform is provided
        if base_transform is not None:
            position, quat = base_transform.transform_pose_to_base_frame(position, quat)
        # Create cuRobo geometry based on type
        # Skip visual meshes (g0) - only use collision geometries (boxes, cylinders, spheres)
        if geom_type == 'mesh' and include_meshes:
            # Only include collision meshes, skip visual meshes (g0)
            if geom_name.endswith('_g0'):
                # Skip visual meshes - these are for rendering only, not collision
                continue
                
            # Handle collision mesh geometries (if any exist)
            mesh_id = geom_data['mesh_id']
            if mesh_id is not None:
                mesh_name = geom_data['mesh_name']
                mesh_data = scene_info['meshes'][mesh_name]
                
                # Get vertices and faces
                vertices = mesh_data['vertices']
                faces = mesh_data['faces']
                
                # Convert to lists for cuRobo
                vertices_list = vertices.tolist()
                faces_list = faces.flatten().tolist()  # cuRobo expects flattened faces
                
                pose = [
                    float(position[0]), float(position[1]), float(position[2]),
                    float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
                ]
                
                meshes.append(Mesh(
                    name=geom_name,
                    pose=pose,
                    vertices=vertices_list,
                    faces=faces_list
                ))
        
        elif geom_type == 'box':
            # size: [half_x, half_y, half_z], cuRobo expects full dimensions
            dims = [float(size[0] * 2), float(size[1] * 2), float(size[2] * 2)]
            pose = [
                float(position[0]), float(position[1]), float(position[2]),
                float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
            ]
            cuboids.append(Cuboid(name=geom_name, pose=pose, dims=dims))
            
        elif geom_type == 'cylinder':
            # size: [radius, half_height]
            radius = float(size[0])
            height = float(size[1] * 2)
            pose = [
                float(position[0]), float(position[1]), float(position[2]),
                float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
            ]
            cylinders.append(Cylinder(name=geom_name, pose=pose, radius=radius, height=height))
            
        elif geom_type == 'sphere':
            # size: [radius, _, _]
            radius = float(size[0])
            position_list = [float(position[0]), float(position[1]), float(position[2])]
            spheres.append(Sphere(name=geom_name, position=position_list, radius=radius))
    
    #HACK: add collision cuboid
    # cuboids.append(Cuboid(name="collision_cuboid", pose=[0.5382635, 0.06717007, 0.35379136, 0.02234961, -0.923187, 0.38369736, 0.0016358], dims=[0.01, 0.01, 0.01]))
    # Create WorldConfig
    world_config = WorldConfig(
        mesh=meshes if meshes else None,
        cuboid=cuboids if cuboids else None,
        cylinder=cylinders if cylinders else None,
        sphere=spheres if spheres else None
    )
    
    logging.info(f"Created collision-only world with {len(meshes)} collision meshes, {len(cuboids)} collision boxes, {len(cylinders)} collision cylinders, {len(spheres)} collision spheres")

    # Debug: Log which collision objects are included
    if meshes:
        mesh_names = [mesh.name for mesh in meshes]
        logging.info(f"Collision mesh objects: {mesh_names}")
    if cuboids:
        cuboid_names = [cuboid.name for cuboid in cuboids]
        logging.info(f"Collision box objects: {cuboid_names}")
    if cylinders:
        cylinder_names = [cylinder.name for cylinder in cylinders]
        logging.info(f"Collision cylinder objects: {cylinder_names}")
    if spheres:
        sphere_names = [sphere.name for sphere in spheres]
        logging.info(f"Collision sphere objects: {sphere_names}")
    return world_config


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a quaternion [w, x, y, z].
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        np.ndarray: Quaternion [w, x, y, z]
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])


def get_current_joint_state(env) -> np.ndarray:
    """
    Get current joint positions from LIBERO environment.
    
    Args:
        env: LIBERO environment
        
    Returns:
        np.ndarray: Joint positions [7 arm joints + 2 gripper joints]
    """
    sim = env.env.sim
    
    # Get joint positions for Panda (7 arm joints + 2 gripper joints)
    qpos = sim.data.qpos[:9].copy()
    logging.info(f"Current joint state: {qpos}")
    
    return qpos


def get_ee_pose(env) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get end-effector pose from LIBERO environment.
    
    Args:
        env: LIBERO environment
        
    Returns:
        tuple: (position [x, y, z], quaternion [w, x, y, z])
    """
    # obs = env.env._get_observations()
    
    # # Get end-effector position and quaternion
    # ee_pos = obs["robot0_eef_pos"]
    # ee_quat = obs["robot0_eef_quat"]  # This is [x, y, z, w]
    
    # # Convert to [w, x, y, z] format
    # ee_quat_wxyz = np.array([ee_quat[3], ee_quat[0], ee_quat[1], ee_quat[2]])
    
    # logging.info(f"End-effector position: {ee_pos}")
    # logging.info(f"End-effector quaternion (w,x,y,z): {ee_quat_wxyz}")
    
    ee_pos = env.env.sim.data.body_xpos[env.env.sim.model.body_name2id("robot0_link7")]
    ee_quat = env.env.sim.data.body_xquat[env.env.sim.model.body_name2id("robot0_link7")]

    # return ee_pos, ee_quat_wxyz
    return ee_pos, ee_quat


def setup_libero_curobo_motion_gen(
    env,
    robot_config_base: str = "franka.yml",
    tensor_args=None,
    exclude_table: bool = False,
    include_meshes: bool = True,
    collision_cache_mesh: int = 50,
    collision_cache_cuboid: int = 500,
    collision_cache_cylinder: int = 50,
    **motion_gen_kwargs
):
    """
    Setup cuRobo MotionGen for LIBERO environment.
    
    This function:
    1. Extracts the robot base transform from LIBERO
    2. Converts all obstacles to robot's base frame
    3. Sets up MotionGen with standard franka.yml config and LIBERO world directly
    
    IMPORTANT: Goal poses must be transformed to robot base frame using the returned
    base_transform object before passing to motion_gen.plan_single().
    
    Args:
        env: LIBERO environment
        robot_config_base: Base robot configuration file (e.g., "franka.yml")
        tensor_args: TensorDeviceType for cuRobo
        exclude_table: If True, don't include table in collision checking
        include_meshes: If True, include mesh objects (bowls, plates, etc.) as obstacles
        collision_cache_mesh: Max number of mesh obstacles (default: 50)
        collision_cache_cuboid: Max number of cuboid obstacles (default: 500)
        collision_cache_cylinder: Max number of cylinder obstacles (default: 50)
        **motion_gen_kwargs: Additional arguments for MotionGenConfig
        
    Returns:
        tuple: (motion_gen, robot_config_path, world_config, base_transform)
            - motion_gen: Configured MotionGen instance
            - robot_config_path: Path to robot config file used
            - world_config: WorldConfig with obstacles in robot base frame
            - base_transform: RobotBaseTransform for coordinate transformations
    """
    if not CUROBO_AVAILABLE:
        raise ImportError("cuRobo not available. Please install cuRobo.")
    
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
    
    # Step 1: Extract robot base transform from LIBERO
    base_position, base_quaternion = get_robot_base_transform(env)
    base_transform = RobotBaseTransform(base_position, base_quaternion)
    
    logging.info(f"Robot base in LIBERO world: position={base_position}, quaternion={base_quaternion}")
    
    # Step 2: Extract scene obstacles and transform to robot base frame
    scene_info = mesh_utils.get_mesh_info_and_poses(env)
    world_config = convert_libero_obstacles_to_curobo(
        scene_info,
        base_transform=base_transform,  # Transform obstacles to robot frame
        exclude_robot=True,
        include_primitives=True,
        include_meshes=include_meshes,
        table_as_collision=not exclude_table
    )
    
    # Step 3: Create MotionGen with LIBERO world directly (no redundant world config)
    # Set collision cache sizes appropriate for LIBERO scenes
    collision_cache = {
        "obb": collision_cache_cuboid,
        "mesh": collision_cache_mesh,
        "cylinder": collision_cache_cylinder,
    }
    
    # Create empty world config for initialization (most efficient approach)
    empty_world = WorldConfig()
    
    motion_gen_cfg = MotionGenConfig.load_from_robot_config(
        robot_config_base,
        empty_world,  # Use empty world directly as world_model parameter
        tensor_args,
        collision_cache=collision_cache,
        **motion_gen_kwargs
    )
    
    motion_gen = MotionGen(motion_gen_cfg)
    motion_gen.warmup()
    
    # Set the actual LIBERO world (no redundant update needed)
    motion_gen.update_world(world_config) 
    
    logging.info("Successfully set up cuRobo MotionGen for LIBERO environment")
    logging.info("NOTE: Goal poses must be transformed to robot base frame using base_transform")
    
    return motion_gen, robot_config_base, world_config, base_transform

