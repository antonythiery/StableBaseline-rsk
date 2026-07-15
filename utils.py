import mujoco
import numpy as np

def draw_target_marker(viewer, target_x, target_y, z=0.05, radius=0.05, rgba=(1, 0, 0, 0.8)):
    """Draws a target marker in the MuJoCo viewer."""
    viewer.user_scn.ngeom = 0

    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0, 0],
        pos=[target_x, target_y, z],
        mat=np.eye(3).flatten(),
        rgba=np.array(rgba, dtype=np.float32),
    )
    viewer.user_scn.ngeom = 1
