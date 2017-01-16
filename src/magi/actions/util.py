"""
Utility functions.
"""

import numpy as np

from openravepy import RaveCreateTrajectory, Robot
from prpy.util import ComputeUnitTiming, GetCollisionCheckPts, IsTimedTrajectory

def get_feasible_path(robot, path):
    """
    Check a path for feasibility (collision) and return a truncated path
    containing the path only up to just before the first collision.

    @param robot: the OpenRAVE robot
    @param path: the path to check for feasiblity
    @return tuple with the (possibly) truncated path, and a boolean indicating
      whether or not path was truncated
    """
    env = robot.GetEnv()
    path_cspec = path.GetConfigurationSpecification()
    dof_indices, _ = path_cspec.ExtractUsedIndices(robot)

    # Create a ConfigurationSpecification containing only positions.
    cspec = path.GetConfigurationSpecification()

    with robot.CreateRobotStateSaver(Robot.SaveParameters.LinkTransformation):

        # Check for collision. Compute a unit timing to simplify this.
        if IsTimedTrajectory(path):
            unit_path = path
        else:
            unit_path = ComputeUnitTiming(robot, path)

        t_collision = None
        t_prev = 0.0
        for t, dofvals in GetCollisionCheckPts(robot, unit_path):
            robot.SetDOFValues(dofvals, dof_indices)

            if env.CheckCollision(robot) or robot.CheckSelfCollision():
                t_collision = t_prev
                break
            t_prev = t

    # Build a path (with no timing) that stops before collision.
    output_path = RaveCreateTrajectory(env, '')

    if t_collision is not None:
        end_idx = unit_path.GetFirstWaypointIndexAfterTime(t_collision)
        if end_idx > 0:
            waypoints = unit_path.GetWaypoints(0, end_idx, cspec)
        else:
            waypoints = list()
        waypoints = np.concatenate((waypoints,
                                    unit_path.Sample(t_collision, cspec)))

        output_path.Init(cspec)
        output_path.Insert(0, waypoints)
    else:
        output_path.Clone(path, 0)

    return output_path, t_collision is not None
