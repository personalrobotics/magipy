"""
Define MoveUntilTouchAction, MoveUntilTouchSolution, and
MoveUntilTouchExecutableSolution.
"""

from contextlib import nested
import logging

import numpy as np

from openravepy import Robot, KinBody
from prpy.exceptions import TrajectoryAborted
from prpy.planning import PlanningError
from prpy.planning.base import Tags
from prpy.rave import AllDisabled
from prpy.util import CopyTrajectory, GetTrajectoryTags
from prpy.viz import RenderVector

from magi.actions.base import Action, ActionError, ExecutableSolution, Solution, from_key, to_key
from magi.actions.util import get_feasible_path

LOGGER = logging.getLogger(__name__)


class MoveUntilTouchExecutableSolution(ExecutableSolution):
    """
    An ExecutableSolution that moves in a direction for a specified distance
    unless an object is touched.
    """

    def __init__(self, solution, traj):
        """
        @param solution: Solution object that generated this ExecutableSolution
        @param traj: planned trajectory to execute
        """
        super(MoveUntilTouchExecutableSolution, self).__init__(solution)
        self.traj = traj

    @property
    def action(self):
        """
        Return the solution's Action.

        @return MoveUntilTouchAction
        """
        return self.solution.action

    def execute(self, env, simulate):
        """
        Execute the planned trajectory.

        @param env: OpenRAVE environment
        @param simulate: flag to run in simulation
        @return True if the trajectory was aborted due to touching an object
        """
        manipulator = self.action.get_manipulator(env)
        robot = manipulator.GetRobot()
        traj = CopyTrajectory(self.traj, env=env)
        touched = False
        try:
            robot.ExecuteTrajectory(traj)
        except TrajectoryAborted:
            touched = True

        return touched


class MoveUntilTouchSolution(Solution):
    """
    A Solution that moves in a direction for a specified distance unless an
    object is touched.
    """

    def __init__(self, action, path, deterministic):
        """
        @param action: Action that generated this Solution
        @param path: path to execute
        @param deterministic: True if this Solution was generated using a
          deterministic planner
        """
        super(MoveUntilTouchSolution, self).__init__(
            action, deterministic=deterministic)
        self.path = path

    def save(self, env):
        """
        Return a context manager that saves the current robot configuration
        (LinkTransformation).

        @param env: OpenRAVE environment
        @return a RobotStateSaver that saves the current configuration
        """
        robot = self.action.get_manipulator(env).GetRobot()

        return robot.CreateRobotStateSaver(
            Robot.SaveParameters.LinkTransformation)

    def postprocess(self, env):
        """
        WARNING: This action has been temporarily disabled.

        Postprocess the trajectory, adding required flags to indicate the trajectory
        should be stopped when high forces/torques are encountered.

        @param env: OpenRAVE environment
        @return a MoveUntilTouchExecutableSolution
        """
        manipulator = self.action.get_manipulator(env)
        robot = manipulator.GetRobot()

        with env:
            # Compute the expected force direction in the sensor frame.
            hand_pose = manipulator.GetEndEffectorTransform()
            relative_direction = np.dot(hand_pose[0:3, 0:3],
                                        self.action.direction)

            # Tell the controller to stop on force/torque input.
            path = CopyTrajectory(self.path, env=env)
            # manipulator.SetTrajectoryExecutionOptions(path,
            #     stop_on_stall=True, stop_on_ft=True,
            #     force_magnitude=self.action.force_magnitude,
            #     force_direction=relative_direction,
            #     torque=self.action.torque)

            # Compute the trajectory timing. This is potentially slow.
            traj = robot.PostProcessPath(path)

        return MoveUntilTouchExecutableSolution(solution=self, traj=traj)

    def jump(self, env):
        """
        Move the robot to the last configuration before a touch is expected to
        occur.

        @param env: OpenRAVE environment
        """
        manipulator = self.action.get_manipulator(env)
        robot = manipulator.GetRobot()

        with robot.CreateRobotStateSaver(Robot.SaveParameters.ActiveDOF):
            robot.SetActiveDOFs(manipulator.GetArmIndices())
            cspec = robot.GetActiveConfigurationSpecification()

            # Jump to the end of the trajectory.
            robot.SetActiveDOFValues(
                self.path.GetWaypoint(self.path.GetNumWaypoints() - 1, cspec))


class MoveUntilTouchAction(Action):
    """
    An Action that moves in a direction for the specified distance unless an
    object is touched.
    """

    TORQUE_MAGNITUDE = 127 * np.ones(3)

    def __init__(self,
                 manipulator,
                 direction,
                 min_distance,
                 max_distance,
                 target_bodies,
                 force_magnitude=5.,
                 torque=None,
                 planner=None,
                 name=None):
        """
        @param manipulator: manipulator to move
        @param direction: 3 dim vector - [x,y,z] in world coordinates
        @param min_distance: min distance to move
        @param max_distance: max distance to move
        @param target_bodies: any bodies that should be disabled during planning
        @param force_magnitude: max allowable magnitude of force before
          considering the manipulator to be touching something
        @param torque: the max allowable torque before considering the
          manipulator to be touching something
        @param planner: planner to use to generate the trajectory
          if None, defaults to robot.planner
        @param name: name of the action
        """
        super(MoveUntilTouchAction, self).__init__(name=name)

        self._manipulator = to_key(manipulator)
        self._target_bodies = [to_key(body) for body in target_bodies]

        self.direction = np.array(direction, dtype='float')
        self.min_distance = float(min_distance)
        self.max_distance = float(max_distance)
        self.force_magnitude = float(force_magnitude)
        self.planner = planner

        if torque is not None:
            self.torque = np.array(torque, dtype='float')
        else:
            self.torque = self.TORQUE_MAGNITUDE

    def get_manipulator(self, env):
        """
        Look up and return the manipulator in the environment.

        @param env: OpenRAVE environment
        @return an OpenRAVE manipulator
        """
        return from_key(env, self._manipulator)

    def get_bodies(self, env, bodies):
        """
        Look up and return the bodies in the environment.

        @param env: OpenRAVE environment
        @param bodies: list of body keys
        @return list of Kinbodies
        """
        return [from_key(env, key) for key in bodies]

    def plan(self, env):
        """
        Plan a trajectory that moves in the specified direction for the
        specified distance unless an object is touched.

        @param env: OpenRAVE environment
        @return a MoveUntilTouchSolution
        """
        target_bodies = self.get_bodies(env, self._target_bodies)
        manipulator = self.get_manipulator(env)
        robot = manipulator.GetRobot()
        env = robot.GetEnv()  # TODO: why is this necessary?

        start_point = manipulator.GetEndEffectorTransform()[0:3, 0]
        ssavers = [
            robot.CreateRobotStateSaver(
                Robot.SaveParameters.ActiveDOF
                | Robot.SaveParameters.ActiveManipulator
                | Robot.SaveParameters.LinkTransformation)
        ]
        for body in target_bodies:
            # ensure we preserve any enable-disabling that wraps
            # this function
            ssavers += [
                body.CreateKinBodyStateSaver(KinBody.SaveParameters.LinkEnable)
            ]

        with \
            nested(*ssavers), \
            RenderVector(
                start_point, self.direction, self.max_distance, env), \
            AllDisabled(env, target_bodies):

            manipulator.SetActive()
            planner = self.planner if self.planner is not None else robot.planner

            try:
                path = planner.PlanToEndEffectorOffset(
                    robot=robot,
                    direction=self.direction,
                    distance=self.min_distance,
                    max_distance=self.max_distance)

                # Mark this action as deterministic based on the traj tags
                path_tags = GetTrajectoryTags(path)
                deterministic = path_tags.get(Tags.DETERMINISTIC_ENDPOINT, None)
                if deterministic is None:
                    LOGGER.warn(
                        "Trajectory does not have DETERMINISTIC_ENDPOINT flag set. "
                        "Assuming non-deterministic.")
                    deterministic = False

            except PlanningError as err:
                raise ActionError(str(err), deterministic=err.deterministic)

        path, _ = get_feasible_path(robot, path)
        return MoveUntilTouchSolution(
            action=self, path=path, deterministic=deterministic)
