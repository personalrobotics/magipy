from .base import Action, ActionError, ExecutableSolution, Solution, from_key, to_key
from openravepy import Robot
from prpy.util import CopyTrajectory

import logging
LOGGER = logging.getLogger(__name__)


class PlanExecutableSolution(ExecutableSolution):
    def __init__(self, solution, traj):
        """
        @param solution The Solution object that generated this ExecutableSolution
        @param traj The trajectory to execute
        """
        super(PlanExecutableSolution, self).__init__(solution)
        self.traj = traj

    @property
    def action(self):
        return self.solution.action

    def execute(self, env, simulate):
        """
        Execute the trajectory defined when constructing the action.
        Timing data about the execution of this trajectory is recorded and logged
        via a call to log_execution_data
        @param env The OpenRAVE environment
        @param simulate If true, simulate execution
        """
        robot = self.action.get_robot(env)

        from prpy.util import Timer, SetTrajectoryTags
        from prpy.planning.base import Tags
        traj_copy = CopyTrajectory(self.traj, env=env)
        with Timer() as timer:
            traj = robot.ExecuteTrajectory(traj_copy)
        SetTrajectoryTags(
            traj, {Tags.EXECUTION_TIME: timer.get_duration()}, append=True)

        from ..logging_utils import log_execution_data
        log_execution_data(traj, self.action.get_name())

        return traj


class PlanSolution(Solution):
    def __init__(self, action, path, deterministic):
        """
        @param action The action that generated this Solution
        @param path The path generated by the Action
        @param determinsitic True if this solution was created in a
        deterministic fashion.
        """
        super(PlanSolution, self).__init__(action, deterministic=deterministic)
        self.path = path

    def save(self, env):
        """
        Creates and returns a RobotStateSaver that saves the
        current configuration (LinkTransformation) of the robot
        @param env The OpenRAVE environment
        """
        return self.action.get_robot(env).CreateRobotStateSaver(
            Robot.SaveParameters.LinkTransformation)

    def jump(self, env):
        """
        Move the robot to the last configuration in the path
        @param env The OpenRAVE environment
        """
        robot = self.action.get_robot(env)

        with robot.CreateRobotStateSaver(Robot.SaveParameters.ActiveDOF):
            robot.SetActiveDOFs(self.action.active_indices)
            cspec = robot.GetActiveConfigurationSpecification()

            lastconfig = self.path.GetWaypoint(self.path.GetNumWaypoints() - 1, cspec)
            robot.SetActiveDOFValues(lastconfig)

    def postprocess(self, env):
        """
        Performing post-processing and timing on the path to produce a trajectory.
        This function also records and logs timing data for the postprocessing.
        @param env The OpenRAVE environment
        @return A PlanExecutableSolution containing the trajectory produced by the postprocessing
        """
        robot = self.action.get_robot(env)

        with robot.CreateRobotStateSaver(Robot.SaveParameters.ActiveDOF):
            from prpy.util import Timer, SetTrajectoryTags
            from prpy.planning.base import Tags

            path_copy = CopyTrajectory(self.path, env=env)
            with Timer() as timer:
                traj = robot.PostProcessPath(path_copy, timelimit=5.)

            SetTrajectoryTags(
                traj, {Tags.POSTPROCESS_TIME: timer.get_duration()},
                append=True)

            from ..logging_utils import log_postprocess_data
            log_postprocess_data(traj, self.action.get_name())

        return PlanExecutableSolution(solution=self, traj=traj)


class PlanAction(Action):
    def __init__(self,
                 robot,
                 active_indices,
                 active_manipulator,
                 method,
                 args=None,
                 kwargs=None,
                 planner=None,
                 name=None,
                 checkpoint=False):
        """
        @param robot The OpenRAVE robot to plan for
        @param active_indices The robot DOF indices to plan for
        @param active_manipulator The manipulator to plan for
        @param method The planning_method to call (e.g.: PlanToConfiguration)
        @param args Arguments to be passed to the planning_method
        @param kwargs Keyword arguments to be passed to the planning_method
        @param planner The specific planner to call the planning method on,
        if None robot.planner is used
        @param name The name of this action
        @param checkpoint True if this action is a checkpoint
        plan again would result in the exact same trajectory
        """
        super(PlanAction, self).__init__(name=name, checkpoint=checkpoint)

        self._robot = to_key(robot)
        self.active_indices = active_indices
        self._active_manipulator = to_key(active_manipulator)
        self.method = method

        # TODO: Some entries in here need to be wrapped with to_key.
        self.args = args if args is not None else list()
        self.kwargs = kwargs if kwargs is not None else dict()
        self.planner = planner

    def get_robot(self, env):
        """
        @param env The OpenRAVE environment
        @return The OpenRAVE robot in this environment that corresponds to the
        robot used by this action
        """
        return from_key(env, self._robot)

    def get_manipulator(self, env):
        """
        @param env The OpenRAVE environment
        @return The OpenRAVE manipulator in this environment that corresponds to the
        active_manipulator defined for this action
        """
        return from_key(env, self._active_manipulator)

    def plan(self, env):
        """
        Execute the defined planning_method on the defined planner
        @return A PlanSolution
        @throws An ActionError if an error is encountered while planning
        """
        from prpy.planning import PlanningError

        robot = self.get_robot(env)
        active_manipulator = self.get_manipulator(env)
        planner = self.planner if self.planner is not None else robot.planner

        with robot.CreateRobotStateSaver(
            Robot.SaveParameters.ActiveDOF |
            Robot.SaveParameters.ActiveManipulator):

            robot.SetActiveDOFs(self.active_indices)

            if active_manipulator is not None:
                robot.SetActiveManipulator(active_manipulator)

            planning_method = getattr(planner, self.method)
            try:
                from prpy.util import Timer, SetTrajectoryTags, GetTrajectoryTags
                from prpy.planning.base import Tags
                with Timer() as timer:
                    path = planning_method(robot, *self.args, **self.kwargs)
                SetTrajectoryTags(
                    path, {Tags.PLAN_TIME: timer.get_duration()}, append=True)

                # Mark this action as deterministic based on the traj tags
                path_tags = GetTrajectoryTags(path)
                deterministic = path_tags.get(Tags.DETERMINISTIC_ENDPOINT,
                                              None)
                if deterministic is None:
                    LOGGER.warn(
                        "Trajectory does not have DETERMINISTIC_ENDPOINT flag set. "
                        "Assuming non-deterministic.")
                    deterministic = False

                from ..logging_utils import log_plan_data
                log_plan_data(path, self.get_name())

            except PlanningError as err:
                raise ActionError(str(err), deterministic=err.deterministic)

        return PlanSolution(
            action=self, path=path, deterministic=deterministic)


class PlanToTSRAction(PlanAction):
    def __init__(self,
                 robot,
                 obj,
                 tsr_name,
                 active_indices,
                 active_manipulator,
                 tsr_args=None,
                 args=None,
                 kwargs=None,
                 planner=None,
                 name=None):
        """
        Runs a PlanToTSR method
        @param robot The robot the tsr is defined on
        @param obj The object the tsr is defined on
        @param tsr_name The name of the tsr
        @param active_indices The indices of the robot that should be active during planning
        @param active_manipulator The manipulator of the robot that should be active during planning
        @param tsr_args Extra arguments to pass to the tsr library for computing the tsr
        @param args Extra arguments to pass to the planner
        @param kwargs Extra keyword arguments to pass to the planner
        @param planner A specific planner to use - if None, robot.planner is used
        """
        super(PlanToTSRAction, self).__init__(
            robot,
            active_indices,
            active_manipulator,
            method='PlanToTSR',
            args=args,
            kwargs=kwargs,
            planner=planner,
            name=name)
        self._obj = to_key(obj)
        self.tsr_name = tsr_name
        self.orig_args = args if args is not None else list()
        self.tsr_args = tsr_args if tsr_args is not None else dict()

    def get_obj(self, env):
        """
        @param env The OpenRAVE environment
        @param obj The object in the environment that the TSR is defined on
        """
        return from_key(env, self._obj)

    def plan(self, env):
        """
        Lookup the appropriate TSR in the tsrlibrary, then plan to that TSR
        @return A PlanSolution object
        @throws An ActionException if an error is encountered during planning
        """
        obj = self.get_obj(env)
        robot = self.get_robot(env)
        active_manipulator = self.get_manipulator(env)

        if active_manipulator is not None:
            self.tsr_args['manip'] = active_manipulator

        tsr_list = robot.tsrlibrary(obj, self.tsr_name, **self.tsr_args)
        self.args = [tsr_list] + self.orig_args

        import prpy
        with prpy.viz.RenderTSRList(tsr_list, env):
            return super(PlanToTSRAction, self).plan(env)


class PlanEndEffectorStraight(PlanAction):
    def __init__(self,
                 robot,
                 active_indices,
                 active_manipulator,
                 distance,
                 args=None,
                 kwargs=None,
                 planner=None,
                 name=None):
        """
        Runs a PlanToTSR method
        @param robot The robot the tsr is defined on
        @param active_indices The indices of the robot that should be active during planning
        @param active_manipulator The manipulator of the robot that should be active during planning
        @param distance The distance to move the end-effector
        @param args Extra arguments to pass to the planner
        @param kwargs Extra keyword arguments to pass to the planner
        @param planner A specific planner to use - if None, robot.planner is used
        """
        super(PlanEndEffectorStraight, self).__init__(
            robot,
            active_indices,
            active_manipulator,
            method='PlanToEndEffectorOffset',
            args=args,
            kwargs=kwargs,
            planner=planner,
            name=name)

        self.distance = distance

    def plan(self, env):
        """
        Lookup the appropriate TSR in the tsrlibrary, then plan to that TSR
        @return A PlanSolution object
        @throws An ActionException if an error is encountered during planning
        """
        active_manipulator = self.get_manipulator(env)

        # Move in the direction of the z-axis of the manipulator
        direction = active_manipulator.GetEndEffectorTransform()[:3, 2]
        self.kwargs['direction'] = direction
        self.kwargs['distance'] = self.distance

        return super(PlanEndEffectorStraight, self).plan(env)
