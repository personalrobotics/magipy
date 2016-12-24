from contextlib import nested
import logging

import numpy as np

from openravepy import Robot, KinBody
from prpy.planning import PlanningError
from prpy.planning.base import Tags
from prpy.util import ComputeEnabledAABB, CopyTrajectory, GetTrajectoryTags, SetTrajectoryTags, Timer

from magi.actions.base import Action, ActionError, ExecutableSolution, Solution, from_key, to_key

try:
    from or_pushing.discrete_search_push_planner import DiscreteSearchPushPlanner
    from or_pushing.push_planner_module import PushPlannerModule
    HAS_RANDOMIZED_REARRANGEMENT_PLANNING = True
except ImportError:
    HAS_RANDOMIZED_REARRANGEMENT_PLANNING = False

LOGGER = logging.getLogger(__name__)


class RearrangeObjectExecutableSolution(ExecutableSolution):
    def __init__(self, solution, traj):
        """
        @param solution The solution this ExecutableSolution was generated from
        @param traj The trajectory to execute
        """
        super(RearrangeObjectExecutableSolution, self).__init__(solution)
        self.traj = traj

    @property
    def action(self):
        return self.solution.action

    def execute(self, env, simulate):
        """
        Executes a rearrangement trajectory.
        @param env The OpenRAVE environment
        @param simulate If True, execute the action in simulation
        """
        # TODO: Assert the action is of type RearrangeAction

        if not HAS_RANDOMIZED_REARRANGEMENT_PLANNING:
            raise ActionError(
                "Unable to import or_pushing. Is the randomized_rearrangement_planning"
                "repository checked out in your workspace?")

        robot = self.action.get_robot(env)
        manip = self.action.get_manipulator(env)
        planner_kwargs = self.action.get_planner_kwargs()
        ompl_path = self.action.get_planned_path()

        traj_copy = CopyTrajectory(self.traj, env=env)

        with robot.CreateRobotStateSaver(
            Robot.SaveParameters.ActiveManipulator):
            robot.SetActiveManipulator(manip)

            module = PushPlannerModule(env, robot)
            module.Initialize(**planner_kwargs)

        if not simulate:
            # Execute the trajectory
            with Timer() as timer:
                traj = robot.ExecuteTrajectory(traj_copy)
            SetTrajectoryTags(
                traj, {Tags.EXECUTION_TIME: timer.get_duration()}, append=True)

            module.SetFinalObjectPoses(ompl_path)
        else:
            # We want to simulate the full trajectory, including the
            # pushing motions of the object. A method to do this is exposed via
            # the module
            traj = traj_copy
            with Timer() as timer:
                module.ExecutePath(ompl_path)
            SetTrajectoryTags(
                traj, {Tags.EXECUTION_TIME: timer.get_duration()}, append=True)
        return traj


class RearrangeObjectSolution(Solution):
    def __init__(self, action, traj, deterministic):
        """
        @param action The RearrangeAction that generated this solution
        @param traj The trajectory planned by the action
        @param deterministic True if the traj for this Solution was generated
        with a deterministic planner
        """
        super(RearrangeObjectSolution, self).__init__(
            action, determinsitic=deterministic)
        self.traj = traj

    def save(self, env):
        """
        Save the pose of the robot and all the movable objects
        @param env The OpenRAVE environment
        """
        robot = self.action.get_robot(env)
        movable_objects = self.action.get_movables(env)
        pushed_obj = self.action.get_pushed_object(env)

        savers = [
            robot.CreateRobotStateSaver(
                Robot.SaveParameters.LinkTransformation)
        ]
        savers += [
            b.CreateKinBodyStateSaver(
                KinBody.SaveParameters.LinkTransformation)
            for b in movable_objects
        ]
        savers += [
            pushed_obj.CreateKinBodyStateSaver(
                KinBody.SaveParameters.LinkTransformation)
        ]
        return nested(*savers)

    def jump(self, env):
        """
        Move the manipulator and objects to their poses
        at the end of the pushing trajectory
        """
        if not HAS_RANDOMIZED_REARRANGEMENT_PLANNING:
            raise ActionError(
                "Unable to import or_pushing. Is the randomized_rearrangement_planning"
                "repository checked out in your workspace?")

        robot = self.action.get_robot(env)
        planner_kwargs = self.action.get_planner_kwargs()
        ompl_path = self.action.get_planned_path()

        module = PushPlannerModule(env, robot)
        module.Initialize(**planner_kwargs)
        module.SetFinalObjectPoses(ompl_path)

    def postprocess(self, env):
        """
        Run post-processing on the trajectory generated for the rearrangement
        """
        # TODO: PostProcessPath?

        return RearrangeObjectExecutableSolution(solution=self, traj=self.traj)


class RearrangeObjectAction(Action):
    def __init__(self,
                 manipulator,
                 obj,
                 on_obj,
                 goal_pose,
                 goal_epsilon=0.02,
                 movables=None,
                 required=True,
                 name=None,
                 kwargs=None):
        """
        @param manipulator The manipulator to use to push the object
        @param obj The object being pushed
        @param on_obj The support surface to push the object along
        @param goal_pose The [x,y] final pose of the object
        @param goal_epsilon The maximum distance away from pose the object can be at
         the end of the trajectory
        @param movables The other objects in the environment that can be moved
         while attempting to achieve the goal
        @param required If True, and a plan cannot be found, raise an ActionError. If False,
        and a plan cannot be found, return an empty solution so execution can proceed normally.
        @param name The name of the action
        @param kwargs Additional keyword arguments to be
        passed to the planner generating the pushing trajectory
        """
        super(RearrangeObjectAction, self).__init__(name=name)
        self._manipulator = to_key(manipulator)
        self._object = to_key(obj)
        self._on_object = to_key(on_obj)

        if movables is None:
            movables = []
        self._movables = [to_key(m) for m in movables]

        self.goal_pose = goal_pose
        self.goal_epsilon = goal_epsilon
        self.required = required

        self.kwargs = kwargs if kwargs is not None else dict()
        self.ompl_path = None
        self.planner_params = None

    def get_manipulator(self, env):
        return from_key(env, self._manipulator)

    def get_robot(self, env):
        return self.get_manipulator(env).GetRobot()

    def get_movables(self, env):
        objs = [from_key(env, m) for m in self._movables]
        return objs

    def get_pushed_object(self, env):
        return from_key(env, self._object)

    def get_on_obj(self, env):
        return from_key(env, self._on_object)

    def get_planner_kwargs(self):
        return self.planner_params

    def get_planned_path(self):
        return self.ompl_path

    def plan(self, env):
        """
        Plan to rearrange objects
        @param env The OpenRAVE environment
        """
        if not HAS_RANDOMIZED_REARRANGEMENT_PLANNING:
            raise ActionError(
                "Unable to import or_pushing. Is the randomized_rearrangement_planning"
                "repository checked out in your workspace?")

        on_obj = self.get_on_obj(env)
        manipulator = self.get_manipulator(env)
        robot = self.get_robot(env)
        movables = self.get_movables(env)
        obj = self.get_pushed_object(env)

        # Make the state bounds be at the edges of the table
        with env:
            on_obj_aabb = ComputeEnabledAABB(on_obj)
        on_obj_pos = on_obj_aabb.pos()
        on_obj_extents = on_obj_aabb.extents()
        sbounds = {
            'high': [
                on_obj_pos[0] + on_obj_extents[0],
                on_obj_pos[1] + on_obj_extents[1], 2. * np.pi
            ],
            'low': [
                on_obj_pos[0] - on_obj_extents[0],
                on_obj_pos[1] - on_obj_extents[1], 0
            ]
        }

        # Assume we want to keep the current orientation and heigh of the manipulator
        #   throughout the push
        with env:
            ee_pushing_transform = manipulator.GetTransform()
        ee_pushing_transform[:2, 3] = [0., 0.]  # ignore x,y pose

        planner = DiscreteSearchPushPlanner(env)
        try:
            with robot.CreateRobotStateSaver():
                robot.SetActiveManipulator(manipulator)
                traj = planner.PushToPose(
                    robot,
                    obj,
                    self.goal_pose,
                    state_bounds=sbounds,
                    movable_objects=movables,
                    pushing_model='quasistatic',
                    pushing_manifold=ee_pushing_transform.flatten().tolist(),
                    goal_radius=self.goal_epsilon,
                    **self.kwargs)
            self.ompl_path = planner.GetPlannedPath()
            self.planner_params = planner.GetPlannerParameters()

            # Mark this action as deterministic based on the traj tags
            path_tags = GetTrajectoryTags(path)
            deterministic = path_tags.get(Tags.DETERMINISTIC_ENDPOINT, None)
            if deterministic is None:
                LOGGER.warn(
                    "Trajectory does not have DETERMINISTIC_ENDPOINT flag set. "
                    "Assuming non-deterministic.")
                deterministic = False

        except PlanningError as err:
            filepath = '/tmp'
            planner.WritePlannerData(filepath)
            raise ActionError(str(err), deterministic=err.deterministic)

        if traj is None:
            raise ActionError(
                'Failed to find pushing plan', deterministic=deterministic)

        return RearrangeObjectSolution(
            action=self, traj=traj, deterministic=deterministic)
