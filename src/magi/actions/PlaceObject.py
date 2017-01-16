"""Defines the PlaceObjectAction."""

from magi.actions.Plan import PlanAction
from magi.actions.base import from_key, to_key


class PlaceObjectAction(PlanAction):
    """
    An Action that places an object onto another object by using the 'point_on'
    and 'place' TSRs.
    """

    def __init__(self,
                 robot,
                 obj,
                 on_obj,
                 active_indices,
                 active_manipulator,
                 height=0.04,
                 allowed_tilt=0.0,
                 args=None,
                 kwargs=None,
                 planner=None,
                 name=None):
        """
        @param robot: robot
        @param obj: object to place
        @param on_obj: object 'obj' should be placed on
        @param active_indices: indices of the robot that should be active during
          planning
        @param active_manipulator: manipulator of the robot that should be
          active during planning
        @param height: height relative to on_obj to place the object (meters)
        @param allowed_tilt: amount of allowed tilt in the object when placing
          it (radians)
        @param args: extra arguments to pass to the planner
        @param kwargs: extra keyword arguments to pass to the planner
        @param planner: specific planner to use
          if None, robot.planner is used
        @param name: name of the action
        """
        super(PlaceObjectAction, self).__init__(
            robot,
            active_indices,
            active_manipulator,
            method='PlanToTSR',
            args=args,
            kwargs=kwargs,
            planner=planner,
            name=name)
        self._obj = to_key(obj)
        self._on_obj = to_key(on_obj)
        self.orig_args = args if args is not None else list()

        self.height = height
        self.allowed_tilt = allowed_tilt

    def get_obj(self, env):
        """
        Look up and return the object to place in the environment.

        @param env: OpenRAVE environment
        @return KinBody to place
        """
        return from_key(env, self._obj)

    def get_on_obj(self, env):
        """
        Look up and return the object to place on in the environment.

        @param env: OpenRAVE environment
        @return KinBody to place on
        """
        return from_key(env, self._on_obj)

    def plan(self, env):
        """
        Generate a TSR for placing an object on another object.

        @param env: OpenRAVE environment
        @return a PlanSolution
        """
        obj = self.get_obj(env)
        on_obj = self.get_on_obj(env)
        robot = self.get_robot(env)
        active_manipulator = self.get_manipulator(env)

        # Get a tsr to sample poses for placement
        obj_extents = obj.ComputeAABB().extents()
        obj_radius = max(obj_extents[0], obj_extents[1])

        on_obj_tsr = robot.tsrlibrary(
            on_obj,
            'point_on',
            padding=obj_radius,
            manip=active_manipulator,
            allowed_tilt=self.allowed_tilt,
            vertical_offset=self.height)

        # Now use this to get a tsr for end-effector poses
        place_tsr = robot.tsrlibrary(
            obj,
            'place',
            pose_tsr_chain=on_obj_tsr[0],
            manip=active_manipulator)

        self.args = [place_tsr] + self.orig_args

        return super(PlaceObjectAction, self).plan(env)
