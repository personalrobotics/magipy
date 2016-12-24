from magi.actions.Plan import PlanAction
from magi.actions.base import from_key, to_key


class PlaceObjectAction(PlanAction):
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
        Places an object onto another object by using the 'point_on' and 'place' tsrs
        @param robot The robot
        @param obj The object to place
        @param on_obj The object 'obj' should be place on
        @param active_indices The indices of the robot that should be active during planning
        @param active_manipulator The manipulator of the robot that should be active during planning
        @param height The height relative to the on_obj to place the object (meters)
        @param allowed_tilt The amount of allowed tilt in the object when placing it (radians)
        @param args Extra arguments to pass to the planner
        @param kwargs Extra keyword arguments to pass to the planner
        @param planner A specific planner to use - if None, robot.planner is used
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
        return from_key(env, self._obj)

    def get_on_obj(self, env):
        return from_key(env, self._on_obj)

    def plan(self, env):
        """
        Generate a tsr for placing the object on another object
        @param env The OpenRAVE environment
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
