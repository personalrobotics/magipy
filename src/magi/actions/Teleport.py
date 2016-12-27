"""Define TeleportAction and TeleportSolution."""

from openravepy import KinBody

from magi.actions.base import Action, ExecutableSolution, Solution, from_key, to_key


class TeleportSolution(Solution, ExecutableSolution):
    """A Solution that teleports an object by a relative offset."""

    def __init__(self, action, obj_transform):
        """
        @param action: Action that created this Solution
        @param obj_transform: pose the object should teleport to
        """
        Solution.__init__(self, action, deterministic=True)
        ExecutableSolution.__init__(self, self)
        self.obj_transform = obj_transform

    def save(self, env):
        """
        Save object pose.

        @param env: OpenRAVE environment
        @return context manager
        """
        obj = self.action.get_obj(env)
        return obj.CreateKinBodyStateSaver(
            KinBody.SaveParameters.LinkTransformation)

    def jump(self, env):
        """
        Move the object to its pose after teleporting.

        @param env: OpenRAVE environment
        """
        obj = self.action.get_obj(env)
        obj.SetTransform(self.obj_transform)

    def postprocess(self, env):
        """
        Do nothing.

        @param env: OpenRAVE environment
        @return a copy of this object
        """
        return TeleportSolution(self.action, self.obj_transform)

    def execute(self, env, simulate):
        """
        Teleport the object.

        @param env: OpenRAVE environment
        @param simulate: ignored
        """
        obj = self.action.get_obj(env)
        obj.SetTransform(self.obj_transform)

        robot = self.action.get_robot(env)
        if robot is not None:
            robot.Release(obj)
            robot.Grab(obj)


class TeleportAction(Action):
    """An Action that teleports an object by a relative offset."""

    # FIXME: why are there **kwargs here?
    def __init__(self, obj, xyz_offset, robot=None, name=None, **kwargs):
        """
        @param obj: object to teleport
        @param xyz: relative xyz offset (3x1 vector)
        @param robot: OpenRAVE robot
          if not None, have the robot grab the object after teleporting
        @param name: name of the action
        """
        super(TeleportAction, self).__init__(name=name)

        if robot is None:
            self._robot = None
        else:
            self._robot = to_key(robot)
        self._obj = to_key(obj)
        self.xyz_offset = xyz_offset

    def get_robot(self, env):
        """
        Look up and return the robot in the environment.

        @param env: OpenRAVE environment
        @return the robot to grab with
        """
        if self._robot is None:
            return None
        return from_key(env, self._robot)

    def get_obj(self, env):
        """
        Look up and return the object in the environment.

        @param env: OpenRAVE environment
        @return the object to teleport
        """
        return from_key(env, self._obj)

    def plan(self, env):
        """
        No real planning is performed; just find the new pose of the object.

        @param env: OpenRAVE environment
        @return a TeleportSolution
        """
        obj = self.get_obj(env)
        obj_transform = obj.GetTransform()
        obj_transform[:3, 3] += self.xyz_offset

        return TeleportSolution(self, obj_transform=obj_transform)
