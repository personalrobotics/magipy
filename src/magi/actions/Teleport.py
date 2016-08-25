from .base import Action, ExecutableSolution, Solution, from_key, to_key
from openravepy import Robot, KinBody

class TeleportSolution(Solution, ExecutableSolution):
    def __init__(self, action, obj_transform):
        """
        @param action The TeleportAction that created this Solution
        @param obj_transform The pose the object should teleport to
        """
        Solution.__init__(self, action, deterministic=True)
        ExecutableSolution.__init__(self, self)
        self.obj_transform = obj_transform

    def save(self, env):
        """
        @param env The OpenRAVE environment
        @return An OpenRAVE state saver for the object we are teleporting
        """
        obj = self.action.get_obj(env)
        return obj.CreateKinBodyStateSaver(KinBody.SaveParameters.LinkTransformation)

    def jump(self, env):
        """
        Move the object to its pose after teleporting
        @param env The OpenRAVE environment
        """
        obj = self.action.get_obj(env)
        obj.SetTransform(self.obj_transform)

    def postprocess(self, env):
        """
        Do nothing
        """
        return TeleportSolution(self.action, self.obj_transform)

    def execute(self, env, simulate):
        """
        Teleport the object
        @param env The OpenRAVE environment
        @param simulate ignored
        """
        obj = self.action.get_obj(env)
        obj.SetTransform(self.obj_transform)

        robot = self.action.get_robot(env)
        if robot is not None:
            robot.Release(obj)
            robot.Grab(obj)

        return

class TeleportAction(Action):
    def __init__(self, obj, xyz_offset, robot=None, name=None, **kwargs):
        """
        @param obj The object to teleport
        @param xyz The relative xyz offset (3x1 vector)
        @param name The name for this action
        @param robot If not None, have the robot grab the object after teleporting
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
        @param env The OpenRAVE environment
        @return The object to be teleported
        """
        if self._robot is None:
            return None
        return from_key(env, self._robot)

    def get_obj(self, env):
        """
        @param env The OpenRAVE environment
        @return The object to be teleported
        """
        return from_key(env, self._obj)

    def plan(self, env):
        """
        No real planning is performed. Just find the new pose of the object
        @return TeleportSolution
        """
        obj = self.get_obj(env)
        obj_transform = obj.GetTransform()
        obj_transform[:3,3] += self.xyz_offset
        
        return TeleportSolution(self, obj_transform=obj_transform)
