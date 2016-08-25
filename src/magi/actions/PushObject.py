from .base import Action, ActionError, ExecutableSolution, Solution, from_key, to_key
from .Plan import PlanExecutableSolution
from openravepy import Robot, KinBody
from prpy.planning import PlanningError
from prpy.util import CopyTrajectory

import logging
logger = logging.getLogger(__name__)

class PushObjectExecutableSolution(ExecutableSolution):
    def __init__(self, solution, traj):
        """
        @param solution The solution this ExecutableSolution was generated from
        @param traj The trajectory to execute
        """
        super(PushObjectExecutableSolution, self).__init__(solution)
        self.traj = traj

    @property
    def action(self):
        return self.solution.action

    def execute(self, env, simulate):
        """
        Executes a pushing trajectory.  This action grabs the object being pushed
        for the duration of the push, then releases it. This ensures
        the object will end up in the proper place relative to the hand in
        the OpenRAVE environment.
        @param env The OpenRAVE environment
        @param simulate If True, execute the action in simulation
        """
        robot = self.action.get_robot(env)
        obj = self.action.get_object(env)
        manipulator = self.action.get_manipulator(env)

        # Put the object in the hand
        import numpy
        obj_pose = numpy.dot(manipulator.GetEndEffectorTransform(),
                             self.solution.obj_in_hand_pose)
        with env:
            obj.SetTransform(obj_pose)

        # Grab the object - the manipulator must be active for the grab
        #  to work properly
        with robot.CreateRobotStateSaver(Robot.SaveParameters.ActiveManipulator):
            robot.SetActiveManipulator(manipulator)
            robot.Grab(obj)

        # Execute the trajectory, forcing the object to move with the arm
        try:
            ret_val = robot.ExecuteTrajectory(
                CopyTrajectory(self.traj, env=env))
        finally:
            # Release the object
            robot.Release(obj)
        
        return ret_val

class PushObjectSolution(Solution):
    def __init__(self, action, path, obj_in_hand_pose, deterministic):
        """
        @param action The Action that generated this solution
        @param path The path planned by the Action
        @param obj_in_hand_pose The pose of the object relative to the hand
        during the push
        @param deterministic True if the path for this Solution was created
        with a determinsitic planner
        """
        super(PushObjectSolution, self).__init__(action, 
                                                 deterministic=deterministic)
        self.path = path
        self.obj_in_hand_pose = obj_in_hand_pose

    def save(self, env):
        """
        Save the pose of the robot and the object we are pushing
        @param env The OpenRAVE environment
        """
        from contextlib import nested
        return nested(self.action.get_robot(env).CreateRobotStateSaver(
            Robot.SaveParameters.LinkTransformation),
                      self.action.get_object(env).CreateKinBodyStateSaver(
                          KinBody.SaveParameters.LinkTransformation) )

    def jump(self, env):
        """
        Move the manipulator to its pose at the end of the pushing trajectory.
        Move the object to its pose at the end of the pushing trajectory
        @param env The OpenRAVE environment
        """
        if self.path is None:
            return

        import numpy

        robot = self.action.get_robot(env)
        obj = self.action.get_object(env)
        manipulator = self.action.get_manipulator(env)

        
        with robot.CreateRobotStateSaver(Robot.SaveParameters.ActiveDOF):
            robot.SetActiveDOFs(manipulator.GetArmIndices())
            cspec = robot.GetActiveConfigurationSpecification()
            
            # Now put the arm at the end of the trajectory and the object in the correct
            # place relative to the arm
            q = self.path.GetWaypoint(self.path.GetNumWaypoints() - 1, cspec)
            robot.SetActiveDOFValues(q)

            with env:
                obj_pose = numpy.dot(manipulator.GetEndEffectorTransform(), self.obj_in_hand_pose)
            
            obj.SetTransform(obj_pose)

    def postprocess(self, env):
        """
        Run post-processing on the trajectory generated for the push
        """
        robot = self.action.get_robot(env)

        with robot.CreateRobotStateSaver(Robot.SaveParameters.ActiveDOF):
            traj = robot.PostProcessPath(
                CopyTrajectory(self.path, env=env))

        return PushObjectExecutableSolution(solution=self, traj=traj)


class PushObjectAction(Action):
    
    def __init__(self, manipulator, obj, push_distance=0.1, required=True, name=None,
                 args=None, kwargs=None):
        """
        @param manipulator The manipulator to use to push the object
        @param obj The object being pushed
        @param push_distance The distance to push the object
        @param required If True, and a plan cannot be found, raise an ActionError. If False,
        and a plan cannot be found, return an empty solution so execution can proceed normally.
        @param name The name of the action
        @param args Additional arguments to be passed to the planner generating the pushing trajectory
        @param kwargs Additional keyword arguments to be passed to the planner generating the pushing trajectory
        """
        super(PushObjectAction, self).__init__(name=name)
        self._manipulator = to_key(manipulator)
        self._object = to_key(obj)
        self.push_distance = push_distance
        self.required=required

        self.args = args if args is not None else list()
        self.kwargs = kwargs if kwargs is not None else dict()

    def get_manipulator(self, env):
        return from_key(env, self._manipulator)
    
    def get_robot(self, env):
        return self.get_manipulator(env).GetRobot()
    
    def get_object(self, env):
        return from_key(env, self._object)

    def get_push_direction(self, manipulator):
        """
        Helper function to get the direction of arm movement
        @param manipulator The manipulator that will move
        @return The direction the palm is facing (z-axis for HERB)
        """
        with manipulator.GetRobot().GetEnv():
            ee_pose = manipulator.GetEndEffectorTransform()
        return ee_pose[:3,2]  # direction the palm (z-axis) is facing

    def obj_to_hand(self, obj, manipulator, env, push_direction, push_distance):
        """
        Move the object so it is right next to, but not touching the manipulator
        The object is translated in the direction opposite the push_direction
        until it makes contact with the manipulator, at which point it is
        translated back in the push_direction until it is just out of contact.
        """
        with env:
            obj_in_world = obj.GetTransform()
            robot = manipulator.GetRobot()

            # First move back until collision
            stepsize = 0.01
            total_distance = 0.0
            while not env.CheckCollision(robot, obj) and total_distance <= push_distance:
                obj_in_world[:3,3] -= stepsize*push_direction
                total_distance += stepsize
                obj.SetTransform(obj_in_world)
            
            if total_distance > push_distance:
                # No contact made, return the object to its original pose
                obj.SetTransform(obj_in_world)
            else:
                # Contact occurred, move forward until just out of collision
                stepsize = 0.005
                while env.CheckCollision(robot, obj):
                    obj_in_world[:3,3] += stepsize*push_direction
                    obj.SetTransform(obj_in_world)

    def plan(self, env):
        """
        Plan a straight line pushing trajectory.
        @param env The OpenRAVE environment
        """
        from prpy.rave import AllDisabled
        import numpy

        obj = self.get_object(env)
        manipulator = self.get_manipulator(env)
        robot = self.get_robot(env)

        path = None
        obj_pose = obj.GetTransform()
        with robot.CreateRobotStateSaver(
                Robot.SaveParameters.ActiveDOF | 
                Robot.SaveParameters.ActiveManipulator |
                Robot.SaveParameters.GrabbedBodies), \
        obj.CreateKinBodyStateSaver(
            KinBody.SaveParameters.LinkTransformation):
            
                
            robot.SetActiveManipulator(manipulator)

            # Get the direction of the push
            push_direction = self.get_push_direction(manipulator)
            
            # Move the object into the hand
            self.obj_to_hand(obj, manipulator, env, push_direction, self.push_distance)
            robot.Grab(obj)

            # Save the pose of the object relative to the end-effector
            obj_in_hand = numpy.dot(numpy.linalg.inv(manipulator.GetEndEffectorTransform()),
                                    obj.GetTransform())

            try:
                path = manipulator.PlanToEndEffectorOffset(direction = push_direction,
                                                           distance = self.push_distance,
                                                           *self.args, **self.kwargs)

                # Mark this action as deterministic based on the traj tags
                from prpy.util import GetTrajectoryTags
                from prpy.planning.base import Tags
                path_tags = GetTrajectoryTags(path)
                deterministic = path_tags.get(Tags.DETERMINISTIC_ENDPOINT, None)
                if deterministic is None:
                    logger.warn("Trajectory does not have DETERMINISTIC_ENDPOINT flag set. "
                                "Assuming non-deterministic.")
                    deterministic = False


            except PlanningError, e:
                deterministic = e.deterministic
                if self.required:
                    raise ActionError(str(e), deterministic=deterministic)
                else:
                    logger.warn('Could not find a plan for straight line push. Ignoring.')
           
        return PushObjectSolution(action=self, path=path, obj_in_hand_pose=obj_in_hand, deterministic=deterministic)
