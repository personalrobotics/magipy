from .base import Action, ExecutableSolution, Solution, from_key, to_key
from openravepy import Robot, KinBody
import numpy


class MoveHandSolution(Solution, ExecutableSolution):
    def __init__(self,
                 action,
                 dof_values,
                 dof_indices,
                 precondition=None,
                 postcondition=None):
        """
        @param action The Action that created this Solution
        @param dof_values A list of dof values to move the hand to
        @param dof_indices A list of the indices of the dofs to move -
        list ordering should correspond to dof_values
        """
        Solution.__init__(
            self,
            action,
            deterministic=True,
            precondition=precondition,
            postcondition=postcondition)
        ExecutableSolution.__init__(self, self)

        self.dof_values = numpy.array(dof_values, dtype='float')
        self.dof_indices = numpy.array(dof_indices, dtype='int')
        self.traj = None

    def save(self, env):
        """
        @param env The OpenRAVE environment
        @return An OpenRAVE RobotStateSaver that saves LinkTransformation values
        - i.e. save the current dof values
        """
        robot = self.action.get_hand(env).GetParent()

        return robot.CreateRobotStateSaver(
            Robot.SaveParameters.LinkTransformation)

    def jump(self, env):
        """
        Move the dof_indices specified when creating the Solution object
        to the specified dof_values, or the last valid configuration
        as the hand moves toward those values.
        @param env The OpenRAVE environment
        """
        hand = self.action.get_hand(env)
        robot = hand.GetParent()

        with robot.CreateRobotStateSaver(Robot.SaveParameters.ActiveDOF):
            if self.traj is None:
                self.traj = self._get_trajectory(robot, self.dof_values,
                                                 self.dof_indices)
            robot.SetActiveDOFs(self.dof_indices)
            cspec = robot.GetActiveConfigurationSpecification()

            waypoint = self.traj.GetWaypoint(self.traj.GetNumWaypoints() - 1)
            robot.SetActiveDOFValues(
                cspec.ExtractJointValues(waypoint, robot, self.dof_indices))

    def postprocess(self, env):
        """
        Does nothing.
        @return This object, unchanged.
        """
        return self

    def execute(self, env, simulate):
        """
        Executes the motion of the hand
        @param env The OpenRAVE environment
        @param simulate If true, the execution should be done in simulation
        """
        hand = self.action.get_hand(env)
        robot = hand.GetParent()

        if simulate:
            if self.traj is None:
                with env:
                    self.traj = self._get_trajectory(robot, self.dof_values,
                                                     self.dof_indices)
            # Simply move hand to the last dof_values on the trajectory.
            waypoint = self.traj.GetWaypoint(self.traj.GetNumWaypoints() -
                                             1).tolist()
            hand = self.action.get_hand(env)
            indices = hand.GetIndices()
            dof_values = []
            dof_indices = list(self.dof_indices)
            for i in indices:
                try:
                    dof_values += [waypoint[dof_indices.index(i)]]
                except ValueError:
                    dof_values += [None]

            hand.MoveHand(*dof_values, timeout=None)

        else:
            hand.MoveHand(*self.dof_values, timeout=None)

    def _get_trajectory(self, robot, q_goal, dof_indices):
        """
        Generates a hand trajectory that attempts to move the specified dof_indices
        to the configuration indicated in q_goal.  The trajectory will move the hand
        until q_goal is reached or an invalid configuration - usually collision
        - is detected.
        @param robot The OpenRAVE robot
        @param q_goal The goal configuration
        @param dof_indices The indices of the dofs specified in q_goal
        """
        from openravepy import (
            CollisionAction,
            CollisionOptions,
            CollisionOptionsStateSaver,
            CollisionReport,
            RaveCreateTrajectory, )

        def collision_callback(report):
            colliding_links.update([report.plink1, report.plink2])
            return CollisionAction.Ignore

        env = robot.GetEnv()
        collision_checker = env.GetCollisionChecker()
        dof_indices = numpy.array(dof_indices, dtype='int')

        report = CollisionReport()

        handle = env.RegisterCollisionCallback(collision_callback)

        with robot.CreateRobotStateSaver(
            Robot.SaveParameters.ActiveDOF
            | Robot.SaveParameters.LinkTransformation),\
             CollisionOptionsStateSaver(collision_checker,
                                        CollisionOptions.ActiveDOFs):

            robot.SetActiveDOFs(dof_indices)
            q_prev = robot.GetActiveDOFValues()

            # Create the output trajectory.
            cspec = robot.GetActiveConfigurationSpecification('linear')
            cspec.AddDeltaTimeGroup()

            traj = RaveCreateTrajectory(env, '')
            traj.Init(cspec)

            # Velocity that each joint will be moving at while active.
            qd = numpy.sign(q_goal - q_prev) * robot.GetActiveDOFMaxVel()

            # Duration required for each joint to reach the goal.
            durations = (q_goal - q_prev) / qd
            durations[qd == 0.] = 0.

            t_prev = 0.
            events = numpy.concatenate((
                [0.],
                durations,
                numpy.arange(0., durations.max(), 0.01), ))
            mask = numpy.array([True] * len(q_goal), dtype='bool')

            for t_curr in numpy.unique(events):
                robot.SetActiveDOFs(dof_indices[mask])

                # Disable joints that have reached the goal.
                mask = numpy.logical_and(mask, durations >= t_curr)

                # Advance the active DOFs.
                q_curr = q_prev.copy()
                q_curr[mask] += (t_curr - t_prev) * qd[mask]
                robot.SetDOFValues(q_curr, dof_indices)

                # Check for collision. This only operates on the active DOFs
                # because the ActiveDOFs flag is set. We use a collision
                # callback to build a set of all links in collision.
                colliding_links = set()
                env.CheckCollision(robot, report=report)
                robot.CheckSelfCollision(report=report)

                # Check which DOF(s) are involved in the collision.
                mask = numpy.logical_and(mask, [
                    not any(
                        robot.DoesAffect(dof_index, link.GetIndex())
                        for link in colliding_links)
                    for dof_index in dof_indices
                ])

                # Revert the DOFs that are in collision.
                # TODO: This doesn't seem to take the links out of collision.
                q_curr = q_prev.copy()
                q_curr[mask] += (t_curr - t_prev) * qd[mask]

                # Add a waypoint to the output trajectory.
                waypoint = numpy.empty(cspec.GetDOF())
                cspec.InsertDeltaTime(waypoint, t_curr - t_prev)
                cspec.InsertJointValues(waypoint, q_curr, robot, dof_indices,
                                        0)
                traj.Insert(traj.GetNumWaypoints(), waypoint)

                t_prev = t_curr
                q_prev = q_curr

                # Terminate if all joints are inactive.
                if not mask.any():
                    break

        del handle
        return traj


class MoveHandAction(Action):
    def __init__(self,
                 hand,
                 dof_values,
                 dof_indices,
                 name=None,
                 precondition=None,
                 postcondition=None):
        """
        Move hand to a desired configuration
        @param hand The hand to move
        @param dof_values The desired configuration for the hand
        @param dof_indices The indices of the dofs specified in dof_values
        @param name The name of the action
        """
        super(MoveHandAction, self).__init__(
            name=name, precondition=precondition, postcondition=postcondition)

        self._hand = to_key(hand)
        self.dof_values = numpy.array(dof_values, dtype='float')
        self.dof_indices = numpy.array(dof_indices, dtype='int')

    def get_hand(self, env):
        """
        Lookup and return the hand in the given OpenRAVE environment
        @param env The OpenRAVE environment
        @param A prpy Hand
        """
        return from_key(env, self._hand)

    def plan(self, env):
        """
        No planning is performed. Just creates and returns a MoveHandSolution.
        @return A MoveHandSolution
        """
        return MoveHandSolution(
            action=self,
            dof_values=self.dof_values,
            dof_indices=self.dof_indices)


def close_hand_action(hand, name=None):
    """
    Helper function that returns a MoveHandAction that closes the fingers
    @param hand The hand to close
    @param name The name of the action - if None the name defaults
    to close_hand_action
    """
    if name is None:
        name = 'close_hand_action'

    return MoveHandAction(
        name=name,
        hand=hand,
        dof_indices=hand.GetIndices()[0:3],
        dof_values=[2.4, 2.4, 2.4])


def open_hand_action(hand, name=None):
    """
    Helper function that returns a MoveHandAction that opens the fingers completely
    @param hand The hand to open
    @param name The name of the action - if None the name defaults to
    open_hand_action
    """
    if name is None:
        name = 'open_hand_action'

    return MoveHandAction(
        name=name,
        hand=hand,
        dof_indices=hand.GetIndices()[0:3],
        dof_values=[0., 0., 0.])


class GrabObjectSolution(MoveHandSolution):
    def __init__(self,
                 action,
                 obj,
                 dof_values,
                 dof_indices,
                 precondition=None,
                 postcondition=None):
        """
        @param action The Action that generated this solution
        @param obj The object to grab
        @param dof_values The configuration to move the hand to before performing the grab
        @param dof_indices The DOF indices that correspond to the values in dof_values
        """
        super(GrabObjectSolution, self).__init__(
            action, dof_values, dof_indices, precondition, postcondition)
        self._obj = to_key(obj)

    def get_obj(self, env):
        """
        Lookup the object to grasp in the environment and return it
        """
        return from_key(env, self._obj)

    def save(self, env):
        """
        @return The result of MoveHandSolution.save and a RobotStateSaver
        that saves the set of GrabbedBodies
        """
        robot = self.action.get_hand(env).GetParent()

        from contextlib import nested
        return nested(
            super(GrabObjectSolution, self).save(env),
            robot.CreateRobotStateSaver(Robot.SaveParameters.GrabbedBodies))

    def jump(self, env):
        """
        Move the hand to the desired dof_values, or the last valid configuration
        on the way to these values. Grab the object.
        @param env The OpenRAVE environment
        """
        hand = self.action.get_hand(env)
        robot = hand.GetParent()
        obj = self.get_obj(env)
        manipulator = self.action.get_manipulator(env)

        super(GrabObjectSolution, self).jump(env)

        # Now grab the object
        with robot.CreateRobotStateSaver(
            Robot.SaveParameters.ActiveManipulator):
            robot.SetActiveManipulator(manipulator)
            if obj in robot.GetGrabbed():
                robot.Release(obj)
            robot.Grab(obj)

    def execute(self, env, simulate):
        """
        Close the hand and grab the object.
        @param env The OpenRAVE environment
        @param simulate If True, simulate execution
        """

        # Close the hand
        super(GrabObjectSolution, self).execute(env, simulate)

        # Grab the object
        obj = self.get_obj(env)
        manipulator = self.action.get_manipulator(env)
        robot = manipulator.GetRobot()

        with robot.CreateRobotStateSaver(
            Robot.SaveParameters.ActiveManipulator):
            # Manipulator must be active for grab to work
            robot.SetActiveManipulator(manipulator)
            if obj in robot.GetGrabbed():
                robot.Release(obj)
            robot.Grab(obj)


class GrabObjectAction(MoveHandAction):
    def __init__(self,
                 hand,
                 obj,
                 dof_values=None,
                 dof_indices=None,
                 name=None,
                 precondition=None,
                 postcondition=None):
        """
        Close the hand and grab an object.
        @param hand The hand to grab the object
        @param obj The object to grab
        @param dof_values The desired dof_values to move the hand to
        if None, defaults to a fully closed configuration
        @param dof_indices The DOF indices corresponding to configuration specified
         in dof_values, if None defaults to only finger dofs
        """
        if dof_values is None:
            dof_values = [2.4, 2.4, 2.4]
        if dof_indices is None:
            dof_indices = hand.GetIndices()[0:3]

        super(GrabObjectAction, self).__init__(
            hand,
            dof_values=dof_values,
            dof_indices=dof_indices,
            name=name,
            precondition=precondition,
            postcondition=postcondition)
        self._manipulator = to_key(hand.manipulator)
        self._obj = to_key(obj)

    def get_manipulator(self, env):
        """
        Lookup the manipulator in the environment and return it
        """
        return from_key(env, self._manipulator)

    def plan(self, env):
        """
        Does nothing.
        @return A GrabObjectSolution
        """

        # Add precondition posevalidator
        from validators.PoseValidator import ObjectPoseValidator
        obj = from_key(env, self._obj)
        obj_pose_validator = ObjectPoseValidator(obj.GetName(),
                                                 obj.GetTransform())

        return GrabObjectSolution(
            action=self,
            obj=from_key(env, self._obj),
            dof_values=self.dof_values,
            dof_indices=self.dof_indices,
            precondition=obj_pose_validator)


class ReleaseObjectsSolution(MoveHandSolution):
    def __init__(self, action, objs, dof_values, dof_indices):
        """
        @param action The action that created the solution
        @param objs The list of objects that should be released
        @param dof_values The configuration the hand should be moved to before releasing objects
        @param dof_indices The indices of the DOFs specified in dof_valuesx
        """
        super(ReleaseObjectsSolution, self).__init__(action, dof_values,
                                                     dof_indices)
        self._objs = [to_key(o) for o in objs]

    def get_obj(self, env, obj_key):
        return from_key(env, obj_key)

    def save(self, env):
        """
        Saves robot pose, object pose and grabbed bodies
        @param env The OpenRAVE environment
        """
        robot = self.action.get_hand(env).GetParent()

        from contextlib import nested
        return nested(
            super(ReleaseObjectsSolution, self).save(env),
            robot.CreateRobotStateSaver(Robot.SaveParameters.GrabbedBodies))

    def jump(self, env):
        """
        Move to the end of the hand opening trajectory and drop the object
         if necessary
        @param env The OpenRAVE environment
        """
        super(ReleaseObjectsSolution, self).jump(env)

        manipulator = self.action.get_manipulator(env)
        robot = manipulator.GetRobot()

        # Release the object
        for obj_key in self._objs:
            obj = self.get_obj(env, obj_key)
            robot.Release(obj)

    def execute(self, env, simulate):
        """
        Open the hand and release the object. If drop was set
        on the action, also move the object down until it is in
        collision with another object, to simulate a drop.
        @param env The OpenRAVE environment
        @param simulate If true, simulate execution
        """
        # Open the hand
        super(ReleaseObjectsSolution, self).execute(env, simulate)

        manipulator = self.action.get_manipulator(env)
        robot = manipulator.GetRobot()

        # Release the object
        for obj_key in self._objs:
            obj = self.get_obj(env, obj_key)
            robot.Release(obj)


class ReleaseObjectAction(MoveHandAction):
    def __init__(self, hand, obj, dof_values, dof_indices, name=None):
        """
        Open the hand and release an object.
        @param hand The hand to grab the object with
        @param obj The object to release
        @param dof_values The configuration to move the hand to before releasing
        @param dof_idnices The indices of the DOFs specified in dof_values
        @param name The name of the action
        """
        super(ReleaseObjectAction, self).__init__(
            hand, dof_values=dof_values, dof_indices=dof_indices, name=name)

        self._obj = to_key(obj)
        self._manipulator = to_key(hand.manipulator)

    def get_obj(self, env):
        """
        Lookup the object in the environment and return it
        """
        return from_key(env, self._obj)

    def get_manipulator(self, env):
        """
        Lookup the manipulator in the environment and return it
        """
        return from_key(env, self._manipulator)

    def plan(self, env):
        """
        Does nothing.
        @return ReleaseObjectSolution
        """
        return ReleaseObjectsSolution(
            action=self,
            objs=[self.get_obj(env)],
            dof_values=self.dof_values,
            dof_indices=self.dof_indices)


class ReleaseAllGrabbedAction(MoveHandAction):
    def __init__(self, hand, dof_values, dof_indices, name=None):
        """
        Open the hand and release all objects grabbed by the robot
        @param hand The hand to grab the object with
        @param dof_values The configuration to move the hand to before releasing
        @param dof_indices The indices of the DOFs specified in dof_values
        @param name The name of the action
        """
        super(ReleaseAllGrabbedAction, self).__init__(
            hand, dof_values=dof_values, dof_indices=dof_indices, name=name)

        self._manipulator = to_key(hand.manipulator)

    def get_obj(self, env):
        """
        Lookup the object in the environment and return it
        """
        return from_key(env, self._obj)

    def get_manipulator(self, env):
        """
        Lookup the manipulator in the environment and return it
        """
        return from_key(env, self._manipulator)

    def plan(self, env):
        """
        Computes the list of bodies currently grabbed
        by the robot.
        @return A ReleaseObjectSolution
        """
        manipulator = self.get_manipulator(env)
        robot = manipulator.GetRobot()
        grabbed_bodies = robot.GetGrabbed()

        return ReleaseObjectsSolution(
            action=self,
            objs=grabbed_bodies,
            dof_values=self.dof_values,
            dof_indices=self.dof_indices)
