"""
Define MoveHandAction, MoveHandSolution, CloseHandAction, OpenHandAction,
GrabObjectAction, GrabObjectSolution, ReleaseObjectAction,
ReleaseObjectSolution, and ReleaseAllGrabbedAction.
"""

from contextlib import nested

import numpy as np

from openravepy import (CollisionAction, CollisionOptions, CollisionOptionsStateSaver,
                        CollisionReport, RaveCreateTrajectory, Robot)

from magi.actions.base import Action, ExecutableSolution, Solution, from_key, to_key


class MoveHandSolution(Solution, ExecutableSolution):
    """
    A Solution and ExecutableSolution that moves a hand to a desired
    configuration.
    """

    def __init__(self,
                 action,
                 dof_values,
                 dof_indices,
                 precondition=None,
                 postcondition=None):
        """
        @param action: Action that generated this Solution
        @param dof_values: list of dof values to move the hand to
        @param dof_indices: indices of the dofs specified in dof_values
        @param precondition: Validator that validates preconditions
        @param postcondition: Validator that validates postconditions
        """
        Solution.__init__(
            self,
            action,
            deterministic=True,
            precondition=precondition,
            postcondition=postcondition)
        ExecutableSolution.__init__(self, self)

        self.dof_values = np.array(dof_values, dtype='float')
        self.dof_indices = np.array(dof_indices, dtype='int')
        self.traj = None

    def save(self, env):
        """
        Return a context manager that saves the current robot configuration.

        @param env: OpenRAVE environment
        @return OpenRAVE RobotStateSaver that saves LinkTransformation values
        """
        robot = self.action.get_hand(env).GetParent()

        return robot.CreateRobotStateSaver(
            Robot.SaveParameters.LinkTransformation)

    def jump(self, env):
        """
        Move the dof_indices to the specified dof_values, or the last valid
        configuration as the hand moves toward those values.

        @param env: OpenRAVE environment
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
        Do nothing.

        @param env: OpenRAVE environment
        @return this object, unchanged
        """
        return self

    def execute(self, env, simulate):
        """
        Execute the motion of the hand.

        @param env: OpenRAVE environment
        @param simulate: flag to run in simulation
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

    # FIXME: do q_goal and dof_indices need to be parameters to this function?
    # It seems that they're always set to self.dof_values, self.dof_indices.
    def _get_trajectory(self, robot, q_goal, dof_indices):
        """
        Generates a hand trajectory that attempts to move the specified
        dof_indices to the configuration indicated in q_goal. The trajectory
        will move the hand until q_goal is reached or an invalid configuration
        (usually collision) is detected.

        @param robot: OpenRAVE robot
        @param q_goal: goal configuration (dof values)
        @param dof_indices: indices of the dofs specified in q_goal
        @return hand trajectory
        """
        def collision_callback(report, is_physics):
            """
            Callback for collision detection. Add the links that are in
            collision to the colliding_links set in the enclosing frame.

            @param report: collision report
            @param is_physics: whether collision is from physics
            @return ignore collision action
            """
            colliding_links.update([report.plink1, report.plink2])
            return CollisionAction.Ignore

        env = robot.GetEnv()
        collision_checker = env.GetCollisionChecker()
        dof_indices = np.array(dof_indices, dtype='int')

        report = CollisionReport()

        handle = env.RegisterCollisionCallback(collision_callback)

        with \
            robot.CreateRobotStateSaver(
                Robot.SaveParameters.ActiveDOF
                | Robot.SaveParameters.LinkTransformation), \
            CollisionOptionsStateSaver(
                collision_checker,
                CollisionOptions.ActiveDOFs):

            robot.SetActiveDOFs(dof_indices)
            q_prev = robot.GetActiveDOFValues()

            # Create the output trajectory.
            cspec = robot.GetActiveConfigurationSpecification('linear')
            cspec.AddDeltaTimeGroup()

            traj = RaveCreateTrajectory(env, '')
            traj.Init(cspec)

            # Velocity that each joint will be moving at while active.
            qd = np.sign(q_goal - q_prev) * robot.GetActiveDOFMaxVel()

            # Duration required for each joint to reach the goal.
            durations = (q_goal - q_prev) / qd
            durations[qd == 0.] = 0.

            t_prev = 0.
            events = np.concatenate((
                [0.],
                durations,
                np.arange(0., durations.max(), 0.01), ))
            mask = np.array([True] * len(q_goal), dtype='bool')

            for t_curr in np.unique(events):
                robot.SetActiveDOFs(dof_indices[mask])

                # Disable joints that have reached the goal.
                mask = np.logical_and(mask, durations >= t_curr)

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
                mask = np.logical_and(mask, [
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
                waypoint = np.empty(cspec.GetDOF())
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
    """An Action that moves a hand to a desired configuration."""

    BARRETT_CLOSE_CONFIGURATION = 2.4 * np.ones(3)
    BARRETT_OPEN_CONFIGURATION = np.zeros(3)
    BARRETT_FINGER_INDICES = np.arange(3)

    def __init__(self,
                 hand,
                 dof_values,
                 dof_indices,
                 name=None,
                 precondition=None,
                 postcondition=None):
        """
        Move hand to a desired configuration
        @param hand: EndEffector to move
        @param dof_values: list of dof values to move the hand to
        @param dof_indices: indices of the dofs specified in dof_values
        @param name: name of the action
        @param precondition: Validator that validates preconditions
        @param postcondition: Validator that validates postconditions
        """
        super(MoveHandAction, self).__init__(
            name=name, precondition=precondition, postcondition=postcondition)

        self._hand = to_key(hand)
        self.dof_values = np.array(dof_values, dtype='float')
        self.dof_indices = np.array(dof_indices, dtype='int')

    def get_hand(self, env):
        """
        Look up and return the hand in the given OpenRAVE environment.

        @param env: OpenRAVE environment
        @return a prpy.EndEffector
        """
        return from_key(env, self._hand)

    def plan(self, env):
        """
        No planning is performed; just create and return a MoveHandSolution.

        @param env: OpenRAVE environment
        @return a MoveHandSolution
        """
        return MoveHandSolution(
            action=self,
            dof_values=self.dof_values,
            dof_indices=self.dof_indices)


class CloseHandAction(MoveHandAction):
    """
    Helper subclass of MoveHandAction that closes the fingers.
    """

    def __init__(self, hand, name='CloseHandAction'):
        """
        @param hand: hand to close
        @param name: name of the action, defaults to CloseHandAction
        """
        super(CloseHandAction, self).__init__(
            hand=hand,
            dof_values=self.BARRETT_CLOSE_CONFIGURATION,
            dof_indices=hand.GetIndices()[self.BARRETT_FINGER_INDICES],
            name=name)


class OpenHandAction(MoveHandAction):
    """
    Helper subclass of MoveHandAction that opens the fingers completely.
    """

    def __init__(self, hand, name='OpenHandAction'):
        """
        @param hand: hand to open
        @param name: name of the action, defaults to OpenHandAction
        """
        super(OpenHandAction, self).__init__(
            hand=hand,
            dof_values=self.BARRETT_OPEN_CONFIGURATION,
            dof_indices=hand.GetIndices()[self.BARRETT_FINGER_INDICES],
            name=name)


class GrabObjectSolution(MoveHandSolution):
    """
    A Solution and ExecutableSolution that closes the hand and grabs an object.
    """
    def __init__(self,
                 action,
                 obj,
                 dof_values,
                 dof_indices,
                 precondition=None,
                 postcondition=None):
        """
        @param action: Action that generated this Solution
        @param obj: object to grab
        @param dof_values: configuration to move the hand to before the grab
        @param dof_indices: indices of the dofs specified in dof_values
        @param precondition: Validator that validates preconditions
        @param postcondition: Validator that validates postconditions
        """
        super(GrabObjectSolution, self).__init__(
            action, dof_values, dof_indices, precondition, postcondition)
        self._obj = to_key(obj)

    def get_obj(self, env):
        """
        Look up and return the object to grasp in the environment.

        @param env: OpenRAVE environment
        @return the KinBody to be grasped
        """
        return from_key(env, self._obj)

    def save(self, env):
        """
        Save robot pose and set of grabbed bodies.

        @param env: OpenRAVE environment
        @return context manager; the result of MoveHandSolution.save and a
          RobotStateSaver that saves the set of GrabbedBodies
        """
        robot = self.action.get_hand(env).GetParent()

        return nested(
            super(GrabObjectSolution, self).save(env),
            robot.CreateRobotStateSaver(Robot.SaveParameters.GrabbedBodies))

    def jump(self, env):
        """
        Move the hand to the desired dof_values, or the last valid configuration
        on the way to these values. Grab the object.

        @param env: OpenRAVE environment
        """
        # Close the hand
        super(GrabObjectSolution, self).jump(env)

        hand = self.action.get_hand(env)
        robot = hand.GetParent()
        obj = self.get_obj(env)
        manipulator = self.action.get_manipulator(env)

        # Grab the object
        with robot.CreateRobotStateSaver(
            Robot.SaveParameters.ActiveManipulator):
            robot.SetActiveManipulator(manipulator)
            if obj in robot.GetGrabbed():
                robot.Release(obj)
            robot.Grab(obj)

    def execute(self, env, simulate):
        """
        Close the hand and grab the object.

        @param env: OpenRAVE environment
        @param simulate: flag to run in simulation
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
    """
    An Action that closes the hand and grabs an object.
    """

    def __init__(self,
                 hand,
                 obj,
                 dof_values=None,
                 dof_indices=None,
                 name=None,
                 precondition=None,
                 postcondition=None):
        """
        @param hand: hand to grab the object with
        @param obj: object to grab
        @param dof_values: configuration to move the hand to before the grab
          if None, defaults to a fully closed configuration
        @param dof_indices: indices of the dofs specified in dof_values
          if None defaults to only finger dofs
        @param name: name of the action
        @param precondition: Validator that validates preconditions
        @param postcondition: Validator that validates postconditions
        """
        if dof_values is None:
            dof_values = self.BARRETT_CLOSE_CONFIGURATION
        if dof_indices is None:
            dof_indices = hand.GetIndices()[self.BARRETT_FINGER_INDICES]

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
        Look up and return the manipulator in the environment.

        @param env: OpenRAVE environment
        @return a prpy.EndEffector
        """
        return from_key(env, self._manipulator)

    def plan(self, env):
        """
        No planning; just create and return a GrabObjectSolution.

        @param env: OpenRAVE environment
        @return a GrabObjectSolution
        """
        # Add PoseValidator precondition
        obj = from_key(env, self._obj)
        # TODO: create obj_pose_validator to use as a precondition in
        # GrabObjectSolution
        #obj_pose_validator = ObjectPoseValidator(obj.GetName(),
        #                                         obj.GetTransform())

        return GrabObjectSolution(
            action=self,
            obj=from_key(env, self._obj),
            dof_values=self.dof_values,
            dof_indices=self.dof_indices)


class ReleaseObjectsSolution(MoveHandSolution):
    """
    A Solution and ExecutableSolution that opens the hand and releases objects.
    """

    def __init__(self, action, objs, dof_values, dof_indices):
        """
        @param action: Action that generated this Solution
        @param objs: list of objects that should be released
        @param dof_values: configuration to move the hand to before releasing
        @param dof_indices: indices of the dofs specified in dof_values
        """
        super(ReleaseObjectsSolution, self).__init__(action, dof_values,
                                                     dof_indices)
        self._objs = [to_key(o) for o in objs]

    def get_obj(self, env, obj_key):
        """
        Look up and return a particular object to release in the environment.

        @param env: OpenRAVE environment
        @param obj_key: key for the object to release
        @return the KinBody to be released
        """
        return from_key(env, obj_key)

    def save(self, env):
        """
        Save robot pose and set of grabbed bodies.

        @param env: OpenRAVE environment
        @return context manager; the result of MoveHandSolution.save and a
          RobotStateSaver that saves the set of GrabbedBodies
        """
        robot = self.action.get_hand(env).GetParent()

        return nested(
            super(ReleaseObjectsSolution, self).save(env),
            robot.CreateRobotStateSaver(Robot.SaveParameters.GrabbedBodies))

    def jump(self, env):
        """
        Open the hands to the desired dof_values and drop the object, if
        necessary.

        @param env: OpenRAVE environment
        """
        super(ReleaseObjectsSolution, self).jump(env)

        manipulator = self.action.get_manipulator(env)
        robot = manipulator.GetRobot()

        # Release the objects
        for obj_key in self._objs:
            obj = self.get_obj(env, obj_key)
            robot.Release(obj)

    def execute(self, env, simulate):
        """
        Open the hand and release the object.

        This *does not* simulate dropping the object until it collides with
        another object.

        @param env: OpenRAVE environment
        @param simulate: flag to run in simulation
        """
        # Open the hand
        super(ReleaseObjectsSolution, self).execute(env, simulate)

        manipulator = self.action.get_manipulator(env)
        robot = manipulator.GetRobot()

        # Release the objects
        for obj_key in self._objs:
            obj = self.get_obj(env, obj_key)
            robot.Release(obj)


class ReleaseObjectAction(MoveHandAction):
    """
    An Action that opens the hand and releases an object.
    """

    def __init__(self, hand, obj, dof_values, dof_indices, name=None):
        """
        @param hand: hand to release the object with
        @param obj: object to release
        @param dof_values: configuration to move the hand to before releasing
        @param dof_indices: indices of the dofs specified in dof_values
        @param name: name of the action
        """
        super(ReleaseObjectAction, self).__init__(
            hand, dof_values=dof_values, dof_indices=dof_indices, name=name)

        self._obj = to_key(obj)
        self._manipulator = to_key(hand.manipulator)

    def get_obj(self, env):
        """
        Look up and return the object in the environment.

        @param env: OpenRAVE environment
        @return KinBody to be released
        """
        return from_key(env, self._obj)

    def get_manipulator(self, env):
        """
        Look up and return the manipulator in the environment.

        @param env: OpenRAVE environment
        @return an OpenRAVE manipulator
        """
        return from_key(env, self._manipulator)

    def plan(self, env):
        """
        No planning; just create and return a ReleaseObjectsSolution.

        @param env: OpenRAVE environment
        @return a ReleaseObjectsSolution
        """
        return ReleaseObjectsSolution(
            action=self,
            objs=[self.get_obj(env)],
            dof_values=self.dof_values,
            dof_indices=self.dof_indices)


class ReleaseAllGrabbedAction(MoveHandAction):
    """
    An Action that opens the hand and releases all grabbed objects.
    """

    def __init__(self, hand, dof_values, dof_indices, name=None):
        """
        @param hand: hand to release the object with
        @param dof_values: configuration to move the hand to before releasing
        @param dof_indices: indices of the dofs specified in dof_values
        @param name: name of the action
        """
        super(ReleaseAllGrabbedAction, self).__init__(
            hand, dof_values=dof_values, dof_indices=dof_indices, name=name)

        self._manipulator = to_key(hand.manipulator)

    def get_manipulator(self, env):
        """
        Look up and return the manipulator in the environment.

        @param env: OpenRAVE environment
        @return a prpy.EndEffector
        """
        return from_key(env, self._manipulator)

    def plan(self, env):
        """
        Compute the list of bodies currently grabbed by the robot and return a
        ReleaseObjectsSolution.

        @param env: OpenRAVE environment
        @return a ReleaseObjectsSolution
        """
        manipulator = self.get_manipulator(env)
        robot = manipulator.GetRobot()
        grabbed_bodies = robot.GetGrabbed()

        return ReleaseObjectsSolution(
            action=self,
            objs=grabbed_bodies,
            dof_values=self.dof_values,
            dof_indices=self.dof_indices)
