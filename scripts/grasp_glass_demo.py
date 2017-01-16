#!/usr/bin/env python

"""Demo for grasping glass."""

import argparse
import logging
import signal
import sys

import IPython
import numpy

from prpy.rave import add_object
import herbpy
import openravepy

from magi.actions.Disable import DisableAction
from magi.actions.MoveHand import MoveHandAction, GrabObjectAction
from magi.actions.Plan import PlanToTSRAction
from magi.actions.PushObject import PushObjectAction
from magi.actions.Sequence import SequenceAction
from magi.actions.base import ActionError, ExecutionError
from magi.execution import execute_pipeline
from magi.planning import DepthFirstPlanner, RestartPlanner
import magi.monitor

LOGGER = logging.getLogger(__name__)


def grasp_glass_action_graph(manipulator, glass, table):
    """
    Generate grasp glass action graph.

    @param manipulator: manipulator to grasp the glass with
    @param glass: glass to grasp
    @param table: table on which the grasp is resting
    @return Action graph
    """
    hand_dof, hand_values = manipulator.hand.configurations.get_configuration('glass_grasp')
    close_dof, close_values = manipulator.hand.configurations.get_configuration('closed')

    actions = [
        MoveHandAction(
            name='OpenHandAction',
            hand=manipulator.hand,
            dof_values=hand_values,
            dof_indices=hand_dof
        ),
        PlanToTSRAction(
            name='PlanToPoseNearGlassAction',
            robot=manipulator.GetRobot(),
            obj=glass,
            tsr_name='push_grasp',
            active_indices=manipulator.GetArmIndices(),
            active_manipulator=manipulator
        ),
        DisableAction(
            objects=[table],
            padding_only=True,
            wrapped_action=PushObjectAction(
                name='PushGlassAction',
                manipulator=manipulator,
                obj=glass,
                push_distance=0.2
            )
        ),
        DisableAction(
            objects=[table],
            padding_only=True,
            wrapped_action=GrabObjectAction(
                name='GraspGlassAction',
                hand=manipulator.hand,
                obj=glass,
                dof_values=close_values,
                dof_indices=close_dof
            )
        )
    ]

    return SequenceAction(actions, name='GraspCupAction')

# OBJ_IN_FRAME transforms

TABLE_IN_ROBOT = numpy.array([[0., 0., 1., 0.945],
                              [1., 0., 0., 0.],
                              [0., 1., 0., 0.02],
                              [0., 0., 0., 1.]])

GLASS_IN_TABLE = numpy.array([[1., 0., 0., -0.359],
                              [0., 0., 1., 0.739],
                              [0., -1., 0., -0.072],
                              [0., 0., 0., 1.]])

def detect_objects(robot):
    """
    Return table and glass objects relative to the robot.
    Add table and glass objects to the world.

    @param robot: OpenRAVE robot
    @return table and glass KinBodies
    """
    env = robot.GetEnv()
    with env:
        robot_in_world = robot.GetTransform()

    table_in_world = numpy.dot(robot_in_world, TABLE_IN_ROBOT)
    table = add_object(env, 'table', 'furniture/table.kinbody.xml',
                       table_in_world)

    glass = add_object(env, 'glass', 'objects/plastic_glass.kinbody.xml',
                       numpy.dot(table_in_world, GLASS_IN_TABLE))
    return table, glass

def main():
    """Execute demo for grasping glass."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewer', '-v', type=str, default='interactivemarker',
                        help='The viewer to attach (none for no viewer)')
    parser.add_argument('--monitor', action='store_true',
                        help='Display a UI to monitor progress of the planner')
    parser.add_argument('--planner', type=str, choices=['dfs', 'restart'], default='restart',
                        help='The planner to use')
    parser.add_argument('--robot', type=str, default='herb',
                        help='Robot to run the task on')

    openravepy.RaveInitialize(True, level=openravepy.DebugLevel.Info)
    openravepy.misc.InitOpenRAVELogging()

    args = parser.parse_args()

    env, robot = herbpy.initialize()

    # Get the desired manipulator
    manipulator = robot.GetManipulator('right')

    if args.viewer != 'none':
        env.SetViewer(args.viewer)

    monitor = None
    # Create a monitor
    if args.monitor:
        monitor = magi.monitor.ActionMonitor()

        def signal_handler(signum, frame):
            """Signal handler to gracefully kill the monitor."""
            monitor.stop()
            sys.exit(0)
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    # Create a planner
    if args.planner == 'restart':
        planner = RestartPlanner(monitor=monitor)
    elif args.planner == 'dfs':
        planner = DepthFirstPlanner(monitor=monitor, use_frustration=True)

    if monitor is not None:
        monitor.reset()

    # Detect objects
    table, glass = detect_objects(robot)

    try:
        # Create the task.
        action = grasp_glass_action_graph(manipulator, glass, table)

        # Plan the task
        with env:
            solution = planner.plan_action(env, action)

        # Execute the task
        execute_pipeline(env, solution, simulate=True, monitor=monitor)

    except ActionError as err:
        LOGGER.info('Failed to complete planning for task: %s', str(err))
        raise

    except ExecutionError as err:
        LOGGER.info('Failed to execute task: %s', str(err))
        raise

    IPython.embed()

    if monitor:
        monitor.stop()

if __name__ == "__main__":
    main()
