#!/usr/bin/env python
import numpy, logging, time
from magi.actions.base import ActionError, ExecutionError, ValidationError
from magi.execution import plan_execute_pipeline, execute_pipeline, execute_serial, execute_interleaved
from magi.planning import DepthFirstPlanner, RestartPlanner, TimeoutException
from prpy.exceptions import TrajectoryAborted
from prpy.rave import add_object
from magi.actions.Plan import PlanToTSRAction
from magi.actions.MoveHand import MoveHandAction, GrabObjectAction
from magi.actions.Sequence import SequenceAction
from magi.actions.Disable import DisableAction
from magi.actions.PushObject import PushObjectAction

LOGGER = logging.getLogger(__name__)

def grasp_glass_action_graph(manipulator_, glass_, table_):
    hand_dof, hand_values = manipulator_.hand.configurations.get_configuration('glass_grasp')
    close_dof, close_values = manipulator_.hand.configurations.get_configuration('closed')

    actions = [
        MoveHandAction(
            name='OpenHandAction',
            hand=manipulator_.hand,
            dof_values=hand_values,
            dof_indices=hand_dof
        ),
        PlanToTSRAction(
            name='PlanToPoseNearGlassAction',
            robot=manipulator_.GetRobot(),
            obj=glass_,
            tsr_name='push_grasp',
            active_indices=manipulator_.GetArmIndices(),
            active_manipulator=manipulator_
        ),
        DisableAction(
            objects=[table_],
            padding_only=True,
            wrapped_action=PushObjectAction(
                name='PushGlassAction',
                manipulator=manipulator_,
                obj=glass_,
                push_distance=0.2
            )
        ),
        DisableAction(
            objects=[table_],
            padding_only=True,
            wrapped_action=GrabObjectAction(
                name='GraspGlassAction',
                hand=manipulator_.hand,
                obj=glass_,
                dof_values=close_values,
                dof_indices=close_dof
            )
        )
    ]

    return SequenceAction(actions, name='GraspCupAction')

def detect_objects(robot_):
    env_ = robot_.GetEnv()
    with env_:
        robot_in_world = robot_.GetTransform()

    table_in_robot = numpy.array([[0., 0., 1., 0.945],
                                  [1., 0., 0., 0.],
                                  [0., 1., 0., 0.02],
                                  [0., 0., 0., 1.]])
    table_in_world = numpy.dot(robot_in_world, table_in_robot)
    table_ = add_object(env_, 'table', 'furniture/table.kinbody.xml',
                        table_in_world)

    glass_in_table = numpy.array([[1., 0., 0., -0.359],
                                  [0., 0., 1., 0.739],
                                  [0., -1., 0., -0.072],
                                  [0., 0., 0., 1.]])
    glass_ = add_object(env_, 'glass', 'objects/plastic_glass.kinbody.xml',
                        numpy.dot(table_in_world, glass_in_table))
    return table_, glass_


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewer', '-v', type=str, default='interactivemarker',
                        help='The viewer to attach (none for no viewer)')
    parser.add_argument('--monitor', action='store_true',
                        help='Display a UI to monitor progress of the planner')
    parser.add_argument('--planner', type=str, choices=['dfs', 'restart'], default='restart',
                        help='The planner to use')
    parser.add_argument('--robot', type=str, default='herb',
                        help='Robot to run the task on')

    import openravepy
    openravepy.RaveInitialize(True, level=openravepy.DebugLevel.Info)
    openravepy.misc.InitOpenRAVELogging()

    args = parser.parse_args()

    import herbpy
    env, robot = herbpy.initialize()

    # Get the desired manipulator
    manipulator = robot.GetManipulator('right')

    if args.viewer != 'none':
        env.SetViewer(args.viewer)

    monitor = None
    # Create a monitor
    if args.monitor:
        import magi.monitor
        monitor = magi.monitor.ActionMonitor()

        # Setup a signal handler to gracefully kill the monitor
        def signal_handler(signum, frame):
            monitor.stop()
            import sys
            sys.exit(0)
        import signal
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

    import IPython
    IPython.embed()

    if monitor:
        monitor.stop()
