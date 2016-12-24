"""Unit-testing utilities for MAGI."""

import logging
import unittest

import numpy as np

import openravepy

from magi.actions.base import to_key, Validate


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def save_env(env):
    """
    Crude implementation of serializing an environment.
    """
    saved_env = {}
    for obj in env.GetBodies():
        obj_key = to_key(obj)
        saved_env[obj_key] = {'pose': obj.GetTransform()}
        if isinstance(obj, openravepy.Robot):
            saved_env[obj_key]['dof_values'] = obj.GetDOFValues()

    return saved_env


def equal_env(env1, env2):
    """
    Crude implementation of testing if two serialized environments
    are equivalent.
    """
    for obj_key in env1.keys():
        if obj_key not in env2.keys():
            LOGGER.error('%s in env1 but not in env2', obj_key)
            return False
        if not np.allclose(env1[obj_key]['pose'], env2[obj_key]['pose']):
            LOGGER.error('%s: pose do not match', str(obj_key))
            LOGGER.info('env1 pose: %s', str(env1[obj_key]['pose']))
            LOGGER.info('env2 pose: %s', str(env2[obj_key]['pose']))
            return False
        if 'dof_values' in env1[obj_key] and 'dof_values' not in env2[obj_key] \
           or 'dof_values' not in env1[obj_key] and 'dof_values' in env2[obj_key]:
            LOGGER.error('%s: only one environment contains dof_values for the object', obj_key)
            return False
        if 'dof_values' in env1[obj_key]:
            if not np.allclose(env1[obj_key]['dof_values'], env2[obj_key]['dof_values']):
                LOGGER.error('%s: dof_values do not match', str(obj_key))
                LOGGER.info('env1 dof_values: %s', str(env1[obj_key]['dof_values']))
                LOGGER.info('env2 dof_values: %s', str(env2[obj_key]['dof_values']))
                return False
    return True


class MAGITest(unittest.TestCase):
    """Base class for MAGI test cases."""

    is_setup = False

    def setUp(self):
        if not self.is_setup:
            openravepy.RaveInitialize(True)
            openravepy.misc.InitOpenRAVELogging()
            openravepy.RaveSetDebugLevel(openravepy.DebugLevel.Fatal)
            self.env = openravepy.Environment()
            self.robot = self.env.ReadRobotXMLFile('robots/barrettwam.robot.xml')
            self.env.Add(self.robot)

    def _action_helper(self, action, validate=False, detector=None):
        """
        Test planning, postprocessing and execution of an action.
        Test save and jump to ensure they are properly restoring the environment.
        """
        self.assertTrue(action is not None)

        # Save the state of the envirnoment before planning
        start_env = save_env(self.env)

        # Run the planner
        with self.env:
            solution = action.plan(self.env)

        # Save and jump
        with solution.save_and_jump(self.env):
            # Get the state of the environment after the jump
            jump_env = save_env(self.env)
        # Get the state of the environment after restoring from save
        saved_env = save_env(self.env)

        # restoring from save should leave the environment back in the start state
        self.assertTrue(equal_env(start_env, saved_env))

        # post process and execute
        executable_solution = solution.postprocess(self.env)
        if validate:
            with Validate(self.env, executable_solution.precondition,
                          executable_solution.postcondition, detector):
                executable_solution.execute(self.env, simulate=True)
        else:
            executable_solution.execute(self.env, simulate=True)


        # After execution the environment should be in the same state as the
        #  end of the jump
        final_env = save_env(self.env)

        self.assertTrue(equal_env(jump_env, final_env))

        # Return this solution in case the caller wants to chain the result
        return solution

    def _recursive_seq_test(self, actions, validate=False, detector=None):
        """
        Test a list of actions that are part of a sequence
        """
        if len(actions) == 0:
            return

        action = actions[0]
        LOGGER.info('Testing action %s', action.get_name())
        solution = self._action_helper(action, validate, detector)
        with solution.save_and_jump(self.env):
            self._recursive_seq_test(actions[1:], validate, detector)
