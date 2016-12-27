"""
Define ParallelAction and OptionalAction.
"""

import logging
import random

from magi.actions.Null import NullAction
from magi.actions.base import Action, ActionError

LOGGER = logging.getLogger(__name__)


class ParallelAction(Action):
    """
    A meta-Action that takes a list of actions, any one of which could succeed
    to consider this Action a success.
    """

    def __init__(self, actions, name=None):
        """
        @param actions: list of Actions
        @param name: name of the action
        """
        super(ParallelAction, self).__init__(name=name)

        self.actions = actions

    def plan(self, env):
        """
        Randomly select and plan an action from the actions list.

        @param env: OpenRAVE environment
        @return Solution to a randomly selected Action
        """
        if not self.actions:
            raise ActionError(
                'The ParallelAction contains no actions to select from.')
        action = random.choice(self.actions)
        return action.plan(env)


class OptionalAction(ParallelAction):
    """
    A meta-Action that takes an action that may not be required.
    """

    def __init__(self, action, name=None):
        """
        @param action: action that may not be required
        @param name: name of the action
        """
        actions = [NullAction(), action]
        super(OptionalAction, self).__init__(actions, name=name)
