from .base import Action, ActionError
import logging
logger = logging.getLogger(__name__)

class ParallelAction(Action):
    def __init__(self, actions, name=None):
        """
        @param actions A list of actions, any one of which could succeed to consider this
          Action a success
        @param name The name of the action
        """
        super(ParallelAction, self).__init__(name=name)
        
        self.actions = actions

    def plan(self, env):
        """
        Randomly select and plan an action from the actions list
        and plans the action.
        """
        
        num_actions = len(self.actions)
        if num_actions == 0:
            raise ActionError("The ParallelAction contains no actions to select from")

        import random
        idx = random.randint(0, num_actions-1)

        return self.actions[idx].plan(env)

class OptionalAction(ParallelAction):
    def __init__(self, action, name=None):
        """
        @param action An action that may not be required
        @param name The name fo the action
        """
        from Null import NullAction
        actions = [ NullAction(), action ]
        super(OptionalAction, self).__init__(actions, name=name)
