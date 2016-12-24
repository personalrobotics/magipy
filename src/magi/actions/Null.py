from contextlib import contextmanager

from magi.actions.base import Action, Solution, ExecutableSolution


@contextmanager
def empty_manager():
    """
    Empty context lib
    """
    yield


class NullSolution(Solution, ExecutableSolution):
    """
    A Solution and ExecutableSolution class that do nothing - simply a place holder
    """

    def __init__(self, action, precondition=None, postcondition=None):
        Solution.__init__(
            self,
            action,
            deterministic=True,
            precondition=precondition,
            postcondition=postcondition)
        ExecutableSolution.__init__(self, self)

    def save(self, env):
        return empty_manager()

    def jump(self, env):
        pass

    def postprocess(self, env):
        return self

    def execute(self, env, simulate):
        pass


class NullAction(Action):
    def plan(self, env):
        """
        Does nothing. Simply returns a NullSolution
        @param env The OpenRAVE environment
        @return A NullSolution class.
        """
        return NullSolution(self)
