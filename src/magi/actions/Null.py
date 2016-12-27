"""Define NullAction and NullSolution."""

from contextlib import contextmanager

from magi.actions.base import Action, Solution, ExecutableSolution


@contextmanager
def empty_manager():
    """Empty context manager."""
    yield


class NullSolution(Solution, ExecutableSolution):
    """A Solution and ExecutableSolution that does nothing."""

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
    """An Action that does nothing."""

    def plan(self, env):
        """Simply returns a NullSolution."""
        return NullSolution(self)
