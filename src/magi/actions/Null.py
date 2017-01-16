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
        """
        @param action: Action that generated this Solution
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

    def save(self, env):
        """
        @param env: OpenRAVE environment
        @return empty context manager
        """
        return empty_manager()

    def jump(self, env):
        """
        @param env: OpenRAVE environment
        """
        pass

    def postprocess(self, env):
        """
        @param env: OpenRAVE environment
        @return this object, unchanged
        """
        return self

    def execute(self, env, simulate):
        """
        @param env: OpenRAVE environment
        @param simulate: flag to run in simulation
        """
        pass


class NullAction(Action):
    """An Action that does nothing."""

    def plan(self, env):
        """
        No planning; simply returns a NullSolution.

        @param env: OpenRAVE environment
        @return a NullSolution
        """
        return NullSolution(self)
