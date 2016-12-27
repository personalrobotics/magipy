"""Define DisableAction, DisableSolution, and DisableExecutableSolution."""

from prpy.rave import AllDisabled

from magi.actions.base import Action, Solution, ExecutableSolution, to_key, from_key


class DisableExecutableSolution(ExecutableSolution):
    """An ExecutableSolution that disables objects during execution."""

    def __init__(self, solution, wrapped_solution):
        """
        @param solution: DisableSolution that created this ExecutableSolution
        @param wrapped_solution: Solution or ExecutableSolution that should be
          called with the objects specified in the action disabled
        """
        # FIXME: ExecutableSolution.__init__ overrides these two
        # attributes. These should either be moved after the __init__ call or
        # deleted. The solution attribute is also already set.
        self.precondition = wrapped_solution.precondition
        self.postcondition = wrapped_solution.postcondition

        ExecutableSolution.__init__(self, solution)
        self.solution = solution
        self.wrapped_solution = wrapped_solution

    def execute(self, env, simulate):
        """
        Disable all required objects, then call execute on the wrapped
        ExecutableSolution.

        @param env: OpenRAVE environment
        @param simulate: flag to run in simulation
        @return result of executing the wrapped solution
        """
        objects = self.solution.action.get_objects(env)
        with AllDisabled(
            env, objects, padding_only=self.solution.action.padding_only):
            return self.wrapped_solution.execute(env, simulate)


class DisableSolution(Solution):
    """A Solution that disables objects during execution."""

    def __init__(self, action, wrapped_solution):
        """
        @param action: DisableAction that created this Solution
        @param wrapped_solution: Solution or ExecutableSolution that should be
          called with the objects specified in the action disabled
        """
        Solution.__init__(
            self,
            action,
            deterministic=wrapped_solution.deterministic,
            precondition=wrapped_solution.precondition,
            postcondition=wrapped_solution.postcondition)

        self.wrapped_solution = wrapped_solution

    def save(self, env):
        """
        @param env: OpenRAVE environment
        @return context manager
        """
        return self.wrapped_solution.save(env)

    def jump(self, env):
        """
        Disable all required objects, then call jump on the wrapped Solution.

        @param env: OpenRAVE environment
        """
        objects = self.action.get_objects(env)
        with AllDisabled(env, objects, padding_only=self.action.padding_only):
            self.wrapped_solution.jump(env)

    def postprocess(self, env):
        """
        Disable all required objects, then call postprocess on the wrapped Solution.

        @param env: OpenRAVE environment
        @return a DisableExecutableSolution that wraps the ExecutableSolution
          generated by the postprocess call
        """
        objects = self.action.get_objects(env)
        with AllDisabled(env, objects, padding_only=self.action.padding_only):
            executable_solution = self.wrapped_solution.postprocess(env)
        return DisableExecutableSolution(
            solution=self, wrapped_solution=executable_solution)


class DisableAction(Action):
    """An Action that disables objects during planning and execution."""

    def __init__(self,
                 objects,
                 wrapped_action,
                 padding_only=False,
                 name=None,
                 checkpoint=False):
        """
        @param objects: list of KinBodies to disable
        @param wrapped_action: Action to call while objects are disabled
        @param padding_only: if True, only disable padding on the objects
        @param name: name of the action
        @param checkpoint: True if the action is a checkpoint
        """
        super(DisableAction, self).__init__(
            name=name,
            precondition=wrapped_action.precondition,
            postcondition=wrapped_action.postcondition,
            checkpoint=checkpoint)
        self.action = wrapped_action
        self._objects = [to_key(o) for o in objects]
        self.padding_only = padding_only

    def get_name(self):
        """
        Override Action.get_name to return the name of the wrapped action if a
        name was not specified for this action.

        @return name of this action, or the name of the wrapped action
        """
        if self._name is not None:
            return self._name
        else:
            return self.action.get_name()

    def get_objects(self, env):
        """
        @param env: OpenRAVE environment with objects to disable
        @return list of KinBodies to disable
        """
        return [from_key(env, o) for o in self._objects]

    def plan(self, env):
        """
        Disables objects, then calls the wrapped action's plan method.

        @param env: OpenRAVE environment
        @return a DisableSolution
        """
        objects = self.get_objects(env)
        with AllDisabled(env, objects, padding_only=self.padding_only):
            solution = self.action.plan(env)

        return DisableSolution(action=self, wrapped_solution=solution)
