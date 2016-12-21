from .base import Action, Solution, ExecutableSolution, to_key, from_key
from prpy.rave import AllDisabled


class DisableExecutableSolution(ExecutableSolution):
    def __init__(self, solution, wrapped_solution):
        """
        @param solution The DisableSolution that created this ExecutableSolution
        @param wrapped_solution The Solution or ExecutableSolution that should be called with
        the objects specified in the action Disabled
        """
        self.precondition = wrapped_solution.precondition
        self.postcondition = wrapped_solution.postcondition

        ExecutableSolution.__init__(self, solution)
        self.solution = solution
        self.wrapped_solution = wrapped_solution

    def execute(self, env, simulate):
        """
        Disable all required objects, then call execute on the wrapped ExecutableSolution
        """
        objects = self.solution.action.get_objects(env)
        with AllDisabled(
                env, objects, padding_only=self.solution.action.padding_only):
            return self.wrapped_solution.execute(env, simulate)


class DisableSolution(Solution, ExecutableSolution):
    def __init__(self, action, wrapped_solution):
        """
        @param action The DisableAction that created this Solution or ExecutableSolution
        @param wrapped_solution The Solution or ExecutableSolution that should be called with
        the objects specified in the action Disabled
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
        Class save on the wrapped Solution obejct
        @param env The OpenRAVE environment
        """
        return self.wrapped_solution.save(env)

    def jump(self, env):
        """
        Disable all required objects, then call jump on the wrapped Solution
        @param env The OpenRAVE environment
        """
        objects = self.action.get_objects(env)
        with AllDisabled(env, objects, padding_only=self.action.padding_only):
            self.wrapped_solution.jump(env)

    def postprocess(self, env):
        """
        Disable all required objects, then call postprocess on the wrapped Solution
        @param env The OpenRAVE environment
        @return A DisableSolution that wraps the ExecutableSolution generated by the
        postpress call
        """
        objects = self.action.get_objects(env)
        with AllDisabled(env, objects, padding_only=self.action.padding_only):
            executable_solution = self.wrapped_solution.postprocess(env)
        return DisableExecutableSolution(
            solution=self, wrapped_solution=executable_solution)


class DisableAction(Action):
    def __init__(self,
                 objects,
                 wrapped_action,
                 padding_only=False,
                 name=None,
                 checkpoint=False):
        """
        @param objects The list of objects to disable during planning and execution
        @param wrapped_action The action to call while objects are disabled
        @param padding_only If true, only disable padding on the objects
        @param name The name of the action
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
        @return The name of the action is one was specified during action construction,
        otherwise, the name of the wrapped Action
        """
        if self._name is not None:
            return self._name
        else:
            return self.action.get_name()

    def get_objects(self, env):
        """
        @return A list of objects to disable
        """
        return [from_key(env, o) for o in self._objects]

    def plan(self, env):
        """
        Disables the objects then calls the plan method on
        the wrapped action
        @return A DisableSolution
        """
        objects = self.get_objects(env)
        with AllDisabled(env, objects, padding_only=self.padding_only):
            solution = self.action.plan(env)

        return DisableSolution(action=self, wrapped_solution=solution)
