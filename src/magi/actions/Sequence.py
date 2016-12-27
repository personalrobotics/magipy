"""
Define SequenceAction, SequenceSolution, SequenceExecutableSolution.
"""

from contextlib import nested
import logging

from magi.actions.base import Action, ExecutableSolution, ExecutionError, Solution
from magi.actions.validate import SequenceValidator

LOGGER = logging.getLogger(__name__)


class SequenceExecutableSolution(ExecutableSolution):
    """An ExecutableSolution that performs a sequence of actions in order."""

    def __init__(self, solution, executable_solutions):
        """
        @param solution: Solution that generated this ExecutableSolution
        @param executable_solutions: list of ExecutableSolutions to be executed
          in order
        """
        super(SequenceExecutableSolution, self).__init__(solution)
        self.executable_solutions = executable_solutions

    def execute(self, env, simulate):
        """
        Execute all ExecutableSolutions in order.

        @param env: OpenRAVE environment
        @param simulate: flag to run in simulation
        @return list of results of executing all solutions
        """
        return [
            executable_solution.execute(env, simulate)
            for executable_solution in self.executable_solutions
        ]


class SequenceSolution(Solution):
    """A Solution that performs a sequence of actions in order."""

    def __init__(self,
                 action,
                 solutions,
                 precondition=None,
                 postcondition=None):
        """
        @param action: Action that generated this Solution
        @param solutions: list of solutions to be processed in order
        @param precondition: Validator that validates preconditions
        @param postcondition: Validator that validates postconditions
        """
        deterministic = all(s.deterministic for s in solutions)
        super(SequenceSolution, self).__init__(action, deterministic,
                                               precondition, postcondition)
        self.solutions = solutions

    def save_and_jump(self, env):
        """
        Return a sequence of context managers.

        @param env: OpenRAVE environment
        @return a sequence of context managers, one for each solution, that must
          be applied in order
        """
        return nested(
            *[solution.save_and_jump(env) for solution in self.solutions])

    def save(self, env):
        """
        Not implemented. Use save_and_jump.
        """
        raise NotImplementedError('save is unsupported for SequenceSolution')

    def jump(self, env):
        """
        Jump through the solutions, one at a time, in order.

        @param env: OpenRAVE environment
        """
        for solution in self.solutions:
            solution.jump(env)

    def postprocess(self, env):
        """
        Postprocess each solution, one at a time, in order. After a solution
        is postprocessed, the environment is saved and then jumped to the end
        of the solution before postprocessing the next solution.

        @param env: OpenRAVE environment
        @return a SequenceExecutableSolution of executable solutions
        """
        executable_solutions = self._postprocess_recursive(env, self.solutions)
        return SequenceExecutableSolution(self, executable_solutions)

    def _postprocess_recursive(self, env, solutions):
        """
        Helper function to postprocess each solution in a list of solutions.

        @param env: OpenRAVE environment
        @param solutions: list of solutions to postprocess
        @return list of executable solutions
        """
        if not solutions:
            return []

        solution = solutions[0]
        executable_solution = solution.postprocess(env)
        with solution.save_and_jump(env):
            next_solutions = self._postprocess_recursive(env, solutions[1:])

        return [executable_solution] + next_solutions


class SequenceAction(Action):
    """A meta-Action that plans and executes a sequence of actions in order."""

    def __init__(self,
                 actions,
                 name=None,
                 precondition=None,
                 postcondition=None):
        """
        @param actions: list of Actions
        @param name: name of the action
        @param precondition: Validator that validates preconditions
        @param postcondition: Validator that validates postconditions
        """
        super(SequenceAction, self).__init__(name=name)

        self.actions = actions

        if self.actions:
            if precondition:
                self.precondition = SequenceValidator(
                    [precondition, self.actions[0].precondition])
            else:
                self.precondition = self.actions[0].precondition
            if postcondition:
                self.postcondition = SequenceValidator(
                    [self.actions[-1].postcondition, postcondition])
            else:
                self.postcondition = self.actions[-1].postcondition

    def plan(self, env):
        """
        Recursively plan each action in the set of actions used to construct
        this action. After an action is planned, the environment is saved and
        jumped to the end of the solution for the action before planning of the
        next action begins.

        @param env: OpenRAVE environment
        @return SequenceSolution of solutions to the actions
        """
        solutions = self._plan_recursive(env, self.actions)
        return SequenceSolution(self, solutions)

    def _plan_recursive(self, env, actions):
        """
        Helper function to plan each action in a list of actions.

        @param env: OpenRAVE environment
        @param actions: list of actions to plan
        @return list of solutions
        """
        if not actions:
            return []

        LOGGER.info('Planning action %s', actions[0].get_name())
        solution = actions[0].plan(env)
        with solution.save_and_jump(env):
            next_solutions = self._plan_recursive(env, actions[1:])
        return [solution] + next_solutions

    def execute(self, env, simulate):
        """
        Plan, postprocess, and execute each action, one at a time, in order.

        @param env: OpenRAVE environment
        @param simulate: flag to run in simulation
        @return list of results of executing all solutions
        """
        results = []

        for action in self.actions:
            with env:
                solution = action.plan(env)
                executable_solution = solution.postprocess(env)

            results.append(executable_solution.execute(env, simulate))
        return results
