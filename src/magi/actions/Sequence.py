from contextlib import nested
import logging

from magi.actions.base import Action, ExecutableSolution, Solution
from magi.actions.validate import SequenceValidator

LOGGER = logging.getLogger(__name__)


class SequenceExecutableSolution(ExecutableSolution):
    def __init__(self, solution, executable_solutions):
        """
        @param solution The solution that generated this ExecutableSolution
        @param executable_solutions A list of ExecutableSolutions to be executed in order
        """
        super(SequenceExecutableSolution, self).__init__(solution)
        self.executable_solutions = executable_solutions

    def execute(self, env, simulate):
        """
        Execute all ExecutableSolutions used to initialize this object in order
        @param env The OpenRAVE environment
        @param simulate If True, simulate execution
        """
        return [
            executable_solution.execute(env, simulate)
            for executable_solution in self.executable_solutions
        ]


class SequenceSolution(Solution):
    def __init__(self,
                 action,
                 solutions,
                 precondition=None,
                 postcondition=None):
        """
        @param action The Action that generated this solution
        @param solutions A list of solutions to be processed in order
        """
        deterministic = all(s.deterministic for s in solutions)
        super(SequenceSolution, self).__init__(action, deterministic,
                                               precondition, postcondition)
        self.solutions = solutions

    def save_and_jump(self, env):
        """
        @param env The OpenRAVe environment
        @return A set of context managers, one for each solution, that must be applied in order
        """
        return nested(
            *[solution.save_and_jump(env) for solution in self.solutions])

    def save(self, env):
        """
        Not implemented. Use save_and_jump
        """
        raise ExecutionError('save is unsupported for SequenceSolution')

    def jump(self, env):
        """
        Jump through the solutions, one at a time, in order
        @param env The OpenRAVe enviornment
        """
        for solution in self.solutions:
            solution.jump(env)

    def postprocess(self, env):
        """
        Postprocess each solution, one at a time, in order. After a solution
        is postprocessed, the environment is saved and then jumped to the end
        of the solution before beginning postprocessing on the next solution.

        @param env The OpenRAVE environment
        """
        executable_solutions = self._postprocess_recursive(env, self.solutions)
        return SequenceExecutableSolution(self, executable_solutions)

    def _postprocess_recursive(self, env, solutions):
        if not solutions:
            return []

        solution = solutions[0]
        executable_solution = solution.postprocess(env)
        with solution.save_and_jump(env):
            next_solutions = self._postprocess_recursive(env, solutions[1:])

        return [executable_solution] + next_solutions


class SequenceAction(Action):
    def __init__(self,
                 actions,
                 name=None,
                 precondition=None,
                 postcondition=None):
        """
        @param actions A list of actions to be planned and executed in order
        @param name The name of this action
        """
        super(SequenceAction, self).__init__(name=name)

        self.actions = actions

        if len(self.actions) > 0:
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
        Recursively plan each action in the set of actions used to construct this action.
        After an action is planned, the environment is saved and jumped to the end of the solution
        for the action before planning of the next action begins.

        @param env The OpenRAVe environment
        """
        solutions = self._plan_recursive(env, self.actions)
        return SequenceSolution(self, solutions)

    def _plan_recursive(self, env, actions):
        if not actions:
            return []

        LOGGER.info('Planning action %s', actions[0].get_name())
        solution = actions[0].plan(env)
        with solution.save_and_jump(env):
            next_solutions = self._plan_recursive(env, actions[1:])
        return [solution] + next_solutions

    def execute(self, env, simulate):
        """
        Plan, postprocess and execute each action, one at a time, in order.
        @param env The OpenRAVE environment
        @param simulate If True, simulate execution
        """
        results = []

        for action in self.actions:
            with env:
                solution = action.plan(env)
                executable_solution = solution.postprocess(env)

            results.append(executable_solution.execute(env, simulate))
        return results
