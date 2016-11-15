from prpy.planning.exceptions import PlanningError
from hgpc.actions.base import ValidationError, ActionError
from hgpc.actions.Sequence import SequenceAction
from hgpc.recovery.recovery_helper import *
import logging
logger = logging.getLogger(__name__)

class NaiveRecoveryPlanner(object):

    def __init__(self):
        self.name = "NaiveRecoveryPlanner"

    def get_naive_recovery_action(self, env, failed_solution,
        failed_sequence, detector=None):
        """ Helper method for recovery planner.
        Finds the first previous action it can replan from
        and returns a SequenceAction from the found action to the end.

        @param env Openrave environment 
        @param failed_solution Fialed solution 
        @param failed_sequence Sequence to recover.
        @param detector Detector to update environment when valiating 
        @throws ReplanError if no previous action's precondition is met.
        @return SequenceAction
        """

        # Located failed solution
        prev, solution, rest = locate_solution(failed_sequence,
                                               failed_solution,
                                               env)

        # Actions upto failed solution
        if prev: 
            action = SequenceAction(actions=map(lambda x: x.action,
                                            prev + [solution]),
                        name="Actions Upto " + solution.action.get_name())
        elif rest:
            action = SequenceAction(actions=map(lambda x: x.action,
                [solution, rest]))
            return action.plan(env)
        else:
            action = solution.action
            return action.plan(env)

        try:
            # Find action to replan from.
            passing, rest_actions = find_action_with_passing_precondition(
                                        action, env, detector)
            recovery_actions = [passing] + rest_actions + map(lambda x: x.action, rest)
            return SequenceAction(recovery_actions,
                    name="RecoveryActionFrom%s"%recovery_actions[0].get_name())

        except ReplanError as e:
            raise ReplanError("All actions\' precondition failed. Cannot replan.")


    def replan(self, env, failed_solution, failed_sequence,
               simulate, detector=None, num_attempts=5):
        """ 
        Returns recovery solution for failed solution and sequence. 
        This finds the latest action it can replan and replan the rest.
        The returned sequence solution contains all solutions
        for the remainder of the sequence.

        @param env Environment where failure occured
        @param failed_executable Failed executable solution
        @param failed_sequence Sequence to recover.
        @param detector Detector being used for detecting failures
        @param simulate True if in sim mode
        @param num_attempts Number of attempts for planning recovery
        """

        # TODO: We should first check if failed solution in fact succeeded,
        # but this is hard to detect with current set of detectors;
        # it often thinks that it succeeded even when it failed,
        # so temporarily disabling this. 
        # try:
        #     solution = get_noreplan_solution(env, failed_soln,
        #                                      failed_seq_soln,
        #                                      detector)
        #     return solution
        # except ReplanError:
        #     logger.info('Current solution failed. Find action to replan from.')
        

        failed_solution = getattr(failed_solution, 'solution', failed_solution)

        # Find recovery action
        try:
            recovery_action = self.get_naive_recovery_action(env,
                failed_solution, failed_sequence, detector)

        except ReplanError as e:
            raise 

        logger.info('Recovery action found. Planning...')
        with env:
            for i in xrange(num_attempts):
                try:
                    solution = recovery_action.plan(env)
                    return solution
                except ActionError as e:
                    logger.info('Plan attempt {} of {}'
                                ' failed.'.format(i+1, num_attempts))
                    
        raise ReplanError('All {} attempts failed.'.format(num_attempts))

