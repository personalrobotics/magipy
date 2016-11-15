from hgpc.actions.base import ValidationError, ActionError
from hgpc.actions.Sequence import SequenceSolution, SequenceAction
from hgpc.actions.Validator import SequenceValidator
from hgpc.recovery.recovery_helper import *

import logging
logger = logging.getLogger(__name__)

class PatchedSolution(object):
    def __init__(self, solution, success):
        """
        @param success : True if solution was patched to existing solution.
        @param solution : solution
        """
        self.solution = solution
        self.success = success 

class PatchRecoveryPlanner(object):

    def __init__(self):
        self.name = "PatchRecoveryPlanner"

    def replan(self, env, failed_solution, failed_sequence, simulate, detector=None,
               num_attempts=5):
        """
        Returns a recovery solution by patching to the earliest remaining solution.
        The solution completes the remainder of the sequence.
        """

        # If there is no pending solution to patch to, do naive-replan.
        if not failed_sequence:
            from hgpc.recovery.NaiveRecoveryPlanner import NaiveRecoveryPlanner
            nrp = NaiveRecoveryPlanner()
            return nrp.replan(env, failed_solution,
                    failed_sequence, detector, num_attempts).plan(env)

        # TODO: this should work, but often fails in real robot experiment
        # Check if failed_soln in fact succeeded, return rest if it did.
        # try:
        #     return get_noreplan_solution(env, failed_soln,
        #                                  failed_seq_soln, detector)
        # except (ValidationError, ReplanError) as ignored:
        #     logger.info("Current solution failed. Proceeding to patch-recovery.")
        #     pass

        patch_solution = self.get_patch_recovery_solution(env,
                                                         failed_solution,
                                                         failed_sequence,
                                                         simulate,
                                                         detector,
                                                         num_attempts)
        if not patch_solution.success:
            logger.info('Failed to patch; replanned everything')
        else:
            logger.info('Patched successfully.')

        return patch_solution.solution

    def recursive_patch_plan(self, env, solutions, detector, num_attempts):
        """
        Helper method for patch plan. 
        Replan elements of solution.action until patchable solution is found.

        @param env Environment to plan in.
        @param solution Solution to patch to. 
        @return Patched solution
        """

        # Base 1: empty solution
        if not solutions:
            return PatchedSolution(solution=None, success=False)

        # Extract the first and rest solutions
        flattened_first_solution = flatten(solutions[0], env)
        
        first_solution = flattened_first_solution[0]
        rest_solutions = flattened_first_solution[1:] + solutions[1:]

        # Try patching to the first solution
        try: 
            patch_solution = self.patch_to(env, first_solution, detector, num_attempts)
            logger.info('Patch to {} successful.'.format(first_solution.action.get_name()))

            if patch_solution is None: 
                return PatchedSolution(
                    solution=make_sequence_solution([first_solution] + rest_solutions),
                    success=True)
            return PatchedSolution(
                solution=make_sequence_solution([patch_solution, first_solution] + rest_solutions),
                success=True)
        except ReplanError:
            logger.info('Patch to {} failed.'.format(first_solution.action.get_name()))
        
        # Plan first action, recursively patch-plan the rest.
        replanned_solution = None
        for i in xrange(num_attempts):
            try:
                with env: 
                    replanned_solution = first_solution.action.plan(env)
                    logger.info('Replanning {} successful.'
                        .format(first_solution.action.get_name()))
                break
            except ActionError: 
                logger.info('Replan attempt {} of {} for {} failed'.format(
                    i+1, num_attempts, first_solution.action.get_name()))

        if replanned_solution is None:
            raise ReplanError('All atttempts to replan {} failed.'.format(
                first_solution.action.get_name()))

        # Return if no remaining solution
        if not rest_solutions:
            return PatchedSolution(solution=replanned_solution, success=False)
        
        # Jump the first action, recursively patch-plan the rest 
        with replanned_solution.save_and_jump(env):
            logger.info('Recursive patch to {}'.format(rest_solutions[0].action.get_name()))
            rest_patched_solution = self.recursive_patch_plan(
                                        env, rest_solutions, detector, num_attempts)

            complete_solution = make_sequence_solution(
                    solutions=[replanned_solution, rest_patched_solution.solution])
            return PatchedSolution(solution=complete_solution,
                                   success=rest_patched_solution.success)


    def patch_to(self, env, solution, detector, num_attempts):
        """ Plans to the precondition of solution.
        Curently, this patches only to a solution with PatchToValidator.
        @return Patched solution.
        """
        from hgpc.actions.validators.PoseValidators import RobotPoseValidator

        precondition = solution.precondition
        if not precondition:
            raise ReplanError('No precondition explicitly stated. Cannot replan.')

        preconditions = getattr(precondition, 'validators', [precondition])
        
        # Check if solution has a RobotPoseValidator in precondition.
        pose_validator = filter(
            lambda x: isinstance(x, RobotPoseValidator), preconditions)

        if not pose_validator:
            raise ReplanError('Cannot patch to %s' %solution.action.get_name())
        
        # Snap to indicated robot pose
        robot = pose_validator[0].robot
        from hgpc.actions.Plan import PlanAction
        snap_action = PlanAction(name='SnapTo{}'.format(solution.action.get_name()),
                                 robot=robot,
                                 active_indices=range(robot.GetDOF()),
                                 # Assumes that robot uses active manipulator
                                 active_manipulator=robot.GetActiveManipulator(),
                                 method='PlanToConfiguration',
                                 args=[pose_validator[0].configuration])

        if isinstance(solution.action, DisableAction):
            snap_action = wrap_action(snap_action, solution.action, env)
        for i in xrange(num_attempts):
            try:
                with env:
                    snap_solution = snap_action.plan(env)
                return snap_solution
            except ActionError as e:
                logger.info('Attempt {} of {} for {} failed.'.format(
                    i, num_attempts, snap_action.get_name()))

        raise ReplanError("Planning {} failed".format(snap_action.get_name()))


    def get_patch_recovery_solution(self, env, failed_solution, failed_sequence,
            simulate, detector=None, num_attempts=5):

        # Locate the failed solution.
        prev, solution, _  = locate_solution(
                                failed_sequence, failed_solution, env)

        # Find action to replan from.
        logger.info("Located failed solution %s." %solution.action.get_name())
        prev_actions = SequenceAction(actions=map(lambda x: x.action,
                                                  prev + [solution]),
                                      name="ActionUpto{}".format(
                                          solution.action.get_name()))
        
        passing_action, rest_actions = find_action_with_passing_precondition(
                                        prev_actions, env, detector)
        _, passing_sol, rest = locate_action(
                                    failed_sequence, passing_action, env) 

        if rest:
            existing_solutions = [passing_sol] + rest
        else:
            existing_solutions = [passing_sol]
        
        # Replan until patch can be made
        return self.recursive_patch_plan(env, existing_solutions, detector, num_attempts)

