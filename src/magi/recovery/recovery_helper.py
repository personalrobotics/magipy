from prpy.planning.exceptions import PlanningError
from hgpc.actions.base import ValidationError, ActionError
from hgpc.actions.Sequence import SequenceSolution, SequenceAction
from hgpc.actions.Validator import SequenceValidator
from hgpc.actions.Disable import DisableAction, DisableSolution

import logging
logger = logging.getLogger(__name__)

class ReplanError(Exception):
    pass

def wrap_action(action, wrapped_action, env):
    """
    Wrap action to match wrapped_action's wrapping properties.
    Currently only supports DisableAction
    @param action action to wrap with Disable
    @param wrapped_action DisableAction to copy options from
    @param env Openrave environment
    """
    if action is wrapped_action: 
        return action 

    if not isinstance(wrapped_action, DisableAction):
        logger.warn("Supports only DisableAction")
        return action

    return DisableAction(
              objects=wrapped_action.get_objects(env),
              wrapped_action=action,
              padding_only=wrapped_action.padding_only)

def wrap_solution(solution, wrapped_solution, env):
    """ Wrap solution to match wrapped_solution's 
    wrapping properties. Currently only support DisableSolution
    @param solution solution to wrap with Disable
    @param wrapped_solution DisableSolution to copy options from 
    @param env Openrave Environment 
    """
    if solution is wrapped_solution:
        return solution

    if not isinstance(wrapped_solution, DisableSolution):
        logger.warn("Supports only DisableSolution")
        return solution

    action = wrap_action(solution.action, wrapped_solution.action, env)
    return DisableSolution(action=action, wrapped_solution=solution)

def locate_solution(solution, sol, env):
    """
    Finds sol in solution. 
    @param solution solution which contains sol
    @param sol solution to find in solution
    @param env Openrave environment
    @return (prev, sol, rest)
      prev - Solutions before sol
      sol  - sol 
      rest - Solutions after sol
    @throws ValueError if not found
    """
    return locate_action(solution, sol.action, env)

def locate_action(solution, action, env):
    """
    Finds solution containing action in seq_soln. 
    @param solution solution which may contain action
    @param action action to find in solution 
    @param env Openrave Environment
    @return (prev, sol, rest)
       prev - Solutions before solution containing action 
       sol  - Solution containing action
       rest - Solutions after action
    @throws ValueError if not found
    """
    
    # Base case 1:
    if solution.action is action:        
        return [], solution, []

    # Unwrap if necessary
    unwrapped_solution = getattr(solution, 'wrapped_solution', solution)

    inner_solutions = getattr(unwrapped_solution, 'solutions',
                              [unwrapped_solution])

    # Base case 2 (primitive solution)
    if len(inner_solutions) == 1 and inner_solutions[0] is solution:
        if solution.action is action:        
            return [], solution, []
        else:
            raise ValueError('{} not found.'.format(action.get_name()))

    for i, sol in zip(range(len(inner_solutions)), inner_solutions): 
        try: 
            prev, _sol, rest = locate_action(sol, action, env)
            prev = inner_solutions[:i] + prev
            rest = rest + inner_solutions[i+1:]

            # Re-wrap Solution
            if hasattr(solution, 'wrapped_solution'):
                if prev: 
                    prev = map(lambda x: wrap_solution(x, solution, env), prev)
                if rest: 
                    rest = map(lambda x: wrap_solution(x, solution, env), rest)
                _sol = wrap_solution(_sol, solution, env)

            return (prev, _sol, rest)

        except ValueError as ignored:
            continue

    raise ValueError("%s not found." % action.get_name())


def find_action_with_passing_precondition(action, env, detector=None):
    """ 
    In action, find last action whose precondition is satisfied.
    @param action Action to investigate
    @param env Openrave enviroment
    @param detector Detector for validation
    @return passing_action Action whose precondition is satisfied 
    @return rest Rest of actions 

    """ 
    
    # Unwrap if necessary
    unwrapped_action = getattr(action, 'wrapped_action', action)
    inner_actions = getattr(unwrapped_action,
                            'actions', [unwrapped_action])

    # Base case (when action is primitive):
    if len(inner_actions) == 1 and inner_actions[0] is action:
        if inner_actions[0].precondition is None: 
            logger.warn('{}\'s precondition not specified. Assume existing'
                ' but not specified. Put PassingValidator to indicate passing'
                ' condition.'.format(inner_actions[0].get_name()))
            raise ReplanError("Precondition not satisfied.")
        try:
            inner_actions[0].precondition.validate(env, detector)
            logger.info('Passes {}\'s precondition {}.'.format(
                inner_actions[0].get_name(), inner_actions[0].precondition))
            return inner_actions[0], []
        except ValidationError as e:
            raise ReplanError("Precondition not satisfied.")

    # Try in reverse order
    for i, iaction in reversed(zip(range(len(inner_actions)), inner_actions)):
        # Recursively call inner actions
        try:
            passing_action, rest = find_action_with_passing_precondition(
                iaction, env, detector)
            return passing_action, rest + inner_actions[i+1:]
        except ReplanError as ignored:
            continue

    # Lastly, check for action's precondition 
    if action.precondition is not None:
        try:
            action.precondition.validate(env, detector)
            logger.info('Passes action {}\'s precondition {}.'.format(
                        action.get_name(), action.precondition))
            return action, []
        except ValidationError as e:
            # Handled later
            pass 

    raise ReplanError("No passing action in %s."%action.get_name())


def make_sequence_solution(solutions):
    name = solutions[0].action.get_name()
    action = SequenceAction(actions=map(lambda x: x.action, solutions),
                            name='SeqStartsWith{}'.format(name))
    return SequenceSolution(action=action,
                            solutions = solutions)

def flatten(solution, env):
    unwrapped_solution = getattr(solution, 'wrapped_solution', solution)
    inner_solutions = getattr(unwrapped_solution, 'solutions', [unwrapped_solution])
    if inner_solutions[0] is solution:
        return [solution]
    
    all_flattened = []
    for sol in inner_solutions: 
        flattened = flatten(sol, env) 
        if isinstance(solution, DisableSolution):
            flattened = map(lambda x: wrap_solution(x, solution, env), flattened)

        all_flattened.extend(flattened)

    return all_flattened

# These are for debugging
def print_action(actions):
    for action in actions:
        print "--------",action.get_name(),action.__class__," begins --------"
        if action.precondition: 
            print "    pre: ", action.precondition

        if hasattr(action, 'actions'):
            print_action(action.actions)

        if action.postcondition: 
            print "    post: ", action.postcondition
        print "--------",action.get_name()," ends --------"

def print_solution(solutions):
    for solution in solutions: 
        print "--------",solution.action.get_name(),solution.action.__class__," begins --------"
        if  solution.precondition:
            print "    pre: ", solution.precondition

        if hasattr(solution, 'solutions'):
            print_solution(solution.solutions)
        if solution.postcondition: 
            print "    post: ", solution.postcondition
        print "--------",solution.action.get_name()," ends --------"


# TODO: Not used at the moment but maybe later
# def get_noreplan_solution(env, failed_solution, failed_seq_solution, detector):
#     """ Check whether current failed executable succeeded or can be re-executed.
#     Helper method for patch and naive replanners.
#     @returns solution Returns rest of solutions to be executed.
#     """
#     # Check if solution.postcondition passes.
#     try:
#         if failed_solution.postcondition:
#             failed_solution.postcondition.validate(env, detector)
#             logger.info("Passes solution.postcondition, " +
#                         "returning the rest of the sequence.")
#         else:
#             raise ValidationError("No postcondition, " +
#                                   "assuming it's not implemented.\n")
#     except ValidationError as e:
#         raise ReplanError(str(e) + "Need to replan.")

#     # Find where it failed, return the rest.
#     if failed_seq_solution:
#         # splits seq_solution at where failed_solution.solution is.
#         try:
#             prev, rest = split_seq_soln_by_soln(failed_seq_solution,
#                                                 failed_solution)
#             if rest:
#                 return rest
#         except ValueError as e:
#             raise ReplanError(str(e))

