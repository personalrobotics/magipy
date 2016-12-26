"""Base classes, context managers, and exceptions for MAGI actions."""

from abc import ABCMeta, abstractmethod
import logging

from openravepy import KinBody, Robot

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


class SaveAndJump(object):
    """
    Save the state of the environment and jump the environment to the result of
    a solution when entering. Jump back to the original state when exiting.
    """

    def __init__(self, solution, env):
        """
        @param solution: a Solution object
        @param env: the OpenRAVE environment to call save and jump on
        """
        self.solution = solution
        self.env = env

    def __enter__(self):
        """
        First call save on the solution, then jump.
        """
        LOGGER.debug('Begin SaveAndJump: %s', self.solution.action.get_name())
        self.cm = self.solution.save(self.env)
        self.cm.__enter__()
        self.solution.jump(self.env)

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager created when this context manager was entered."""
        LOGGER.debug('End SaveAndJump: %s', (self.solution.action.get_name()))
        retval = self.cm.__exit__(exc_type, exc_value, traceback)
        return retval


class Validate(object):
    """
    Check a precondition when entering and a postcondition when exiting.
    """

    def __init__(self,
                 env,
                 precondition=None,
                 postcondition=None,
                 detector=None):
        """
        @param env: OpenRAVE environment
        @param precondition: Validator that validates preconditions
        @param postcondition: Validator that validates postconditions
        @param detector: object detector (implements DetectObjects, Update)
        """
        self.env = env
        self.precondition = precondition
        self.postcondition = postcondition
        self.detector = detector

    def __enter__(self):
        """Validate precondition."""
        LOGGER.info('Validate precondition: %s', self.precondition)
        if self.precondition is not None:
            self.precondition.validate(self.env, self.detector)

    def __exit__(self, exc_type, exc_value, traceback):
        """Validate postcondition."""
        LOGGER.info('Validate postcondition: %s', self.postcondition)
        if self.postcondition is not None:
            self.postcondition.validate(self.env, self.detector)


class ActionError(Exception):
    """Base exception class for actions."""

    KNOWN_KWARGS = {'deterministic'}

    def __init__(self, *args, **kwargs):
        super(ActionError, self).__init__(*args)

        assert self.KNOWN_KWARGS.issuperset(kwargs.keys())
        self.deterministic = kwargs.get('deterministic', None)


class CheckpointError(ActionError):
    """Exception class for checkpoints."""

    pass


class ExecutionError(Exception):
    """Exception class for executing solutions."""

    def __init__(self, message='', solution=None):
        super(ExecutionError, self).__init__(message)
        self.failed_solution = solution


class ValidationError(Exception):
    """Exception class for validating solutions."""

    def __init__(self, message='', validator=None):
        super(ValidationError, self).__init__(message)
        self.failed_validator = validator


class Action(object):
    """Abstract base class for actions."""

    __metaclass__ = ABCMeta

    def __init__(self,
                 name=None,
                 precondition=None,
                 postcondition=None,
                 checkpoint=False):
        """
        @param name: name of the action
        @param precondition: Validator that validates preconditions
        @param postcondition: Validator that validates postconditions
        @param checkpoint: True if this action is a checkpoint - once a Solution
          is achieved, neither the plan method of this action nor any of its
          predecessors will be called again
        """
        self._name = name
        self.precondition = precondition
        self.postcondition = postcondition
        self.checkpoint = checkpoint

    def get_name(self):
        """
        Return the name of the action.
        """
        return self._name

    @abstractmethod
    def plan(self, env):
        """
        Return a Solution that realizes this action.

        This method attempts to realize this action in the input environment, if
        possible. It MUST restore the environment to its original state before
        returning. If successful, this method returns a Solution object.
        Otherwise, it raises an ActionError.

        The implementation of this method MAY be stochastic. If so, the method
        may return a different solution each time it is called.

        The environment MUST be locked when calling this method.

        Ideally, planners should "with Validate(env, self.precondition)" when
        calling this.

        @param env: OpenRAVE environment
        @return Solution object
        """
        pass

    def execute(self, env, simulate):
        """
        Plan, postprocess, and execute this action.

        This is a helper method that wraps the plan() method.

        The environment MUST NOT be locked while calling this method.

        @param env: OpenRAVE environment
        @param simulate: flag to run in simulation
        @return result of executing the action
        """
        with env:
            solution = self.plan(env)
            executable_solution = solution.postprocess(env)

        return executable_solution.execute(env, simulate)


class Solution(object):
    """Abstract base class for solutions."""

    __metaclass__ = ABCMeta

    def __init__(self,
                 action,
                 deterministic,
                 precondition=None,
                 postcondition=None):
        """
        @param action: the action that this Solution realizes
        @param deterministic: True if calling the plan method on the action
          multiple times will give the exact same solution
        @param precondition: Validator. Can be more specific than action's
          precondition.
        @param postcondition: Validator. Can be more specific than action's
          postcondition.
        """
        self.action = action
        self.deterministic = deterministic
        self.precondition = precondition if precondition else action.precondition
        self.postcondition = postcondition if postcondition else action.postcondition

    def save_and_jump(self, env):
        """
        Return a context manager that preserves the state of the environmnet
        then jumps the environment to the result of this solution.

        This context manager MUST restore the environment to its original state
        before returning.

        @param env: OpenRAVE environment
        @return context manager
        """
        return SaveAndJump(self, env)

    @abstractmethod
    def save(self, env):
        """
        Return a context manager that preserves the state of the environment.

        This method returns a context manager that preserves the state of the
        robot that is changed by the jump() method or by executing the
        solution. This context manager MUST restore the environment to its
        original state before returning.

        @param env: OpenRAVE environment
        @return context manager
        """
        pass

    @abstractmethod
    def jump(self, env):
        """
        Set the state of the environment to the result of this solution.

        The input environment to this method MUST be in the same state that was
        used to plan this action. The environment MUST be modified to match the
        result of executing action. This method SHOULD perform the minimal
        computation necessary to achieve this result.

        The environment MUST be locked while calling this method.

        @param env: OpenRAVE environment
        """
        pass

    @abstractmethod
    def postprocess(self, env):
        """
        Return an ExecutableSolution that can be executed.

        Post-process this solution to prepare for execution. The input
        environment to this method MUST be in the same state that was used to
        plan the environment. The environment MUST be restored to this state
        before returning.

        This operation MUST NOT be capable of failing and MUST NOT change the
        state of the environment after executing the action. As long as these
        two properties are satisfied, the result MAY be stochastic.

        The environment MUST be locked while calling this method.

        @param env: OpenRAVE environment
        @return ExecutableSolution object
        """
        pass

    def execute(self, env, simulate):
        """
        Postprocess and execute this solution.

        This is a helper method that wraps the postprocess() method.

        The environment MUST NOT be locked while calling this method.

        @param env: OpenRAVE environment
        @param simulate: flag to run in simulation
        @return result of executing the solution
        """
        with env:
            executable_solution = self.postprocess(env)

        return executable_solution.execute(env, simulate)


class ExecutableSolution(object):
    """Abstract base class for executing post-processed solutions."""

    __metaclass__ = ABCMeta

    def __init__(self, solution):
        self.solution = solution
        self.precondition = solution.precondition
        self.postcondition = solution.postcondition

    @abstractmethod
    def execute(self, env, simulate):
        """
        Execute this solution.

        If execution fails, this method should raise an ExecutionError.

        The environment MUST NOT be locked while calling this method.

        @param env: OpenRAVE environment
        @param simulate: flag to run in simulation
        @return result of executing the solution
        """
        pass


def to_key(obj):
    """
    Return a tuple that uniquely identifies an object in an Environment.

    The output of this function can be passed to from_key to find the
    equivalent object in, potentially, a different OpenRAVE environment.

    @param obj: object in an OpenRAVE environment
    @return tuple that uniquely identifies the object
    """
    if obj is None:
        return None
    elif isinstance(obj, (KinBody, Robot)):
        key = obj.GetName(),
    elif isinstance(obj, (KinBody.Joint, KinBody.Link)):
        key = obj.GetParent().GetName(), obj.GetName()
    elif isinstance(obj, Robot.Manipulator):
        key = obj.GetRobot().GetName(), obj.GetName()
    else:
        raise TypeError('Unknown type "{!s}".'.format(type(obj)))

    return (type(obj), ) + key


def from_key(env, key):
    """
    Return the object identified by the input key in an Environment.

    The input of this function is constructed by the to_key function.

    @param env: an OpenRAVE environment
    @param key: tuple that uniquely identifies the object
    @return object in the input OpenRAVE environment
    """
    if key is None:
        return None

    obj_type = key[0]

    if issubclass(obj_type, (KinBody, Robot)):
        return env.GetKinBody(key[1])
    elif issubclass(obj_type, KinBody.Joint):
        return env.GetKinBody(key[1]).GetJoint(key[2])
    elif issubclass(obj_type, KinBody.Link):
        return env.GetKinBody(key[1]).GetLink(key[2])
    elif issubclass(obj_type, Robot.Manipulator):
        return env.GetRobot(key[1]).GetManipulator(key[2])
    else:
        raise TypeError('Unknown type "{!s}".'.format(obj_type))
