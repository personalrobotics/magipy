"""Base classes for MAGI validators."""

from abc import abstractmethod

from magi.actions.base import Action, Solution, ExecutableSolution, ExecutionError


class Validator(object):
    """
    Validates the state of an environment before or after execution of an
    action. An instance of this class should be created and passed to
    base.Validate.
    """

    def __init__(self, name='Validator'):
        """
        @param name: name of the validator
        """
        self._name = name

    @abstractmethod
    def validate(self, env, detector=None):
        """
        @param env: OpenRAVE environment to validate
        @param detector: object detector
        @throws ValidationError if the environment is not valid
        """
        pass

    def __str__(self):
        return self._name


class SequenceValidator(Validator):
    """
    Validates validators in order.
    """

    def __init__(self, validators, name='SequenceValidator'):
        """
        @param validators: list of validators
        @param name: name of the validator
        """
        Validator.__init__(self, name)
        self.validators = validators

    def validate(self, env, detector=None):
        """
        @param env: OpenRAVE environment to validate
        @param detector: object detector
        @throws ValidationError if the environment is not valid
        """
        for validator in self.validators:
            validator.validate(env, detector)


class ValidateSolution(Solution, ExecutableSolution):
    """
    Meta-solution that allows validating an environment after action execution.
    """

    def __init__(self, action, wrapped_solution, validator):
        """
        @param action: Action that generated this Solution or ExecutableSolution
        @param wrapped_solution: Solution or ExecutableSolution that should be
          processed and executed prior to validation
        @param validator: Validator to use to validate the environment after
          execution
        """
        Solution.__init__(self, action,
                          deterministic=wrapped_solution.deterministic)
        ExecutableSolution.__init__(self, self)
        self.wrapped_solution = wrapped_solution
        self.validator = validator

    def save(self, env):
        """
        @param env: OpenRAVE environment
        @return context manager
        """
        return self.wrapped_solution.save(env)

    def jump(self, env):
        """
        @param env: OpenRAVE environment
        """
        self.wrapped_solution.jump(env)

    def postprocess(self, env):
        """
        Postprocess the wrapped_solution and return a ValidateSolution that
        wraps the generated ExecutableSolution.

        @param env: OpenRAVE environment
        @return a ValidateSolution
        """
        # FIXME: if the wrapped_solution is an ExecutableSolution, it does not
        # have a postprocess method and the following line will error.
        executable_solution = self.wrapped_solution.postprocess(env)
        return ValidateSolution(
            action=self.action,
            wrapped_solution=executable_solution,
            validator=self.validator)

    def execute(self, env, simulate):
        """
        Execute the solution, then validate the environment.

        @param env: OpenRAVE environment
        @param simulate: flag to run in simulation
        @return result of executing the wrapped solution
        """
        # FIXME: the execute method should return the result of executing the
        # wrapped solution
        self.wrapped_solution.execute(env, simulate)
        try:
            # FIXME: validate can take a detector, but there's no way to pass
            # one in
            self.validator.validate(env)
        except ExecutionError as err:
            raise ExecutionError(err, solution=self)


class ValidateAction(Action):
    """
    Meta-action that allows an action to be validated after execution.
    """

    def __init__(self, validator, wrapped_action, name=None):
        """
        @param validator: Validator to validate the Action with
        @param wrapped_action: Action to be validated after execution
        @param name: name of the action
        """
        super(ValidateAction, self).__init__(name=name)
        self.action = wrapped_action
        self.validator = validator

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

    def plan(self, env):
        """
        Plan the wrapped action.

        @param env: OpenRAVE environment to plan in
        @return a ValidateSolution object
        """
        solution = self.action.plan(env)
        return ValidateSolution(
            action=self,
            wrapped_solution=solution,
            validator=self.validator)
