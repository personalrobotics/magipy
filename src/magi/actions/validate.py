from .base import Action, Solution, ExecutableSolution, ExecutionError
from abc import ABCMeta, abstractmethod

class Validator(object):
    """
    This class allows for validating the state
    of an environment before/after execution of an action.
    An instance of this class should be created and passed to 
    base.Validate().
    """

    def __init__(self, name='Validator'): 
        self._name = name

    @abstractmethod
    def validate(self, env, detector=None, validating=None):
        """
        @param env The OpenRAVE environment to validate
        @param detector Detector to update environment
        @param validating Action or solution being validated
        @throws ValidationError if the environment is not valid
        """
        pass

    def append(self, validator):
        if not validator:
            return self
        return SequenceValidator(validators=[self, validator])

    def __str__(self):
        return self._name 


class SequenceValidator(Validator): 
    def __init__(self, validators, name='SequenceValidator'):
        """
        Validates validators in-order.
        @param validators: list of validators.
        """
        Validator.__init__(self, name)
        self.validators = validators 
    
    def validate(self, env, detector=None, validating=None):
        """
        @param env The OpenRAVE environment to validate
        @param detector Detector to update environment
        @param validating Action or solution being validated
        """
        for validator in self.validators: 
            validator.validate(env, detector, validating)
    
    def append(self, validator):
        if not validator or validator in self.validators:
            return self

        validators = list(self.validators)
        validators.append(validator)
        return SequenceValidator(validators=validators)

    def __str__(self):
        return ','.join(map(lambda x: str(x), self.validators))

