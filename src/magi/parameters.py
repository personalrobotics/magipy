"""Define RealParameter, DiscreteParameter, DictParameter."""

from abc import ABCMeta, abstractmethod
from random import choice, uniform

class Parameter(object):
    """Parameters return samples from their distribution."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def sample(self):
        """
        Return a sample of this parameter.
        """
        pass

class RealParameter(Parameter):
    """Sample real numbers from a uniform distribution."""

    def __init__(self, min_value, max_value, name=None):
        """
        @param min_value: minimum value of uniform distribution
        @param max_value: maximum value of uniform distribution
        @param name: name of the parameter
        """
        self.min_value = min_value
        self.max_value = max_value
        self.name = name or '<RealParameter>'

    def sample(self):
        """
        Return a sample from the uniform distribution [min_value, max_value].

        @return a double
        """
        return uniform(self.min_value, self.max_value)


class DiscreteParameter(Parameter):
    """Sample uniformly from a discrete distribution."""

    def __init__(self, choices, name=None):
        """
        @param choices: a non-empty sequence of choices
        @param name: name of the parameter
        """
        self.choices = choices
        self.name = name or '<DiscreteParameter>'

    def sample(self):
        """
        Return a sample from the choices.

        @return an element of self.choices
        """
        return choice(self.choices)


class DictParameter(Parameter):
    """Sample several parameters."""

    def __init__(self, **parameters):
        """
        @params parameters: sequence of Parameters to sample
        """
        self.parameters = parameters

    def sample(self):
        """
        Return a dictionary of parameter names and values by recursively
        sampling from the Parameters.

        @return a dictionary
        """
        return {
            key: value.sample()
            for key, value in self.parameters.iteritems()
        }
