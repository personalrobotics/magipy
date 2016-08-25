
class RealParameter(object):
    def __init__(self, min_value, max_value, name=None):
        self.min_value = min_value
        self.max_value = max_value
        self.name = name or '<RealParameter>'

    def sample(self):
        from random import uniform
        return uniform(self.min_value, self.max_value)


class DiscreteParameter(object):
    def __init__(self, choices, name=None):
        self.choices = choices
        self.name = name or '<DiscreteParameter>'

    def sample(self):
        from random import choice
        return choice(self.choices)


class DictParameter(object):
    def __init__(self, **parameters):
        self.parameters = parameters

    def sample(self):
        return {
            key: value.sample()
            for key, value in self.parameter.iteritems()
        }
