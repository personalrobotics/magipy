#!/usr/bin/env python
from validate import Validator
from base import to_key, from_key, ExecutionError

import logging
logger = logging.getLogger(__name__)


class GraspValidator(Validator):
    def __init__(self, infile, robot, dof_indices, name='GraspValidator'):
        """
        This class validates a grasp by using performing
        simply binary classification using an SVM
        @param infile A csvfile containing grasp data used to build the classifier
        @param robot The OpenRAVE robot to validate
        @param dof_indices The dof indices to validate
        """
        super(GraspValidator, self).__init__(name=name)

        import csv, numpy
        from sklearn import svm

        data = []
        # Read in the csv file
        with open(infile, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                data.append(row)

        # Chop off the header
        data = data[1:]

        # Convert the classification bit to binary
        for row in data:
            if row[-1] == 'y':
                row[-1] = 1
            else:
                row[-1] = 0

            # Convert everything else to float
            row[:-1] = [float(v) for v in row[:-1]]

        # Now build the classifier
        self.data = numpy.array(data)
        self.X = self.data[:, :-2]
        self.y = self.data[:, -1]

        self.clf = svm.SVC()
        self.clf.fit(self.X, self.y)

        self._robot = to_key(robot)
        self.dof_indices = dof_indices

    def get_robot(self, env):
        """
        Lookup and return the appropriate robot from the environment
        @param env The OpenRAVE environment
        """
        return from_key(env, self._robot)

    def validate(self, env):
        """
        @throws An ExecutionError if the grasp is not valid
        """
        robot = self.get_robot(env)
        with env:
            dof_values = robot.GetDOFValues(self.dof_indices)
        v = self.clf.predict(dof_values)

        # Raise an error if the validation doesn't check out
        if v[0] == 0:
            raise ExecutionError('Grasp failed validation')


class GraspMeanValidator(Validator):
    def __init__(self, object_type, name='GraspMeanValidator'):
        """ 
        Validates grasp by comparing current grasp dof values 
        with (simulated) success dof values
        """
        super(GraspMeanValidator, self).__init__(name=name)
        import numpy as np

        if object_type == 'glass':
            self.mean = np.array([1.22, 1.16, 1.2])
        elif object_type == 'bowl':
            self.mean = np.array([1.58, 1.56, 0.8])
        elif object_type == 'plate':
            self.mean = np.array([1.46, 1.46, 1.34])
        else:
            raise ValueError('object type not one of (glass, bowl, plate)')

        self.max = 1.0  # may need to be calibrated (or different value per object, but this seems to work for now. 

    def validate(self, env):
        """
        @throws An ExecutionError if the grasp is not valid
        """
        import numpy as np
        from scipy.linalg import norm

        difference = norm(self.mean - np.array(dof_values))
        logger.info('GraspMeanValidator - validating grasp: "%s"',
                    str(difference <= self.max))
        if difference > self.max:
            raise ExecutionError('Grasp failed validation')
