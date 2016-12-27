#!/usr/bin/env python

"""Define GraspValidator and GraspMeanValidator."""

import csv
import logging

from scipy.linalg import norm
from sklearn import svm
import numpy as np

from magi.actions.base import to_key, from_key, ValidationError
from magi.actions.validate import Validator

LOGGER = logging.getLogger(__name__)


class GraspValidator(Validator):
    """Validate a grasp by using binary classification with an SVM."""

    def __init__(self, infile, robot, dof_indices, name='GraspValidator'):
        """
        @param infile: csvfile containing grasp data used to build the
          classifier
        @param robot: OpenRAVE robot to validate
        @param dof_indices: dof indices to validate
        @param name: name of the validator
        """
        super(GraspValidator, self).__init__(name=name)

        data = []
        # Read in the csv file
        with open(infile, 'r') as in_f:
            reader = csv.reader(in_f, delimiter=' ')
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
        self.data = np.array(data)
        self.X = self.data[:, :-2]
        self.y = self.data[:, -1]

        self.clf = svm.SVC()
        self.clf.fit(self.X, self.y)

        self._robot = to_key(robot)
        self.dof_indices = dof_indices

    def get_robot(self, env):
        """
        Look up and return the robot in the environment.

        @param env: OpenRAVE environment
        @return an OpenRAVE robot
        """
        return from_key(env, self._robot)

    def validate(self, env, detector=None):
        """
        Validate a grasp by predicting with the SVM.

        @param env: OpenRAVE environment
        @param detector: object detector (implements DetectObjects, Update)
        @throws a ValidationError if the grasp is not valid
        """
        robot = self.get_robot(env)
        with env:
            dof_values = robot.GetDOFValues(self.dof_indices)
        pred = self.clf.predict(dof_values.reshape(1, -1))

        # Raise an error if the validation doesn't check out
        if pred[0] == 0:
            raise ValidationError(
                message='Grasp failed validation', validator=self)


class GraspMeanValidator(Validator):
    """
    Validate a grasp by comparing current grasp dof values with (simulated)
    success dof values.
    """

    def __init__(self, object_type, name='GraspMeanValidator'):
        """
        @param object_type: name of object (i.e. 'glass', 'bowl', 'plate')
        @param name: name of the validator
        """
        super(GraspMeanValidator, self).__init__(name=name)

        if object_type == 'glass':
            self.mean = np.array([1.22, 1.16, 1.2])
        elif object_type == 'bowl':
            self.mean = np.array([1.58, 1.56, 0.8])
        elif object_type == 'plate':
            self.mean = np.array([1.46, 1.46, 1.34])
        else:
            raise ValueError('object type not one of (glass, bowl, plate)')

        # May need to be calibrated (or different value per object),
        # but this seems to work for now.
        self.max = 1.0

    def validate(self, env, detector=None):
        """
        Validate a grasp by checking the difference from the simulated dof
        values.

        @param env: OpenRAVE environment
        @throws a ValidationError if the grasp is not valid
        """
        difference = norm(self.mean - np.array(dof_values))
        LOGGER.info('GraspMeanValidator - validating grasp: "%s"',
                    str(difference <= self.max))
        if difference > self.max:
            raise ValidationError(
                message='Grasp failed validation', validator=self)
