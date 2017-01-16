#!/usr/bin/env python

"""Define ObjectPoseValidator."""

import logging

import numpy as np

from table_clearing.perception_utils import get_obj

from magi.actions.base import ValidationError
from magi.actions.validate import Validator

LOGGER = logging.getLogger(__name__)


class ObjectPoseValidator(Validator):
    """Validate that an object is in a pose."""

    def __init__(self, obj_name, pose, name='PoseValidator'):
        """
        @param obj_name: name of the object
        @param pose: expected pose of the object
        @param name: name of the validator
        """
        super(ObjectPoseValidator, self).__init__(name=name)
        self.obj_name = obj_name
        self.expected_pose = pose

    def validate(self, env, detector=None):
        """
        Validate an object by checking the difference from the expected pose.

        @param env: OpenRAVE environment
        @param detector: object detector (implements DetectObjects, Update)
        @throws a ValidationError if pose is not close enough
        """
        if detector:
            LOGGER.info('Redetect before validation')
            detector.Update(env.GetRobots()[0])

        with env:
            obj = get_obj(env, self.obj_name, first_match=True)
            obj_transform = obj.GetTransform()
            # TODO: how close should this be?
            if not np.allclose(obj_transform, self.expected_pose):
                raise ValidationError(
                    message='%s not in expected pose' % self.obj_name,
                    validator=self)
