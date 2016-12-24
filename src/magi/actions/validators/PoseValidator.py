#!/usr/bin/env python

import logging

import numpy as np

from table_clearing.perception_utils import get_obj

from magi.actions.base import ValidationError
from magi.actions.validate import Validator

LOGGER = logging.getLogger(__name__)


class ObjectPoseValidator(Validator):
    def __init__(self, obj_name, pose, name='PoseValidator'):
        """
        This class validates that object is in pose
        """
        super(ObjectPoseValidator, self).__init__(name=name)
        self.obj_name = obj_name
        self.expected_pose = pose

    def validate(self, env, detector=None):
        """
        @throws An ValidationError if pose is not close enough
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
