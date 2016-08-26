#!/usr/bin/env python
import os.path, sys
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path

import unittest
from magi_test_helper import MAGITest
from prpy.base.endeffector import EndEffector

class DummyHand(EndEffector):
    def __init__(self, manipulator):
        EndEffector.__init__(self, manipulator)
        self.dofs = [0.0]

    def MoveHand(self, *dofs, **kwargs):
        dof_values = self.GetDOFValues()
        for idx, v in enumerate(dofs):
            if v is not None:
                dof_values[idx] = v

        self.SetDOFValues(dof_values)

class TestHandMethods(MAGITest):

    def test_MoveHand(self):
        import prpy
        self.robot.arm = self.robot.GetManipulator('arm')
        self.robot.hand = self.robot.arm.GetEndEffector()
        prpy.bind_subclass(self.robot.hand, DummyHand, manipulator=self.robot.arm)
        from magi.actions.MoveHand import MoveHandAction
        action = MoveHandAction(self.robot.hand, [0.7],
                                self.robot.hand.GetIndices()[:1])
        return self._action_helper(action)

if __name__ == '__main__':
    unittest.main()
