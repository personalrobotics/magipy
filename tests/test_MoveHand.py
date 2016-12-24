#!/usr/bin/env python

"""Tests for magi.actions.MoveHand."""

import os.path
import sys
import unittest

from prpy.base.endeffector import EndEffector
import prpy

from magi.actions.MoveHand import MoveHandAction
from magi_test_helper import MAGITest

# Is this path manipulation necessary? The current directory should already be on the path.
sys.path = [os.path.abspath(os.path.dirname(__file__))] + sys.path


class DummyHand(EndEffector):
    """Dummy EndEffector that jumps directly to the given DOF values."""

    def __init__(self, manipulator):
        EndEffector.__init__(self, manipulator)
        self.dofs = [0.0]

    def move_hand(self, *dofs):
        """Jump directly to DOFs."""
        dof_values = self.GetDOFValues()
        for idx, dofval in enumerate(dofs):
            if dofval is not None:
                dof_values[idx] = dofval

        self.SetDOFValues(dof_values)

class TestHandMethods(MAGITest):
    """Test hand methods."""

    def test_MoveHand(self):
        """Test MoveHandAction."""
        self.robot.arm = self.robot.GetManipulator('arm')
        self.robot.hand = self.robot.arm.GetEndEffector()
        prpy.bind_subclass(self.robot.hand, DummyHand, manipulator=self.robot.arm)
        action = MoveHandAction(self.robot.hand, [0.7],
                                self.robot.hand.GetIndices()[:1])
        return self._action_helper(action)

if __name__ == '__main__':
    unittest.main()
