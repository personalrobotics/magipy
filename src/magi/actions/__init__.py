"""Define MAGI Actions."""


from magi.actions.Disable import DisableAction
from magi.actions.MoveHand import (MoveHandAction, CloseHandAction, OpenHandAction,
                                   GrabObjectAction, ReleaseObjectAction)
from magi.actions.MoveUntilTouch import MoveUntilTouchAction
from magi.actions.PlaceObject import PlaceObjectAction
from magi.actions.Plan import PlanAction, PlanToTSRAction
from magi.actions.PushObject import PushObjectAction
from magi.actions.Sequence import SequenceAction

__all__ = [
    'CloseHandAction',
    'DisableAction',
    'GrabObjectAction',
    'MoveHandAction',
    'MoveUntilTouchAction',
    'OpenHandAction',
    'PlanAction',
    'PlanToTSRAction',
    'PushObjectAction',
    'ReleaseObjectAction',
    'SequenceAction',
]
