import typing as tp


__all__ = ['Pick']


class Pick(tp.NamedTuple):
    entity: tp.Any


class ActionPick(tp.NamedTuple):
    leftPressEvent: tp.Any
