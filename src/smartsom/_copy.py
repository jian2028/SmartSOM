"""Detached boundary evidence with scalar-heavy builtin container fast paths."""

import copy

_ATOMIC = frozenset((type(None), bool, int, float, str, bytes))
_DEEPCOPY = copy.deepcopy
_DISPATCH = {
    kind: copy._deepcopy_dispatch[kind] for kind in (*_ATOMIC, dict, list, tuple)
}


def detached_copy(value):
    """Preserve deepcopy aliases, cycles and extension hooks without scalar dispatch.

    Only exact builtins take the fast path. A changed stdlib copy registration
    declines it for the entire graph; unknown objects use the same shared memo.
    No mutable input is retained across calls.
    """
    if copy.deepcopy is not _DEEPCOPY or any(
        copy._deepcopy_dispatch.get(kind) is not handler or kind in copy.dispatch_table
        for kind, handler in _DISPATCH.items()
    ):
        return copy.deepcopy(value)
    return _clone(value, {}, [True])


def _clone(value, memo, plain):
    # An extension hook can replace even scalar memo entries or copy dispatch.
    # After the first unknown object, the remainder uses stdlib with this memo.
    if not plain[0]:
        return _DEEPCOPY(value, memo)
    identity = id(value)
    if identity in memo:
        return memo[identity]
    kind = type(value)
    if kind in _ATOMIC:
        return value
    # While plain is true, only our container entries can be in memo. Scalar
    # overrides require a hook, which disables both row shortcuts below.
    if kind is list:
        if all(type(child) in _ATOMIC for child in value):
            result = value.copy()
            memo[identity] = result
        else:
            result = []
            memo[identity] = result
            result.extend(_clone(child, memo, plain) for child in value)
        memo[identity] = result
        copy._keep_alive(value, memo)
        return result
    if kind is dict:
        if all(type(k) in _ATOMIC and type(v) in _ATOMIC for k, v in value.items()):
            result = value.copy()
            memo[identity] = result
        else:
            result = {}
            memo[identity] = result
            for key, child in value.items():
                result[_clone(key, memo, plain)] = _clone(child, memo, plain)
        memo[identity] = result
        copy._keep_alive(value, memo)
        return result
    if kind is tuple:
        children = [_clone(child, memo, plain) for child in value]
        # A tuple may be reached again through a mutable child while recursing.
        if identity in memo:
            result = memo[identity]
        else:
            result = (
                value
                if all(a is b for a, b in zip(value, children))
                else tuple(children)
            )
        if result is not value:
            memo[identity] = result
            copy._keep_alive(value, memo)
        return result
    plain[0] = False
    return _DEEPCOPY(value, memo)
