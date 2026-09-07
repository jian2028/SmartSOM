"""Shared standard-library domain validation primitives."""


class DomainValidationError(ValueError):
    """An input violates a supported domain rule."""


def _identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise DomainValidationError(f"{label} must be a non-empty string")


def _unique[T](values: tuple[T, ...], label: str) -> None:
    seen: set[T] = set()
    for value in values:
        if value in seen:
            raise DomainValidationError(f"duplicate {label}: {value!r}")
        seen.add(value)


def _items[T](values: tuple[T, ...], item_type: type[T], label: str) -> tuple[T, ...]:
    # Copy caller-owned lists before storing them in a frozen object.
    if not isinstance(values, (tuple, list)) or not values:
        raise DomainValidationError(f"{label} must be a non-empty tuple or list")
    result = tuple(values)
    if any(not isinstance(value, item_type) for value in result):
        raise DomainValidationError(
            f"{label} must contain {item_type.__name__} objects"
        )
    return result
