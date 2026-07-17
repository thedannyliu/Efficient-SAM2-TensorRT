from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Mapping, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class MemorySelection(Generic[T]):
    memories: tuple[tuple[int, int, T], ...]
    pointers: tuple[tuple[int, int, T], ...]


def select_closest_conditioning(
    frame_idx: int, conditioning: Mapping[int, T], maximum: int
) -> tuple[dict[int, T], dict[int, T]]:
    if maximum == -1 or len(conditioning) <= maximum:
        return dict(conditioning), {}
    if maximum < 2:
        raise ValueError("maximum conditioning frames must be -1 or at least 2")
    selected: dict[int, T] = {}
    before = max((idx for idx in conditioning if idx < frame_idx), default=None)
    after = min((idx for idx in conditioning if idx >= frame_idx), default=None)
    if before is not None:
        selected[before] = conditioning[before]
    if after is not None:
        selected[after] = conditioning[after]
    remaining = sorted(
        (idx for idx in conditioning if idx not in selected),
        key=lambda idx: abs(idx - frame_idx),
    )[: maximum - len(selected)]
    selected.update((idx, conditioning[idx]) for idx in remaining)
    return selected, {idx: value for idx, value in conditioning.items() if idx not in selected}


def select_memory(
    *,
    frame_idx: int,
    num_frames: int,
    conditioning: Mapping[int, T],
    non_conditioning: Mapping[int, T],
    num_maskmem: int = 7,
    max_conditioning: int = -1,
    stride: int = 1,
    max_object_pointers: int = 16,
    reverse: bool = False,
    pointers_only_in_past: bool = True,
) -> MemorySelection[T]:
    """Mirror SAM2.1's inference-time memory and pointer frame selection.

    Each memory tuple is ``(temporal_position, frame_index, value)`` and each
    pointer tuple is ``(frame_distance, frame_index, value)``. Tensor layout and
    temporal embeddings are deliberately left to the device runtime.
    """
    if num_maskmem < 1 or stride < 1 or max_object_pointers < 1:
        raise ValueError("num_maskmem, stride, and max_object_pointers must be positive")

    selected, unselected = select_closest_conditioning(
        frame_idx, conditioning, max_conditioning
    )
    memories: list[tuple[int, int, T]] = [
        (0, idx, value) for idx, value in selected.items()
    ]
    direction = 1 if reverse else -1
    for temporal_position in range(1, num_maskmem):
        relative = num_maskmem - temporal_position
        if relative == 1:
            previous_idx = frame_idx + direction
        elif reverse:
            previous_idx = -(-(frame_idx + 2) // stride) * stride
            previous_idx += (relative - 2) * stride
        else:
            previous_idx = ((frame_idx - 2) // stride) * stride
            previous_idx -= (relative - 2) * stride
        value = non_conditioning.get(previous_idx, unselected.get(previous_idx))
        if value is not None:
            memories.append((temporal_position, previous_idx, value))

    maximum = min(num_frames, max_object_pointers)
    if pointers_only_in_past:
        pointer_conditioning = {
            idx: value
            for idx, value in selected.items()
            if (idx >= frame_idx if reverse else idx <= frame_idx)
        }
    else:
        pointer_conditioning = selected
    sign = -1 if reverse else 1
    pointers: list[tuple[int, int, T]] = [
        (sign * (frame_idx - idx), idx, value)
        for idx, value in pointer_conditioning.items()
    ]
    for distance in range(1, maximum):
        idx = frame_idx + distance if reverse else frame_idx - distance
        if idx < 0 or idx >= num_frames:
            break
        value = non_conditioning.get(idx, unselected.get(idx))
        if value is not None:
            pointers.append((distance, idx, value))
    return MemorySelection(tuple(memories), tuple(pointers))


def object_batch_size(count: int) -> int:
    if not 1 <= count <= 8:
        raise ValueError("object count must be in [1, 8]")
    return next(size for size in (1, 2, 4, 8) if count <= size)


def bucket_by_memory_length(selections: Mapping[int, MemorySelection[T]]) -> dict[tuple[int, int], list[int]]:
    buckets: dict[tuple[int, int], list[int]] = {}
    for object_id, selection in selections.items():
        key = (len(selection.memories), len(selection.pointers))
        buckets.setdefault(key, []).append(object_id)
    return buckets
