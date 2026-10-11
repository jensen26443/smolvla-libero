"""LIBERO task registry used by the frozen three-task protocol."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskSpec:
    key: str
    suite: str
    suite_task_id: int
    dataset_task_index: int
    task_name: str
    instruction: str
    target_object: str
    target_region: str
    scene_group: str = "libero_goal_common_scene"
    language_pair_status: str = "requires_runtime_alignment_check"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# The suite-local IDs follow the official LIBERO_GOAL order. The dataset task
# indices follow HuggingFaceVLA/libero/meta/tasks.parquet, which concatenates
# LIBERO-10, Goal, Object and Spatial tasks into one table.
TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(
        key="goal_bowl_plate",
        suite="libero_goal",
        suite_task_id=8,
        dataset_task_index=10,
        task_name="put_the_bowl_on_the_plate",
        instruction="Put the bowl on the plate",
        target_object="akita_black_bowl_1",
        target_region="plate_1",
    ),
    TaskSpec(
        key="goal_bowl_stove",
        suite="libero_goal",
        suite_task_id=1,
        dataset_task_index=17,
        task_name="put_the_bowl_on_the_stove",
        instruction="Put the bowl on the stove",
        target_object="akita_black_bowl_1",
        target_region="flat_stove_1_cook_region",
    ),
    TaskSpec(
        key="goal_bowl_cabinet_top",
        suite="libero_goal",
        suite_task_id=4,
        dataset_task_index=18,
        task_name="put_the_bowl_on_top_of_the_cabinet",
        instruction="Put the bowl on top of the cabinet",
        target_object="akita_black_bowl_1",
        target_region="wooden_cabinet_1_top_region",
    ),
)

TASK_BY_KEY = {task.key: task for task in TASKS}
TASK_BY_DATASET_INDEX = {task.dataset_task_index: task for task in TASKS}


def selected_tasks(keys: Iterable[str] | None = None) -> tuple[TaskSpec, ...]:
    wanted = tuple(TASK_BY_KEY) if keys is None else tuple(keys)
    unknown = [key for key in wanted if key not in TASK_BY_KEY]
    if unknown:
        raise KeyError(f"unknown task keys: {unknown}; choose from {sorted(TASK_BY_KEY)}")
    if len(set(wanted)) != len(wanted):
        raise ValueError("task selection contains duplicates")
    return tuple(TASK_BY_KEY[key] for key in wanted)


def selected_suite_ids(keys: Iterable[str] | None = None) -> list[int]:
    return [task.suite_task_id for task in selected_tasks(keys)]


def selected_dataset_indices(keys: Iterable[str] | None = None) -> list[int]:
    return [task.dataset_task_index for task in selected_tasks(keys)]


def all_task_metadata() -> list[dict[str, object]]:
    return [task.to_dict() for task in TASKS]


def _strip_goal_block(text: str) -> str:
    """Remove the BDDL goal block while preserving scene/object declarations."""
    match = re.search(r"\(:goal\s*", text, flags=re.IGNORECASE)
    if match is None:
        return text
    start = match.start()
    depth = 0
    in_string = False
    for index in range(match.end() - 1, len(text)):
        char = text[index]
        if char == '"' and (index == 0 or text[index - 1] != "\\"):
            in_string = not in_string
        if in_string:
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[:start] + text[index + 1 :]
    raise ValueError("unterminated BDDL goal block")


def _strip_named_blocks(text: str, names: Iterable[str]) -> str:
    """Remove top-level ``:keyword`` blocks from a BDDL document.

    LIBERO stores the natural-language description and the success-target
    bookkeeping in the same BDDL file as the visual scene.  Those fields must
    not be included when checking whether two files describe the same visual
    scene.  This small balanced-parenthesis scanner is intentionally local and
    conservative; it does not attempt to parse the full BDDL grammar.
    """

    result = text
    for name in names:
        pattern = re.compile(rf"\(\s*:{re.escape(name)}\b", flags=re.IGNORECASE)
        match = pattern.search(result)
        while match is not None:
            start = match.start()
            depth = 0
            in_string = False
            end: int | None = None
            for index in range(start, len(result)):
                char = result[index]
                if char == '"' and (index == 0 or result[index - 1] != "\\"):
                    in_string = not in_string
                if in_string:
                    continue
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        end = index + 1
                        break
            if end is None:
                raise ValueError(f"unterminated BDDL :{name} block")
            result = result[:start] + result[end:]
            match = pattern.search(result)
    return result


def bddl_scene_signature(paths: Iterable[str | Path]) -> dict[str, object]:
    """Compare BDDL scene declarations and return an auditable signature.

    A matching signature is necessary but not sufficient for paired-language
    evaluation: the corresponding initial-state assets still need a runtime
    alignment check.
    """
    normalized: dict[str, str] = {}
    for path in paths:
        source = Path(path)
        text = re.sub(r"\s+", " ", _strip_goal_block(source.read_text(encoding="utf-8")).strip())
        normalized[str(source)] = text
    hashes = {path: hashlib.sha256(value.encode()).hexdigest() for path, value in normalized.items()}
    return {"matching": len(set(hashes.values())) <= 1, "sha256_without_goal": hashes}


def bddl_visual_scene_signature(paths: Iterable[str | Path]) -> dict[str, object]:
    """Compare BDDL visual-scene declarations while excluding target metadata.

    ``:language``, ``:obj_of_interest`` and ``:goal`` are task-condition
    metadata, not geometry or initial placement declarations.  This signature
    is therefore useful as the first (static) stage of an E3/E4 pairing gate,
    but it is deliberately not sufficient on its own: the associated
    ``*.pruned_init`` assets and runtime observations still have to match.
    """

    normalized: dict[str, str] = {}
    for path in paths:
        source = Path(path)
        text = _strip_named_blocks(
            source.read_text(encoding="utf-8"), ("language", "obj_of_interest", "goal")
        )
        normalized[str(source)] = re.sub(r"\s+", " ", text.strip())
    hashes = {path: hashlib.sha256(value.encode()).hexdigest() for path, value in normalized.items()}
    return {"matching": len(set(hashes.values())) <= 1, "sha256_visual_scene": hashes}
