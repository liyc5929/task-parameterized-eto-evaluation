import hashlib
import json
import pathlib
import shutil
import uuid
from enum import Enum
from typing import Any, Mapping, Protocol

import torch
from evox.core import Problem
from tensordict import TensorDict, TensorDictBase

from ._sto_knowledge_base import KnowledgeBase


__all__ = [
    "build_sto_knowledge_database",
]


class _KnowledgeBuildingStrategy(Protocol):
    knowledge_spec: Mapping[str, Any]

    def __call__(self,
        problem: Problem,
        knowledge_tensors: TensorDictBase,
    ) -> None:
        ...


def build_sto_knowledge_database(
    problem: Problem,
    num_source_tasks: int,
    strategy: _KnowledgeBuildingStrategy,
    build_seed: int | None,
    data_root: str | pathlib.Path = "./data/", 
    problem_key: str | Enum | None = None,
    force_rebuild: bool = False,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[KnowledgeBase, int, int]:
    """
    Build or load an STO knowledge database backed by a memory-mapped TensorDict.

    The building strategy defines the knowledge schema and fills the
    preallocated memory-mapped tensors in place. An existing database is
    reused when its manifest matches the current request, unless
    `force_rebuild` is enabled. Database construction is performed only in
    the non-distributed setting; distributed callers load an existing
    database and receive their local source-task range.

    Args:
        problem: Problem used to generate the source knowledge.
        num_source_tasks: Number of source tasks or source models represented 
            in the database.
        strategy: Strategy that defines `knowledge_spec` and fills the
            allocated knowledge tensors in place.
        build_seed: Seed used to initialize the PyTorch random number
            generator before knowledge construction. When `None`, the
            existing random number generator state is used without reseeding.
        data_root: Root directory under which knowledge databases are stored.
        problem_key: Stable identifier for the problem. When omitted, the
            `problem_id` attribute of `problem` is used.
        force_rebuild: Whether to rebuild the database when a matching
            database already exists.
        distributed: Whether the caller is running in a distributed setting.
            Construction is not supported when this is enabled.
        rank: Rank of the current distributed process.
        world_size: Total number of distributed processes.

    Returns:
        A tuple containing the loaded knowledge base and the half-open local
        source-task range `[task_start, task_end)`.

    Note:
        When `build_seed` is provided, `torch.manual_seed(build_seed)` is
        called immediately before the building strategy is executed. When it
        is `None`, the current PyTorch RNG state is used as-is. Callers that
        require reproducible cache identity should therefore pass the actual
        construction seed.
    """

    if problem_key is None:
        problem_key = getattr(problem, "problem_id", None)
        if problem_key is None:
            raise ValueError("`problem_key` must be provided when the problem has no `problem_id` attribute.")
    problem_id = str(problem_key.value if isinstance(problem_key, Enum) else problem_key)

    knowledge_spec = strategy.knowledge_spec
    if num_source_tasks != knowledge_spec["dimensions"]["num_sources"]:
        raise ValueError("`num_source_tasks` does not match `knowledge_spec`.")
    manifest: dict[str, Any] = {
        **knowledge_spec,
        "format_version": KnowledgeBase.FORMAT_VERSION,
        "storage_backend": KnowledgeBase.STORAGE_BACKEND,
        "source_collection": {
            "problem_class": type(problem).__name__,
            "problem_id": problem_id,
        },
    }
    manifest["builder"] = {
        **manifest["builder"], 
        "seed": build_seed,
    }
    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    manifest_hash = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()[:10]
    database_id = f"{problem_id.lower()}_{manifest_hash}"
    database_path = pathlib.Path(data_root) / "knowledge_bases" / database_id

    if database_path.exists() and not force_rebuild:
        knowledge_base = KnowledgeBase(database_path, expected_manifest=manifest)
    else:
        if distributed:
            raise NotImplementedError("Knowledge database should be built in the non-distributed setting.")
        temp_path = database_path.with_name(f"{database_id}.temp-{uuid.uuid4().hex}")
        rollback_path = database_path.with_name(f"{database_id}.rollback-{uuid.uuid4().hex}")
        try:
            temp_path.mkdir(parents=True, exist_ok=False)

            schema: Mapping[str, Any] = manifest["schema"]
            field_schemas: Mapping[str, Mapping[str, Any]] = schema["fields"]
            batch_size = torch.Size(schema["batch_size"])
            tensors = TensorDict({}, batch_size=batch_size).memmap_(temp_path / "tensors", existsok=False)
            for field, field_schema in field_schemas.items():
                tensors.make_memmap(
                    key=field,
                    shape=batch_size + torch.Size(field_schema["shape"]),
                    dtype=getattr(torch, field_schema["dtype"]),
                )

            # Fill the preallocated memory-mapped tensors in place
            if build_seed is not None:
                torch.manual_seed(build_seed)
            strategy(problem=problem, knowledge_tensors=tensors)

            # Format TensorDict metadata for readability
            for meta_path in (temp_path / "tensors").rglob("meta.json"):
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta_path.write_text(
                    json.dumps(meta, indent=4, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
            # Write the human-readable manifest while retaining the canonical JSON 
            # representation above for database identity hashing
            (temp_path / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True, indent=4, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            # Reopen and validate the completed temporary database before publishing it
            KnowledgeBase(temp_path, expected_manifest=manifest)
            if database_path.exists():
                database_path.rename(rollback_path)
            temp_path.rename(database_path)
        except Exception:
            if temp_path.exists():
                shutil.rmtree(temp_path)
            if rollback_path.exists() and not database_path.exists():
                rollback_path.rename(database_path)
            raise
        shutil.rmtree(rollback_path, ignore_errors=True)
        knowledge_base = KnowledgeBase(database_path, expected_manifest=manifest)
        print(f"Knowledge database is built and saved into path `{database_path}`.")

    if distributed:
        raise NotImplementedError
    else:
        task_start, task_end = 0, num_source_tasks
    return knowledge_base, task_start, task_end
