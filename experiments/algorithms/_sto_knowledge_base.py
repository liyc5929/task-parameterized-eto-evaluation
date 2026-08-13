import json
import pathlib
from enum import Enum
from typing import Any, Mapping

import torch
from tensordict import TensorDictBase, load_memmap


__all__ = [
    "KnowledgeType",
    "KnowledgeBase",
]


class KnowledgeType(str, Enum):
    SOLUTION_POPULATION = "solution_population"


class KnowledgeBase:
    """
    Read-only handle for a memory-mapped knowledge base.

    `FORMAT_VERSION` is incremented only for backward-incompatible changes
    to the manifest or storage schema.
    """
    FORMAT_VERSION  = 1
    STORAGE_BACKEND = "tensordict_memmap"

    def __init__(self,
        path: str | pathlib.Path,
        expected_manifest: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = pathlib.Path(path)
        with (self.path / "manifest.json").open("r", encoding="utf-8") as file:
            self.manifest: dict[str, Any] = json.load(file)
        self._validate_manifest(expected_manifest)
        self._tensors: TensorDictBase = load_memmap(self.path / "tensors")
        self._validate_storage()

    def __getitem__(self, key):
        return self._tensors[key]

    def _validate_manifest(self, expected_manifest: Mapping[str, Any] | None) -> None:
        required_keys = {
            "format_version", 
            "knowledge_type", 
            "storage_backend", 
            "source_collection",
            "builder", 
            "dimensions", 
            "schema",
        }
        missing_keys = required_keys.difference(self.manifest)
        if missing_keys:
            raise ValueError(f"Knowledge-base manifest is missing keys: {sorted(missing_keys)}")
        if self.manifest["format_version"] != self.FORMAT_VERSION:
            raise ValueError(f"Unsupported knowledge-base format version: {self.manifest['format_version']}")
        if self.manifest["storage_backend"] != self.STORAGE_BACKEND:
            raise ValueError(f"Unsupported knowledge-base storage backend: {self.manifest['storage_backend']}")
        try:
            KnowledgeType(self.manifest["knowledge_type"])
        except ValueError as error:
            raise ValueError(f"Unsupported knowledge type: {self.manifest['knowledge_type']}") from error
        if expected_manifest is not None and self.manifest != expected_manifest:
            raise ValueError("The cached knowledge-base manifest is incompatible with the current request.")

    def _validate_storage(self) -> None:
        schema: Mapping[str, Any] = self.manifest["schema"]
        fields: Mapping[str, Mapping[str, Any]] = schema["fields"]
        batch_size = torch.Size(schema["batch_size"])
        if self._tensors.batch_size != batch_size:
            raise ValueError(f"TensorDict batch size {self._tensors.batch_size} does not match {batch_size}.")
        for field, field_schema in fields.items():
            tensor = self._tensors[field]
            expected_shape = batch_size + torch.Size(field_schema["shape"])
            expected_dtype = getattr(torch, field_schema["dtype"], None)
            if not isinstance(expected_dtype, torch.dtype):
                raise ValueError(f"Unsupported dtype for knowledge field `{field}`: {field_schema['dtype']}")
            if tensor.shape != expected_shape or tensor.dtype != expected_dtype:
                raise ValueError(
                    f"Knowledge field `{field}` has shape/dtype {tensor.shape}/{tensor.dtype}, "
                    f"expected {expected_shape}/{expected_dtype}."
                )
