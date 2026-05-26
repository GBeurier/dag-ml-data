"""Minimal ctypes wrapper for the dag-ml-data C ABI provider vtable.

This example intentionally depends only on the Python standard library. It is a
binding smoke, not the final Python package API.
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path
from typing import Any


class DagMlDataString(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_void_p), ("len", ctypes.c_size_t)]


class DagMlDataBytesView(ctypes.Structure):
    _fields_ = [("ptr", ctypes.POINTER(ctypes.c_uint8)), ("len", ctypes.c_size_t)]


class ArrowArray(ctypes.Structure):
    pass


class ArrowSchema(ctypes.Structure):
    pass


ArrowArray._fields_ = [
    ("length", ctypes.c_int64),
    ("null_count", ctypes.c_int64),
    ("offset", ctypes.c_int64),
    ("n_buffers", ctypes.c_int64),
    ("n_children", ctypes.c_int64),
    ("buffers", ctypes.POINTER(ctypes.c_void_p)),
    ("children", ctypes.POINTER(ctypes.POINTER(ArrowArray))),
    ("dictionary", ctypes.POINTER(ArrowArray)),
    ("release", ctypes.c_void_p),
    ("private_data", ctypes.c_void_p),
]

ArrowSchema._fields_ = [
    ("format", ctypes.c_char_p),
    ("name", ctypes.c_char_p),
    ("metadata", ctypes.c_char_p),
    ("flags", ctypes.c_int64),
    ("n_children", ctypes.c_int64),
    ("children", ctypes.POINTER(ctypes.POINTER(ArrowSchema))),
    ("dictionary", ctypes.POINTER(ArrowSchema)),
    ("release", ctypes.c_void_p),
    ("private_data", ctypes.c_void_p),
]


MaterializeFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_uint64,
    DagMlDataBytesView,
    ctypes.POINTER(ctypes.c_uint64),
)
MakeViewFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_uint64,
    DagMlDataBytesView,
    ctypes.POINTER(ctypes.c_uint64),
)
ViewIdentityFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_uint64,
    ctypes.POINTER(ctypes.POINTER(ArrowArray)),
    ctypes.POINTER(ctypes.POINTER(ArrowSchema)),
)
TargetArrowFn = ctypes.CFUNCTYPE(
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_uint64,
    DagMlDataBytesView,
    ctypes.POINTER(ctypes.POINTER(ArrowArray)),
    ctypes.POINTER(ctypes.POINTER(ArrowSchema)),
)
ReleaseFn = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_uint64)
DestroyFn = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


class DagMlDataVTable(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("user_data", ctypes.c_void_p),
        ("materialize", MaterializeFn),
        ("make_view", MakeViewFn),
        ("view_identity", ViewIdentityFn),
        ("target_arrow", TargetArrowFn),
        ("release", ReleaseFn),
        ("destroy", DestroyFn),
    ]


def _u8_buffer(data: bytes) -> tuple[ctypes.Array[ctypes.c_char], ctypes.POINTER(ctypes.c_uint8)]:
    buffer = ctypes.create_string_buffer(data)
    return buffer, ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint8))


def _bytes_view(data: bytes) -> tuple[ctypes.Array[ctypes.c_char], DagMlDataBytesView]:
    buffer, ptr = _u8_buffer(data)
    return buffer, DagMlDataBytesView(ptr, len(data))


def _valid(bitmap: int | None, idx: int) -> bool:
    if not bitmap:
        return True
    byte = ctypes.cast(bitmap, ctypes.POINTER(ctypes.c_uint8))[idx // 8]
    return (byte & (1 << (idx % 8))) != 0


def _utf8_values(array_ptr: ctypes.POINTER(ArrowArray)) -> list[str | None]:
    array = array_ptr.contents
    offsets = ctypes.cast(array.buffers[1], ctypes.POINTER(ctypes.c_int32))
    value_ptr = ctypes.cast(array.buffers[2], ctypes.POINTER(ctypes.c_uint8))
    values: list[str | None] = []
    for idx in range(array.length):
        if not _valid(array.buffers[0], idx):
            values.append(None)
            continue
        start = offsets[idx]
        end = offsets[idx + 1]
        values.append(bytes(value_ptr[start:end]).decode("utf-8"))
    return values


def _bool_values(array_ptr: ctypes.POINTER(ArrowArray)) -> list[bool]:
    array = array_ptr.contents
    value_ptr = array.buffers[1]
    return [_valid(value_ptr, idx) for idx in range(array.length)]


def _f64_values(array_ptr: ctypes.POINTER(ArrowArray)) -> list[float | None]:
    array = array_ptr.contents
    values = ctypes.cast(array.buffers[1], ctypes.POINTER(ctypes.c_double))
    return [values[idx] if _valid(array.buffers[0], idx) else None for idx in range(array.length)]


class InMemoryProvider:
    def __init__(
        self,
        library_path: str | Path,
        envelope_json: bytes,
        target_tables: list[dict[str, Any]] | None = None,
    ) -> None:
        self._lib = ctypes.CDLL(str(library_path))
        self._configure_lib()
        target_json = json.dumps(target_tables or []).encode("utf-8")
        envelope_buffer, envelope_ptr = _u8_buffer(envelope_json)
        target_buffer, target_ptr = _u8_buffer(target_json)
        self._buffers = [envelope_buffer, target_buffer]
        self._vtable = DagMlDataVTable()
        error = DagMlDataString()
        status = self._lib.dagmldata_inmemory_provider_new_json(
            envelope_ptr,
            len(envelope_json),
            target_ptr,
            len(target_json),
            ctypes.byref(self._vtable),
            ctypes.byref(error),
        )
        if status != 0:
            message = self._consume_error(error) or f"status {status}"
            raise RuntimeError(f"provider creation failed: {message}")
        if not self._vtable.user_data:
            raise RuntimeError("provider creation returned null user_data")

    @classmethod
    def from_files(
        cls,
        library_path: str | Path,
        envelope_path: str | Path,
        target_tables: list[dict[str, Any]] | None = None,
    ) -> "InMemoryProvider":
        return cls(Path(library_path), Path(envelope_path).read_bytes(), target_tables)

    def materialize(self, request: dict[str, Any] | bytes) -> int:
        payload = request if isinstance(request, bytes) else json.dumps(request).encode("utf-8")
        buffer, view = _bytes_view(payload)
        self._buffers.append(buffer)
        handle = ctypes.c_uint64()
        status = self._vtable.materialize(self._vtable.user_data, 0, view, ctypes.byref(handle))
        if status != 0:
            raise RuntimeError(f"materialize failed: status {status}")
        return int(handle.value)

    def materialize_file(self, request_path: str | Path) -> int:
        return self.materialize(Path(request_path).read_bytes())

    def make_view(self, data_handle: int, view_spec: dict[str, Any]) -> int:
        buffer, view = _bytes_view(json.dumps(view_spec).encode("utf-8"))
        self._buffers.append(buffer)
        handle = ctypes.c_uint64()
        status = self._vtable.make_view(
            self._vtable.user_data,
            data_handle,
            view,
            ctypes.byref(handle),
        )
        if status != 0:
            raise RuntimeError(f"make_view failed: status {status}")
        return int(handle.value)

    def view_identity(self, view_handle: int) -> list[dict[str, Any]]:
        array, schema = self._call_arrow(self._vtable.view_identity, view_handle)
        try:
            children = array.contents.children
            columns = [
                _utf8_values(children[0]),
                _utf8_values(children[1]),
                _utf8_values(children[2]),
                _utf8_values(children[3]),
                _utf8_values(children[4]),
                _utf8_values(children[5]),
                _bool_values(children[6]),
            ]
            return [
                {
                    "observation_id": columns[0][idx],
                    "sample_id": columns[1][idx],
                    "target_id": columns[2][idx],
                    "group_id": columns[3][idx],
                    "origin_sample_id": columns[4][idx],
                    "source_id": columns[5][idx],
                    "is_augmented": columns[6][idx],
                }
                for idx in range(array.contents.length)
            ]
        finally:
            self._lib.dagmldata_arrow_array_free(array)
            self._lib.dagmldata_arrow_schema_free(schema)

    def target_values(self, view_handle: int, target_id: str) -> list[dict[str, Any]]:
        target_buffer, target_view = _bytes_view(target_id.encode("utf-8"))
        self._buffers.append(target_buffer)
        array = ctypes.POINTER(ArrowArray)()
        schema = ctypes.POINTER(ArrowSchema)()
        status = self._vtable.target_arrow(
            self._vtable.user_data,
            view_handle,
            target_view,
            ctypes.byref(array),
            ctypes.byref(schema),
        )
        if status != 0:
            raise RuntimeError(f"target_arrow failed: status {status}")
        try:
            children = array.contents.children
            sample_ids = _utf8_values(children[0])
            target_ids = _utf8_values(children[1])
            values = _f64_values(children[2])
            return [
                {"sample_id": sample_ids[idx], "target_id": target_ids[idx], "value": values[idx]}
                for idx in range(array.contents.length)
            ]
        finally:
            self._lib.dagmldata_arrow_array_free(array)
            self._lib.dagmldata_arrow_schema_free(schema)

    def release(self, handle: int) -> None:
        self._vtable.release(self._vtable.user_data, handle)

    def close(self) -> None:
        if self._vtable.user_data:
            self._lib.dagmldata_inmemory_provider_destroy(ctypes.byref(self._vtable))

    def __enter__(self) -> "InMemoryProvider":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def _configure_lib(self) -> None:
        self._lib.dagmldata_inmemory_provider_new_json.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.POINTER(DagMlDataVTable),
            ctypes.POINTER(DagMlDataString),
        ]
        self._lib.dagmldata_inmemory_provider_new_json.restype = ctypes.c_int
        self._lib.dagmldata_inmemory_provider_destroy.argtypes = [ctypes.POINTER(DagMlDataVTable)]
        self._lib.dagmldata_inmemory_provider_destroy.restype = None
        self._lib.dagmldata_string_free.argtypes = [DagMlDataString]
        self._lib.dagmldata_string_free.restype = None
        self._lib.dagmldata_arrow_array_free.argtypes = [ctypes.POINTER(ArrowArray)]
        self._lib.dagmldata_arrow_array_free.restype = None
        self._lib.dagmldata_arrow_schema_free.argtypes = [ctypes.POINTER(ArrowSchema)]
        self._lib.dagmldata_arrow_schema_free.restype = None

    def _consume_error(self, error: DagMlDataString) -> str | None:
        if not error.ptr:
            return None
        value = ctypes.string_at(error.ptr, error.len).decode("utf-8")
        self._lib.dagmldata_string_free(error)
        return value

    def _call_arrow(
        self,
        callback: ViewIdentityFn,
        view_handle: int,
    ) -> tuple[ctypes.POINTER(ArrowArray), ctypes.POINTER(ArrowSchema)]:
        array = ctypes.POINTER(ArrowArray)()
        schema = ctypes.POINTER(ArrowSchema)()
        status = callback(self._vtable.user_data, view_handle, ctypes.byref(array), ctypes.byref(schema))
        if status != 0:
            raise RuntimeError(f"Arrow callback failed: status {status}")
        return array, schema
