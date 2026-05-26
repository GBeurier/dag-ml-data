use std::ffi::{c_void, CString};
use std::os::raw::c_char;
use std::slice;

use dag_ml_data_core::{
    schema_fingerprint, CoordinatorDataMaterializationRequest, CoordinatorDataPlanEnvelope,
    CoordinatorHandleArena, CoordinatorTargetBlock, CoordinatorTargetTable, DataView,
    DatasetSchema,
};
use serde::Deserialize;

pub type DagMlDataHandle = u64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DagMlDataStatusCode {
    Ok = 0,
    InvalidArgument = 1,
    ValidationError = 2,
    Panic = 255,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DagMlDataVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DagMlDataString {
    pub ptr: *mut c_char,
    pub len: usize,
}

impl Default for DagMlDataString {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            len: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DagMlDataBytesView {
    pub ptr: *const u8,
    pub len: usize,
}

#[repr(C)]
pub struct ArrowArray {
    pub length: i64,
    pub null_count: i64,
    pub offset: i64,
    pub n_buffers: i64,
    pub n_children: i64,
    pub buffers: *mut *const c_void,
    pub children: *mut *mut ArrowArray,
    pub dictionary: *mut ArrowArray,
    pub release: Option<unsafe extern "C" fn(array: *mut ArrowArray)>,
    pub private_data: *mut c_void,
}

#[repr(C)]
pub struct ArrowSchema {
    pub format: *const c_char,
    pub name: *const c_char,
    pub metadata: *const c_char,
    pub flags: i64,
    pub n_children: i64,
    pub children: *mut *mut ArrowSchema,
    pub dictionary: *mut ArrowSchema,
    pub release: Option<unsafe extern "C" fn(schema: *mut ArrowSchema)>,
    pub private_data: *mut c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DagMlDataVTable {
    pub abi_version: u32,
    pub user_data: *mut c_void,
    pub materialize: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            dataset: DagMlDataHandle,
            request_json: DagMlDataBytesView,
            out_handle: *mut DagMlDataHandle,
        ) -> DagMlDataStatusCode,
    >,
    pub make_view: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            dataset: DagMlDataHandle,
            selector_json: DagMlDataBytesView,
            out_view: *mut DagMlDataHandle,
        ) -> DagMlDataStatusCode,
    >,
    pub view_identity: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            view: DagMlDataHandle,
            out_arrow_array: *mut *mut ArrowArray,
            out_arrow_schema: *mut *mut ArrowSchema,
        ) -> DagMlDataStatusCode,
    >,
    pub target_arrow: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            view: DagMlDataHandle,
            target_name: DagMlDataBytesView,
            out_arrow_array: *mut *mut ArrowArray,
            out_arrow_schema: *mut *mut ArrowSchema,
        ) -> DagMlDataStatusCode,
    >,
    pub release: Option<unsafe extern "C" fn(user_data: *mut c_void, handle: DagMlDataHandle)>,
    pub destroy: Option<unsafe extern "C" fn(user_data: *mut c_void)>,
}

#[no_mangle]
pub extern "C" fn dagmldata_version() -> DagMlDataVersion {
    DagMlDataVersion {
        major: 0,
        minor: 1,
        patch: 0,
    }
}

/// Releases a string allocated by DAG-ML-DATA.
///
/// # Safety
///
/// `value.ptr` must either be null or a pointer previously returned by a
/// DAG-ML-DATA C ABI function in a `DagMlDataString`. Passing any other pointer,
/// or freeing the same string twice, is undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_string_free(value: DagMlDataString) {
    if !value.ptr.is_null() {
        drop(CString::from_raw(value.ptr));
    }
}

/// Releases an `ArrowArray` allocated by DAG-ML-DATA.
///
/// # Safety
///
/// `array` must either be null or a pointer returned by a DAG-ML-DATA C ABI
/// function. Passing any other pointer, or freeing the same pointer twice, is
/// undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_arrow_array_free(array: *mut ArrowArray) {
    if array.is_null() {
        return;
    }
    if let Some(release) = (*array).release {
        release(array);
    }
    drop(Box::from_raw(array));
}

/// Releases an `ArrowSchema` allocated by DAG-ML-DATA.
///
/// # Safety
///
/// `schema` must either be null or a pointer returned by a DAG-ML-DATA C ABI
/// function. Passing any other pointer, or freeing the same pointer twice, is
/// undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_arrow_schema_free(schema: *mut ArrowSchema) {
    if schema.is_null() {
        return;
    }
    if let Some(release) = (*schema).release {
        release(schema);
    }
    drop(Box::from_raw(schema));
}

/// Computes the deterministic fingerprint of a canonical JSON `DatasetSchema`.
///
/// # Safety
///
/// When `json_ptr` is non-null it must point to `json_len` readable bytes for
/// the duration of the call. `fingerprint_out` and `error_out` may be null; when
/// non-null each must point to writable memory for one `DagMlDataString`. Any
/// returned string must be released with `dagmldata_string_free`.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_schema_fingerprint_json(
    json_ptr: *const u8,
    json_len: usize,
    fingerprint_out: *mut DagMlDataString,
    error_out: *mut DagMlDataString,
) -> DagMlDataStatusCode {
    clear_string(fingerprint_out);
    clear_string(error_out);
    if json_ptr.is_null() {
        set_string(error_out, "json pointer is null");
        return DagMlDataStatusCode::InvalidArgument;
    }

    let json = slice::from_raw_parts(json_ptr, json_len);
    match serde_json::from_slice::<DatasetSchema>(json) {
        Ok(schema) => match schema_fingerprint(&schema) {
            Ok(fingerprint) => {
                set_string(fingerprint_out, fingerprint);
                DagMlDataStatusCode::Ok
            }
            Err(error) => {
                set_string(error_out, error.to_string());
                DagMlDataStatusCode::ValidationError
            }
        },
        Err(error) => {
            set_string(error_out, error.to_string());
            DagMlDataStatusCode::ValidationError
        }
    }
}

/// Builds an Arrow C Data identity table from a coordinator data-plan envelope.
///
/// The returned table has one row per coordinator relation and these columns:
/// `observation_id`, `sample_id`, `target_id`, `group_id`,
/// `origin_sample_id`, `source_id`, `is_augmented`.
///
/// # Safety
///
/// When `json_ptr` is non-null it must point to `json_len` readable bytes for
/// the duration of the call. `out_arrow_array`, `out_arrow_schema` and
/// `error_out` may be null. Returned Arrow pointers must be released with
/// `dagmldata_arrow_array_free` and `dagmldata_arrow_schema_free`.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_coordinator_identity_arrow_json(
    json_ptr: *const u8,
    json_len: usize,
    out_arrow_array: *mut *mut ArrowArray,
    out_arrow_schema: *mut *mut ArrowSchema,
    error_out: *mut DagMlDataString,
) -> DagMlDataStatusCode {
    clear_arrow_array(out_arrow_array);
    clear_arrow_schema(out_arrow_schema);
    clear_string(error_out);
    if json_ptr.is_null() {
        set_string(error_out, "json pointer is null");
        return DagMlDataStatusCode::InvalidArgument;
    }
    if out_arrow_array.is_null() || out_arrow_schema.is_null() {
        set_string(error_out, "arrow output pointer is null");
        return DagMlDataStatusCode::InvalidArgument;
    }

    let json = slice::from_raw_parts(json_ptr, json_len);
    match serde_json::from_slice::<CoordinatorDataPlanEnvelope>(json) {
        Ok(envelope) => match build_identity_arrow(&envelope) {
            Ok((array, schema)) => {
                *out_arrow_array = Box::into_raw(Box::new(array));
                *out_arrow_schema = Box::into_raw(Box::new(schema));
                DagMlDataStatusCode::Ok
            }
            Err(error) => {
                set_string(error_out, error.to_string());
                DagMlDataStatusCode::ValidationError
            }
        },
        Err(error) => {
            set_string(error_out, error.to_string());
            DagMlDataStatusCode::ValidationError
        }
    }
}

/// Builds an Arrow C Data target table from a coordinator envelope, data view
/// and sample-level target table.
///
/// The request JSON shape is `{ envelope, materialization_request, view,
/// target_table, owner_controller? }`. The returned table has `sample_id`,
/// `target_id` and numeric `value` columns. Repeated observations in the view
/// are de-duplicated to one target row per sample.
///
/// # Safety
///
/// When `json_ptr` is non-null it must point to `json_len` readable bytes for
/// the duration of the call. `out_arrow_array`, `out_arrow_schema` and
/// `error_out` may be null. Returned Arrow pointers must be released with
/// `dagmldata_arrow_array_free` and `dagmldata_arrow_schema_free`.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_coordinator_target_arrow_json(
    json_ptr: *const u8,
    json_len: usize,
    out_arrow_array: *mut *mut ArrowArray,
    out_arrow_schema: *mut *mut ArrowSchema,
    error_out: *mut DagMlDataString,
) -> DagMlDataStatusCode {
    clear_arrow_array(out_arrow_array);
    clear_arrow_schema(out_arrow_schema);
    clear_string(error_out);
    if json_ptr.is_null() {
        set_string(error_out, "json pointer is null");
        return DagMlDataStatusCode::InvalidArgument;
    }
    if out_arrow_array.is_null() || out_arrow_schema.is_null() {
        set_string(error_out, "arrow output pointer is null");
        return DagMlDataStatusCode::InvalidArgument;
    }

    let json = slice::from_raw_parts(json_ptr, json_len);
    match serde_json::from_slice::<CoordinatorTargetArrowRequest>(json) {
        Ok(request) => {
            match build_target_block(&request).and_then(|block| build_target_arrow(&block)) {
                Ok((array, schema)) => {
                    *out_arrow_array = Box::into_raw(Box::new(array));
                    *out_arrow_schema = Box::into_raw(Box::new(schema));
                    DagMlDataStatusCode::Ok
                }
                Err(error) => {
                    set_string(error_out, error.to_string());
                    DagMlDataStatusCode::ValidationError
                }
            }
        }
        Err(error) => {
            set_string(error_out, error.to_string());
            DagMlDataStatusCode::ValidationError
        }
    }
}

unsafe fn clear_string(out: *mut DagMlDataString) {
    if !out.is_null() {
        *out = DagMlDataString::default();
    }
}

unsafe fn clear_arrow_array(out: *mut *mut ArrowArray) {
    if !out.is_null() {
        *out = std::ptr::null_mut();
    }
}

unsafe fn clear_arrow_schema(out: *mut *mut ArrowSchema) {
    if !out.is_null() {
        *out = std::ptr::null_mut();
    }
}

unsafe fn set_string(out: *mut DagMlDataString, value: impl Into<String>) {
    if out.is_null() {
        return;
    }
    let sanitized = value.into().replace('\0', "\\0");
    let c_string = CString::new(sanitized).expect("nul bytes were sanitized");
    let len = c_string.as_bytes().len();
    *out = DagMlDataString {
        ptr: c_string.into_raw(),
        len,
    };
}

#[derive(Debug, Deserialize)]
struct CoordinatorTargetArrowRequest {
    envelope: CoordinatorDataPlanEnvelope,
    materialization_request: CoordinatorDataMaterializationRequest,
    view: DataView,
    target_table: CoordinatorTargetTable,
    #[serde(default = "default_owner_controller")]
    owner_controller: String,
}

fn default_owner_controller() -> String {
    "controller:data.provider".to_string()
}

#[allow(dead_code)]
struct StringArrayPrivate {
    validity: Option<Vec<u8>>,
    offsets: Vec<i32>,
    values: Vec<u8>,
    buffers: Box<[*const c_void]>,
}

#[allow(dead_code)]
struct BoolArrayPrivate {
    values: Vec<u8>,
    buffers: Box<[*const c_void]>,
}

#[allow(dead_code)]
struct F64ArrayPrivate {
    validity: Option<Vec<u8>>,
    values: Vec<f64>,
    buffers: Box<[*const c_void]>,
}

struct StructArrayPrivate {
    children: Box<[*mut ArrowArray]>,
    buffers: Box<[*const c_void]>,
}

#[allow(dead_code)]
struct SchemaPrivate {
    format: CString,
    name: CString,
    metadata: Option<CString>,
    children: Box<[*mut ArrowSchema]>,
}

fn build_identity_arrow(
    envelope: &CoordinatorDataPlanEnvelope,
) -> dag_ml_data_core::Result<(ArrowArray, ArrowSchema)> {
    envelope.validate()?;
    let relations = envelope.coordinator_relations.as_ref().ok_or_else(|| {
        dag_ml_data_core::DataError::Validation(
            "coordinator identity export requires coordinator_relations".to_string(),
        )
    })?;
    let records = &relations.records;
    let child_arrays = vec![
        Box::into_raw(Box::new(string_array(
            records
                .iter()
                .map(|record| Some(record.observation_id.as_str())),
        )?)),
        Box::into_raw(Box::new(string_array(
            records.iter().map(|record| Some(record.sample_id.as_str())),
        )?)),
        Box::into_raw(Box::new(string_array(records.iter().map(|record| {
            record.target_id.as_ref().map(|value| value.as_str())
        }))?)),
        Box::into_raw(Box::new(string_array(
            records
                .iter()
                .map(|record| record.group_id.as_ref().map(|value| value.as_str())),
        )?)),
        Box::into_raw(Box::new(string_array(records.iter().map(|record| {
            record.origin_sample_id.as_ref().map(|value| value.as_str())
        }))?)),
        Box::into_raw(Box::new(string_array(records.iter().map(|record| {
            record.source_id.as_ref().map(|value| value.as_str())
        }))?)),
        Box::into_raw(Box::new(bool_array(
            records.iter().map(|record| record.is_augmented),
        ))),
    ];
    let child_schemas = vec![
        Box::into_raw(Box::new(field_schema("observation_id", "u", false)?)),
        Box::into_raw(Box::new(field_schema("sample_id", "u", false)?)),
        Box::into_raw(Box::new(field_schema("target_id", "u", true)?)),
        Box::into_raw(Box::new(field_schema("group_id", "u", true)?)),
        Box::into_raw(Box::new(field_schema("origin_sample_id", "u", true)?)),
        Box::into_raw(Box::new(field_schema("source_id", "u", true)?)),
        Box::into_raw(Box::new(field_schema("is_augmented", "b", false)?)),
    ];
    Ok((
        struct_array(records.len(), child_arrays),
        struct_schema("coordinator_identity", child_schemas)?,
    ))
}

fn build_target_block(
    request: &CoordinatorTargetArrowRequest,
) -> dag_ml_data_core::Result<CoordinatorTargetBlock> {
    let arena = CoordinatorHandleArena::new(&request.owner_controller)?;
    let data = arena.materialize(&request.envelope, &request.materialization_request)?;
    let view = arena.make_view(data.handle.handle, &request.view)?;
    arena.target_values(view.handle.handle, &request.target_table)
}

fn build_target_arrow(
    target: &CoordinatorTargetBlock,
) -> dag_ml_data_core::Result<(ArrowArray, ArrowSchema)> {
    let target_ids = std::iter::repeat_n(Some(target.target_id.as_str()), target.sample_ids.len());
    let numeric_values = target
        .values
        .iter()
        .map(|value| match value {
            serde_json::Value::Null => Ok(None),
            serde_json::Value::Number(number) => number.as_f64().map(Some).ok_or_else(|| {
                dag_ml_data_core::DataError::Validation(format!(
                    "target `{}` contains a non-f64 numeric value",
                    target.target_id
                ))
            }),
            _ => Err(dag_ml_data_core::DataError::Validation(format!(
                "target `{}` Arrow smoke only supports numeric or null values",
                target.target_id
            ))),
        })
        .collect::<dag_ml_data_core::Result<Vec<_>>>()?;
    let child_arrays = vec![
        Box::into_raw(Box::new(string_array(
            target
                .sample_ids
                .iter()
                .map(|sample_id| Some(sample_id.as_str())),
        )?)),
        Box::into_raw(Box::new(string_array(target_ids)?)),
        Box::into_raw(Box::new(f64_array(numeric_values.into_iter()))),
    ];
    let child_schemas = vec![
        Box::into_raw(Box::new(field_schema("sample_id", "u", false)?)),
        Box::into_raw(Box::new(field_schema("target_id", "u", false)?)),
        Box::into_raw(Box::new(field_schema("value", "g", true)?)),
    ];
    Ok((
        struct_array(target.sample_ids.len(), child_arrays),
        struct_schema("coordinator_target", child_schemas)?,
    ))
}

fn string_array<'a>(
    values: impl Iterator<Item = Option<&'a str>>,
) -> dag_ml_data_core::Result<ArrowArray> {
    let values = values.collect::<Vec<_>>();
    let mut validity = Vec::new();
    let mut offsets = Vec::with_capacity(values.len() + 1);
    let mut data = Vec::new();
    offsets.push(0);
    let mut null_count = 0i64;
    for (idx, value) in values.iter().enumerate() {
        if let Some(value) = value {
            set_bitmap(&mut validity, idx, true);
            data.extend_from_slice(value.as_bytes());
        } else {
            set_bitmap(&mut validity, idx, false);
            null_count += 1;
        }
        let offset = i32::try_from(data.len()).map_err(|_| {
            dag_ml_data_core::DataError::Validation(
                "identity Arrow UTF-8 payload exceeds i32 offsets".to_string(),
            )
        })?;
        offsets.push(offset);
    }
    let validity = (null_count > 0).then_some(validity);
    let buffers = vec![
        validity
            .as_ref()
            .map(|buffer| buffer.as_ptr().cast::<c_void>())
            .unwrap_or(std::ptr::null()),
        offsets.as_ptr().cast::<c_void>(),
        data.as_ptr().cast::<c_void>(),
    ]
    .into_boxed_slice();
    let private = Box::new(StringArrayPrivate {
        validity,
        offsets,
        values: data,
        buffers,
    });
    let buffers = private.buffers.as_ptr() as *mut *const c_void;
    Ok(ArrowArray {
        length: values.len() as i64,
        null_count,
        offset: 0,
        n_buffers: 3,
        n_children: 0,
        buffers,
        children: std::ptr::null_mut(),
        dictionary: std::ptr::null_mut(),
        release: Some(release_string_array),
        private_data: Box::into_raw(private).cast::<c_void>(),
    })
}

fn f64_array(values: impl Iterator<Item = Option<f64>>) -> ArrowArray {
    let values = values.collect::<Vec<_>>();
    let mut validity = Vec::new();
    let mut data = Vec::with_capacity(values.len());
    let mut null_count = 0i64;
    for (idx, value) in values.iter().enumerate() {
        if let Some(value) = value {
            set_bitmap(&mut validity, idx, true);
            data.push(*value);
        } else {
            set_bitmap(&mut validity, idx, false);
            data.push(0.0);
            null_count += 1;
        }
    }
    let validity = (null_count > 0).then_some(validity);
    let buffers = vec![
        validity
            .as_ref()
            .map(|buffer| buffer.as_ptr().cast::<c_void>())
            .unwrap_or(std::ptr::null()),
        data.as_ptr().cast::<c_void>(),
    ]
    .into_boxed_slice();
    let private = Box::new(F64ArrayPrivate {
        validity,
        values: data,
        buffers,
    });
    let buffers = private.buffers.as_ptr() as *mut *const c_void;
    ArrowArray {
        length: values.len() as i64,
        null_count,
        offset: 0,
        n_buffers: 2,
        n_children: 0,
        buffers,
        children: std::ptr::null_mut(),
        dictionary: std::ptr::null_mut(),
        release: Some(release_f64_array),
        private_data: Box::into_raw(private).cast::<c_void>(),
    }
}

fn bool_array(values: impl Iterator<Item = bool>) -> ArrowArray {
    let values = values.collect::<Vec<_>>();
    let mut bitmap = Vec::new();
    for (idx, value) in values.iter().enumerate() {
        set_bitmap(&mut bitmap, idx, *value);
    }
    let buffers = vec![std::ptr::null(), bitmap.as_ptr().cast::<c_void>()].into_boxed_slice();
    let private = Box::new(BoolArrayPrivate {
        values: bitmap,
        buffers,
    });
    let buffers = private.buffers.as_ptr() as *mut *const c_void;
    ArrowArray {
        length: values.len() as i64,
        null_count: 0,
        offset: 0,
        n_buffers: 2,
        n_children: 0,
        buffers,
        children: std::ptr::null_mut(),
        dictionary: std::ptr::null_mut(),
        release: Some(release_bool_array),
        private_data: Box::into_raw(private).cast::<c_void>(),
    }
}

fn struct_array(length: usize, children: Vec<*mut ArrowArray>) -> ArrowArray {
    let child_count = children.len() as i64;
    let children = children.into_boxed_slice();
    let buffers = vec![std::ptr::null()].into_boxed_slice();
    let private = Box::new(StructArrayPrivate { children, buffers });
    let children = private.children.as_ptr() as *mut *mut ArrowArray;
    let buffers = private.buffers.as_ptr() as *mut *const c_void;
    ArrowArray {
        length: length as i64,
        null_count: 0,
        offset: 0,
        n_buffers: 1,
        n_children: child_count,
        buffers,
        children,
        dictionary: std::ptr::null_mut(),
        release: Some(release_struct_array),
        private_data: Box::into_raw(private).cast::<c_void>(),
    }
}

fn field_schema(name: &str, format: &str, nullable: bool) -> dag_ml_data_core::Result<ArrowSchema> {
    schema(name, format, nullable, Vec::new())
}

fn struct_schema(
    name: &str,
    children: Vec<*mut ArrowSchema>,
) -> dag_ml_data_core::Result<ArrowSchema> {
    schema(name, "+s", false, children)
}

fn schema(
    name: &str,
    format: &str,
    nullable: bool,
    children: Vec<*mut ArrowSchema>,
) -> dag_ml_data_core::Result<ArrowSchema> {
    let format = CString::new(format).map_err(|_| {
        dag_ml_data_core::DataError::Validation("Arrow schema format contains nul".to_string())
    })?;
    let name = CString::new(name).map_err(|_| {
        dag_ml_data_core::DataError::Validation("Arrow schema name contains nul".to_string())
    })?;
    let child_count = children.len() as i64;
    let private = Box::new(SchemaPrivate {
        format,
        name,
        metadata: None,
        children: children.into_boxed_slice(),
    });
    let schema = ArrowSchema {
        format: private.format.as_ptr(),
        name: private.name.as_ptr(),
        metadata: std::ptr::null(),
        flags: if nullable { 1 } else { 0 },
        n_children: child_count,
        children: private.children.as_ptr() as *mut *mut ArrowSchema,
        dictionary: std::ptr::null_mut(),
        release: Some(release_schema),
        private_data: Box::into_raw(private).cast::<c_void>(),
    };
    Ok(schema)
}

fn set_bitmap(bitmap: &mut Vec<u8>, idx: usize, value: bool) {
    let byte_idx = idx / 8;
    if bitmap.len() <= byte_idx {
        bitmap.resize(byte_idx + 1, 0);
    }
    if value {
        bitmap[byte_idx] |= 1 << (idx % 8);
    }
}

unsafe extern "C" fn release_string_array(array: *mut ArrowArray) {
    if array.is_null() || (*array).release.is_none() {
        return;
    }
    (*array).release = None;
    if !(*array).private_data.is_null() {
        let private = Box::from_raw((*array).private_data.cast::<StringArrayPrivate>());
        drop(private);
    }
    (*array).private_data = std::ptr::null_mut();
    (*array).buffers = std::ptr::null_mut();
}

unsafe extern "C" fn release_bool_array(array: *mut ArrowArray) {
    if array.is_null() || (*array).release.is_none() {
        return;
    }
    (*array).release = None;
    if !(*array).private_data.is_null() {
        let private = Box::from_raw((*array).private_data.cast::<BoolArrayPrivate>());
        drop(private);
    }
    (*array).private_data = std::ptr::null_mut();
    (*array).buffers = std::ptr::null_mut();
}

unsafe extern "C" fn release_f64_array(array: *mut ArrowArray) {
    if array.is_null() || (*array).release.is_none() {
        return;
    }
    (*array).release = None;
    if !(*array).private_data.is_null() {
        let private = Box::from_raw((*array).private_data.cast::<F64ArrayPrivate>());
        drop(private);
    }
    (*array).private_data = std::ptr::null_mut();
    (*array).buffers = std::ptr::null_mut();
}

unsafe extern "C" fn release_struct_array(array: *mut ArrowArray) {
    if array.is_null() || (*array).release.is_none() {
        return;
    }
    (*array).release = None;
    if !(*array).private_data.is_null() {
        let private = Box::from_raw((*array).private_data.cast::<StructArrayPrivate>());
        for child in private.children.iter().copied() {
            if !child.is_null() {
                if let Some(release) = (*child).release {
                    release(child);
                }
                drop(Box::from_raw(child));
            }
        }
        drop(private);
    }
    (*array).private_data = std::ptr::null_mut();
    (*array).buffers = std::ptr::null_mut();
    (*array).children = std::ptr::null_mut();
}

unsafe extern "C" fn release_schema(schema: *mut ArrowSchema) {
    if schema.is_null() || (*schema).release.is_none() {
        return;
    }
    (*schema).release = None;
    if !(*schema).private_data.is_null() {
        let private = Box::from_raw((*schema).private_data.cast::<SchemaPrivate>());
        for child in private.children.iter().copied() {
            if !child.is_null() {
                if let Some(release) = (*child).release {
                    release(child);
                }
                drop(Box::from_raw(child));
            }
        }
        drop(private);
    }
    (*schema).private_data = std::ptr::null_mut();
    (*schema).format = std::ptr::null();
    (*schema).name = std::ptr::null();
    (*schema).metadata = std::ptr::null();
    (*schema).children = std::ptr::null_mut();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn fingerprints_schema_json_over_abi() {
        let schema = include_bytes!("../../../examples/minimal_schema.json");
        let mut fingerprint = DagMlDataString::default();
        let mut error = DagMlDataString::default();

        let status = unsafe {
            dagmldata_schema_fingerprint_json(
                schema.as_ptr(),
                schema.len(),
                &mut fingerprint,
                &mut error,
            )
        };

        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert!(!fingerprint.ptr.is_null());
        assert!(error.ptr.is_null());

        unsafe {
            dagmldata_string_free(fingerprint);
        }
    }

    #[test]
    fn exports_coordinator_identity_arrow_over_abi() {
        let envelope = include_bytes!(
            "../../../examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json"
        );
        let mut array = std::ptr::null_mut();
        let mut schema = std::ptr::null_mut();
        let mut error = DagMlDataString::default();

        let status = unsafe {
            dagmldata_coordinator_identity_arrow_json(
                envelope.as_ptr(),
                envelope.len(),
                &mut array,
                &mut schema,
                &mut error,
            )
        };

        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert!(error.ptr.is_null());
        assert!(!array.is_null());
        assert!(!schema.is_null());
        unsafe {
            assert_eq!((*array).length, 4);
            assert_eq!((*array).n_children, 7);
            assert_eq!((*schema).n_children, 7);
            assert_eq!(CStr::from_ptr((*schema).format).to_str().unwrap(), "+s");
            let schema_children =
                slice::from_raw_parts((*schema).children, (*schema).n_children as usize);
            let array_children =
                slice::from_raw_parts((*array).children, (*array).n_children as usize);
            let first_child = schema_children[0];
            assert_eq!(
                CStr::from_ptr((*first_child).name).to_str().unwrap(),
                "observation_id"
            );
            assert_eq!(CStr::from_ptr((*first_child).format).to_str().unwrap(), "u");
            assert_eq!(
                utf8_values(array_children[0]),
                vec![
                    Some("obs.S001.aug0".to_string()),
                    Some("obs.S001.base".to_string()),
                    Some("obs.S001.rep1".to_string()),
                    Some("obs.S002.base".to_string()),
                ]
            );
            assert_eq!(
                utf8_values(array_children[4]),
                vec![Some("S001".to_string()), None, None, None]
            );
            assert_eq!(
                bool_values(array_children[6]),
                vec![true, false, false, false]
            );
            dagmldata_arrow_array_free(array);
            dagmldata_arrow_schema_free(schema);
        }
    }

    #[test]
    fn exports_coordinator_target_arrow_over_abi() {
        let envelope: serde_json::Value = serde_json::from_slice(include_bytes!(
            "../../../examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json"
        ))
        .unwrap();
        let materialization_request: serde_json::Value = serde_json::from_slice(include_bytes!(
            "../../../examples/fixtures/oof_campaign/materialization_request_model_base_x.json"
        ))
        .unwrap();
        let request = serde_json::json!({
            "envelope": envelope,
            "materialization_request": materialization_request,
            "view": {
                "sample_ids": ["S001"],
                "include_augmented": false
            },
            "target_table": {
                "target_id": "y",
                "values": [
                    {"sample_id": "S001", "value": 42.0},
                    {"sample_id": "S002", "value": 7.0}
                ]
            }
        });
        let request = serde_json::to_vec(&request).unwrap();
        let mut array = std::ptr::null_mut();
        let mut schema = std::ptr::null_mut();
        let mut error = DagMlDataString::default();

        let status = unsafe {
            dagmldata_coordinator_target_arrow_json(
                request.as_ptr(),
                request.len(),
                &mut array,
                &mut schema,
                &mut error,
            )
        };

        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert!(error.ptr.is_null());
        unsafe {
            assert_eq!((*array).length, 1);
            assert_eq!((*array).n_children, 3);
            assert_eq!(CStr::from_ptr((*schema).format).to_str().unwrap(), "+s");
            let array_children =
                slice::from_raw_parts((*array).children, (*array).n_children as usize);
            assert_eq!(
                utf8_values(array_children[0]),
                vec![Some("S001".to_string())]
            );
            assert_eq!(f64_values(array_children[2]), vec![Some(42.0)]);
            dagmldata_arrow_array_free(array);
            dagmldata_arrow_schema_free(schema);
        }
    }

    unsafe fn utf8_values(array: *const ArrowArray) -> Vec<Option<String>> {
        assert!(!array.is_null());
        let buffers = slice::from_raw_parts((*array).buffers, (*array).n_buffers as usize);
        assert_eq!(buffers.len(), 3);
        let offsets = slice::from_raw_parts(
            buffers[1].cast::<i32>(),
            usize::try_from((*array).length).unwrap() + 1,
        );
        let values = slice::from_raw_parts(
            buffers[2].cast::<u8>(),
            usize::try_from(*offsets.last().unwrap()).unwrap(),
        );
        (0..usize::try_from((*array).length).unwrap())
            .map(|idx| {
                if !is_valid(buffers[0], idx) {
                    return None;
                }
                let start = usize::try_from(offsets[idx]).unwrap();
                let end = usize::try_from(offsets[idx + 1]).unwrap();
                Some(String::from_utf8(values[start..end].to_vec()).unwrap())
            })
            .collect()
    }

    unsafe fn f64_values(array: *const ArrowArray) -> Vec<Option<f64>> {
        assert!(!array.is_null());
        let buffers = slice::from_raw_parts((*array).buffers, (*array).n_buffers as usize);
        assert_eq!(buffers.len(), 2);
        let values = slice::from_raw_parts(buffers[1].cast::<f64>(), (*array).length as usize);
        (0..usize::try_from((*array).length).unwrap())
            .map(|idx| is_valid(buffers[0], idx).then_some(values[idx]))
            .collect()
    }

    unsafe fn bool_values(array: *const ArrowArray) -> Vec<bool> {
        assert!(!array.is_null());
        let buffers = slice::from_raw_parts((*array).buffers, (*array).n_buffers as usize);
        assert_eq!(buffers.len(), 2);
        (0..usize::try_from((*array).length).unwrap())
            .map(|idx| is_valid(buffers[1], idx))
            .collect()
    }

    unsafe fn is_valid(bitmap: *const c_void, idx: usize) -> bool {
        if bitmap.is_null() {
            return true;
        }
        let byte = *bitmap.cast::<u8>().add(idx / 8);
        byte & (1 << (idx % 8)) != 0
    }
}
