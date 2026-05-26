use std::collections::BTreeMap;
use std::ffi::{c_void, CString};
use std::os::raw::c_char;
use std::slice;

use dag_ml_data_core::{
    schema_fingerprint, CoordinatorDataMaterializationRequest, CoordinatorDataPlanEnvelope,
    CoordinatorFeatureBlock, CoordinatorFeatureTable, CoordinatorHandleArena,
    CoordinatorTargetBlock, CoordinatorTargetTable, DataView, DatasetSchema, TargetId,
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
    pub feature_arrow: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            view: DagMlDataHandle,
            feature_set_name: DagMlDataBytesView,
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

/// Builds an Arrow C Data feature table from a coordinator envelope, data view
/// and observation-level feature table.
///
/// The request JSON shape is `{ envelope, materialization_request, view,
/// feature_table, owner_controller? }`. The returned table has
/// `observation_id`, `sample_id` and one numeric column per selected feature.
/// Repeated observations are preserved; `DataView.columns` filters feature
/// columns without changing row identity.
///
/// # Safety
///
/// When `json_ptr` is non-null it must point to `json_len` readable bytes for
/// the duration of the call. `out_arrow_array`, `out_arrow_schema` and
/// `error_out` may be null. Returned Arrow pointers must be released with
/// `dagmldata_arrow_array_free` and `dagmldata_arrow_schema_free`.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_coordinator_feature_arrow_json(
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
    match serde_json::from_slice::<CoordinatorFeatureArrowRequest>(json) {
        Ok(request) => {
            match build_feature_block(&request).and_then(|block| build_feature_arrow(&block)) {
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

/// Creates a Rust-owned in-memory provider and returns its C ABI vtable.
///
/// `envelope_ptr/envelope_len` must encode a `CoordinatorDataPlanEnvelope`.
/// `target_tables_ptr/target_tables_len` may be null/zero, or a JSON array of
/// `CoordinatorTargetTable` values. The caller owns the returned vtable value
/// but must eventually call either `vtable.destroy(vtable.user_data)` or
/// `dagmldata_inmemory_provider_destroy(&vtable)`.
///
/// # Safety
///
/// Non-null byte pointers must point to readable memory for the duration of the
/// call. `out_vtable` may be null only if the caller is probing error handling.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_inmemory_provider_new_json(
    envelope_ptr: *const u8,
    envelope_len: usize,
    target_tables_ptr: *const u8,
    target_tables_len: usize,
    out_vtable: *mut DagMlDataVTable,
    error_out: *mut DagMlDataString,
) -> DagMlDataStatusCode {
    clear_vtable(out_vtable);
    clear_string(error_out);
    if envelope_ptr.is_null() {
        set_string(error_out, "envelope pointer is null");
        return DagMlDataStatusCode::InvalidArgument;
    }
    if out_vtable.is_null() {
        set_string(error_out, "vtable output pointer is null");
        return DagMlDataStatusCode::InvalidArgument;
    }

    let envelope_json = slice::from_raw_parts(envelope_ptr, envelope_len);
    let envelope = match serde_json::from_slice::<CoordinatorDataPlanEnvelope>(envelope_json) {
        Ok(envelope) => envelope,
        Err(error) => {
            set_string(error_out, error.to_string());
            return DagMlDataStatusCode::ValidationError;
        }
    };
    let target_tables = match parse_target_tables(target_tables_ptr, target_tables_len) {
        Ok(target_tables) => target_tables,
        Err(error) => {
            set_string(error_out, error.to_string());
            return DagMlDataStatusCode::ValidationError;
        }
    };
    match InMemoryProvider::new(envelope, target_tables, BTreeMap::new()) {
        Ok(provider) => {
            *out_vtable = provider_vtable(Box::into_raw(Box::new(provider)).cast::<c_void>());
            DagMlDataStatusCode::Ok
        }
        Err(error) => {
            set_string(error_out, error.to_string());
            DagMlDataStatusCode::ValidationError
        }
    }
}

/// Creates a Rust-owned in-memory provider with target and feature tables.
///
/// `feature_tables_ptr/feature_tables_len` may be null/zero, or a JSON array of
/// `CoordinatorFeatureTable` values. This is the current conformance helper for
/// binding tests that need real observation-level feature data.
///
/// # Safety
///
/// Non-null byte pointers must point to readable memory for the duration of the
/// call. `out_vtable` may be null only if the caller is probing error handling.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_inmemory_provider_new_with_features_json(
    envelope_ptr: *const u8,
    envelope_len: usize,
    target_tables_ptr: *const u8,
    target_tables_len: usize,
    feature_tables_ptr: *const u8,
    feature_tables_len: usize,
    out_vtable: *mut DagMlDataVTable,
    error_out: *mut DagMlDataString,
) -> DagMlDataStatusCode {
    clear_vtable(out_vtable);
    clear_string(error_out);
    if envelope_ptr.is_null() {
        set_string(error_out, "envelope pointer is null");
        return DagMlDataStatusCode::InvalidArgument;
    }
    if out_vtable.is_null() {
        set_string(error_out, "vtable output pointer is null");
        return DagMlDataStatusCode::InvalidArgument;
    }

    let envelope_json = slice::from_raw_parts(envelope_ptr, envelope_len);
    let envelope = match serde_json::from_slice::<CoordinatorDataPlanEnvelope>(envelope_json) {
        Ok(envelope) => envelope,
        Err(error) => {
            set_string(error_out, error.to_string());
            return DagMlDataStatusCode::ValidationError;
        }
    };
    let target_tables = match parse_target_tables(target_tables_ptr, target_tables_len) {
        Ok(target_tables) => target_tables,
        Err(error) => {
            set_string(error_out, error.to_string());
            return DagMlDataStatusCode::ValidationError;
        }
    };
    let feature_tables = match parse_feature_tables(feature_tables_ptr, feature_tables_len) {
        Ok(feature_tables) => feature_tables,
        Err(error) => {
            set_string(error_out, error.to_string());
            return DagMlDataStatusCode::ValidationError;
        }
    };
    match InMemoryProvider::new(envelope, target_tables, feature_tables) {
        Ok(provider) => {
            *out_vtable = provider_vtable(Box::into_raw(Box::new(provider)).cast::<c_void>());
            DagMlDataStatusCode::Ok
        }
        Err(error) => {
            set_string(error_out, error.to_string());
            DagMlDataStatusCode::ValidationError
        }
    }
}

/// Destroys a provider vtable returned by `dagmldata_inmemory_provider_new_json`.
///
/// # Safety
///
/// `vtable` must be null or point to a vtable previously initialized by
/// `dagmldata_inmemory_provider_new_json` and not already destroyed.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_inmemory_provider_destroy(vtable: *mut DagMlDataVTable) {
    if vtable.is_null() {
        return;
    }
    if let Some(destroy) = (*vtable).destroy {
        destroy((*vtable).user_data);
    }
    *vtable = empty_vtable();
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

unsafe fn clear_vtable(out: *mut DagMlDataVTable) {
    if !out.is_null() {
        *out = empty_vtable();
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

#[derive(Debug, Deserialize)]
struct CoordinatorFeatureArrowRequest {
    envelope: CoordinatorDataPlanEnvelope,
    materialization_request: CoordinatorDataMaterializationRequest,
    view: DataView,
    feature_table: CoordinatorFeatureTable,
    #[serde(default = "default_owner_controller")]
    owner_controller: String,
}

fn default_owner_controller() -> String {
    "controller:data.provider".to_string()
}

struct InMemoryProvider {
    arena: CoordinatorHandleArena,
    envelope: CoordinatorDataPlanEnvelope,
    target_tables: BTreeMap<TargetId, CoordinatorTargetTable>,
    feature_tables: BTreeMap<String, CoordinatorFeatureTable>,
}

impl InMemoryProvider {
    fn new(
        envelope: CoordinatorDataPlanEnvelope,
        target_tables: BTreeMap<TargetId, CoordinatorTargetTable>,
        feature_tables: BTreeMap<String, CoordinatorFeatureTable>,
    ) -> dag_ml_data_core::Result<Self> {
        envelope.validate()?;
        Ok(Self {
            arena: CoordinatorHandleArena::new(default_owner_controller())?,
            envelope,
            target_tables,
            feature_tables,
        })
    }
}

fn parse_target_tables(
    target_tables_ptr: *const u8,
    target_tables_len: usize,
) -> dag_ml_data_core::Result<BTreeMap<TargetId, CoordinatorTargetTable>> {
    if target_tables_ptr.is_null() {
        if target_tables_len != 0 {
            return Err(dag_ml_data_core::DataError::Validation(
                "target tables pointer is null".to_string(),
            ));
        }
        return Ok(BTreeMap::new());
    }
    if target_tables_len == 0 {
        return Ok(BTreeMap::new());
    }
    let json = unsafe { slice::from_raw_parts(target_tables_ptr, target_tables_len) };
    let tables = serde_json::from_slice::<Vec<CoordinatorTargetTable>>(json).map_err(|error| {
        dag_ml_data_core::DataError::Validation(format!(
            "failed to parse target tables JSON: {error}"
        ))
    })?;
    let mut by_target = BTreeMap::new();
    for table in tables {
        table.validate()?;
        let target_id = table.target_id.clone();
        if by_target.insert(target_id.clone(), table).is_some() {
            return Err(dag_ml_data_core::DataError::Validation(format!(
                "duplicate target table `{target_id}`"
            )));
        }
    }
    Ok(by_target)
}

fn parse_feature_tables(
    feature_tables_ptr: *const u8,
    feature_tables_len: usize,
) -> dag_ml_data_core::Result<BTreeMap<String, CoordinatorFeatureTable>> {
    if feature_tables_ptr.is_null() {
        if feature_tables_len != 0 {
            return Err(dag_ml_data_core::DataError::Validation(
                "feature tables pointer is null".to_string(),
            ));
        }
        return Ok(BTreeMap::new());
    }
    if feature_tables_len == 0 {
        return Ok(BTreeMap::new());
    }
    let json = unsafe { slice::from_raw_parts(feature_tables_ptr, feature_tables_len) };
    let tables = serde_json::from_slice::<Vec<CoordinatorFeatureTable>>(json).map_err(|error| {
        dag_ml_data_core::DataError::Validation(format!(
            "failed to parse feature tables JSON: {error}"
        ))
    })?;
    let mut by_feature_set = BTreeMap::new();
    for table in tables {
        table.validate()?;
        let feature_set_id = table.feature_set_id.clone();
        if by_feature_set
            .insert(feature_set_id.clone(), table)
            .is_some()
        {
            return Err(dag_ml_data_core::DataError::Validation(format!(
                "duplicate feature table `{feature_set_id}`"
            )));
        }
    }
    Ok(by_feature_set)
}

fn provider_vtable(user_data: *mut c_void) -> DagMlDataVTable {
    DagMlDataVTable {
        abi_version: 2,
        user_data,
        materialize: Some(provider_materialize),
        make_view: Some(provider_make_view),
        view_identity: Some(provider_view_identity),
        target_arrow: Some(provider_target_arrow),
        feature_arrow: Some(provider_feature_arrow),
        release: Some(provider_release),
        destroy: Some(provider_destroy),
    }
}

fn empty_vtable() -> DagMlDataVTable {
    DagMlDataVTable {
        abi_version: 2,
        user_data: std::ptr::null_mut(),
        materialize: None,
        make_view: None,
        view_identity: None,
        target_arrow: None,
        feature_arrow: None,
        release: None,
        destroy: None,
    }
}

unsafe extern "C" fn provider_materialize(
    user_data: *mut c_void,
    _dataset: DagMlDataHandle,
    request_json: DagMlDataBytesView,
    out_handle: *mut DagMlDataHandle,
) -> DagMlDataStatusCode {
    if user_data.is_null() || out_handle.is_null() || request_json.ptr.is_null() {
        return DagMlDataStatusCode::InvalidArgument;
    }
    *out_handle = 0;
    let provider = &*(user_data.cast::<InMemoryProvider>());
    let request = match serde_json::from_slice::<CoordinatorDataMaterializationRequest>(
        slice::from_raw_parts(request_json.ptr, request_json.len),
    ) {
        Ok(request) => request,
        Err(_) => return DagMlDataStatusCode::ValidationError,
    };
    match provider.arena.materialize(&provider.envelope, &request) {
        Ok(record) => {
            *out_handle = record.handle.handle;
            DagMlDataStatusCode::Ok
        }
        Err(_) => DagMlDataStatusCode::ValidationError,
    }
}

unsafe extern "C" fn provider_make_view(
    user_data: *mut c_void,
    data: DagMlDataHandle,
    selector_json: DagMlDataBytesView,
    out_view: *mut DagMlDataHandle,
) -> DagMlDataStatusCode {
    if user_data.is_null() || out_view.is_null() || selector_json.ptr.is_null() {
        return DagMlDataStatusCode::InvalidArgument;
    }
    *out_view = 0;
    let provider = &*(user_data.cast::<InMemoryProvider>());
    let view = match serde_json::from_slice::<DataView>(slice::from_raw_parts(
        selector_json.ptr,
        selector_json.len,
    )) {
        Ok(view) => view,
        Err(_) => return DagMlDataStatusCode::ValidationError,
    };
    match provider.arena.make_view(data, &view) {
        Ok(record) => {
            *out_view = record.handle.handle;
            DagMlDataStatusCode::Ok
        }
        Err(_) => DagMlDataStatusCode::ValidationError,
    }
}

unsafe extern "C" fn provider_view_identity(
    user_data: *mut c_void,
    view: DagMlDataHandle,
    out_arrow_array: *mut *mut ArrowArray,
    out_arrow_schema: *mut *mut ArrowSchema,
) -> DagMlDataStatusCode {
    clear_arrow_array(out_arrow_array);
    clear_arrow_schema(out_arrow_schema);
    if user_data.is_null() || out_arrow_array.is_null() || out_arrow_schema.is_null() {
        return DagMlDataStatusCode::InvalidArgument;
    }
    let provider = &*(user_data.cast::<InMemoryProvider>());
    match provider
        .arena
        .view_identity(view)
        .and_then(|relations| build_identity_relations_arrow(&relations))
    {
        Ok((array, schema)) => {
            *out_arrow_array = Box::into_raw(Box::new(array));
            *out_arrow_schema = Box::into_raw(Box::new(schema));
            DagMlDataStatusCode::Ok
        }
        Err(_) => DagMlDataStatusCode::ValidationError,
    }
}

unsafe extern "C" fn provider_target_arrow(
    user_data: *mut c_void,
    view: DagMlDataHandle,
    target_name: DagMlDataBytesView,
    out_arrow_array: *mut *mut ArrowArray,
    out_arrow_schema: *mut *mut ArrowSchema,
) -> DagMlDataStatusCode {
    clear_arrow_array(out_arrow_array);
    clear_arrow_schema(out_arrow_schema);
    if user_data.is_null()
        || target_name.ptr.is_null()
        || out_arrow_array.is_null()
        || out_arrow_schema.is_null()
    {
        return DagMlDataStatusCode::InvalidArgument;
    }
    let provider = &*(user_data.cast::<InMemoryProvider>());
    let target_name =
        match std::str::from_utf8(slice::from_raw_parts(target_name.ptr, target_name.len)) {
            Ok(target_name) => target_name,
            Err(_) => return DagMlDataStatusCode::ValidationError,
        };
    let target_id = match TargetId::new(target_name) {
        Ok(target_id) => target_id,
        Err(_) => return DagMlDataStatusCode::ValidationError,
    };
    let target_table = match provider.target_tables.get(&target_id) {
        Some(target_table) => target_table,
        None => return DagMlDataStatusCode::ValidationError,
    };
    match provider
        .arena
        .target_values(view, target_table)
        .and_then(|target| build_target_arrow(&target))
    {
        Ok((array, schema)) => {
            *out_arrow_array = Box::into_raw(Box::new(array));
            *out_arrow_schema = Box::into_raw(Box::new(schema));
            DagMlDataStatusCode::Ok
        }
        Err(_) => DagMlDataStatusCode::ValidationError,
    }
}

unsafe extern "C" fn provider_feature_arrow(
    user_data: *mut c_void,
    view: DagMlDataHandle,
    feature_set_name: DagMlDataBytesView,
    out_arrow_array: *mut *mut ArrowArray,
    out_arrow_schema: *mut *mut ArrowSchema,
) -> DagMlDataStatusCode {
    clear_arrow_array(out_arrow_array);
    clear_arrow_schema(out_arrow_schema);
    if user_data.is_null()
        || feature_set_name.ptr.is_null()
        || out_arrow_array.is_null()
        || out_arrow_schema.is_null()
    {
        return DagMlDataStatusCode::InvalidArgument;
    }
    let provider = &*(user_data.cast::<InMemoryProvider>());
    let feature_set_name = match std::str::from_utf8(slice::from_raw_parts(
        feature_set_name.ptr,
        feature_set_name.len,
    )) {
        Ok(feature_set_name) if !feature_set_name.trim().is_empty() => feature_set_name,
        _ => return DagMlDataStatusCode::ValidationError,
    };
    let feature_table = match provider.feature_tables.get(feature_set_name) {
        Some(feature_table) => feature_table,
        None => return DagMlDataStatusCode::ValidationError,
    };
    match provider
        .arena
        .feature_values(view, feature_table)
        .and_then(|features| build_feature_arrow(&features))
    {
        Ok((array, schema)) => {
            *out_arrow_array = Box::into_raw(Box::new(array));
            *out_arrow_schema = Box::into_raw(Box::new(schema));
            DagMlDataStatusCode::Ok
        }
        Err(_) => DagMlDataStatusCode::ValidationError,
    }
}

unsafe extern "C" fn provider_release(user_data: *mut c_void, handle: DagMlDataHandle) {
    if user_data.is_null() {
        return;
    }
    let provider = &*(user_data.cast::<InMemoryProvider>());
    provider.arena.release_handle(handle);
}

unsafe extern "C" fn provider_destroy(user_data: *mut c_void) {
    if user_data.is_null() {
        return;
    }
    drop(Box::from_raw(user_data.cast::<InMemoryProvider>()));
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
    build_identity_relations_arrow(relations)
}

fn build_identity_relations_arrow(
    relations: &dag_ml_data_core::CoordinatorRelationSet,
) -> dag_ml_data_core::Result<(ArrowArray, ArrowSchema)> {
    relations.validate()?;
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

fn build_feature_block(
    request: &CoordinatorFeatureArrowRequest,
) -> dag_ml_data_core::Result<CoordinatorFeatureBlock> {
    let arena = CoordinatorHandleArena::new(&request.owner_controller)?;
    let data = arena.materialize(&request.envelope, &request.materialization_request)?;
    let view = arena.make_view(data.handle.handle, &request.view)?;
    arena.feature_values(view.handle.handle, &request.feature_table)
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

fn build_feature_arrow(
    features: &CoordinatorFeatureBlock,
) -> dag_ml_data_core::Result<(ArrowArray, ArrowSchema)> {
    let mut child_arrays = vec![
        Box::into_raw(Box::new(string_array(
            features
                .observation_ids
                .iter()
                .map(|observation_id| Some(observation_id.as_str())),
        )?)),
        Box::into_raw(Box::new(string_array(
            features
                .sample_ids
                .iter()
                .map(|sample_id| Some(sample_id.as_str())),
        )?)),
    ];
    let mut child_schemas = vec![
        Box::into_raw(Box::new(field_schema("observation_id", "u", false)?)),
        Box::into_raw(Box::new(field_schema("sample_id", "u", false)?)),
    ];
    for (feature_idx, feature_name) in features.feature_names.iter().enumerate() {
        let numeric_values = features
            .values
            .iter()
            .map(|row| match &row[feature_idx] {
                serde_json::Value::Null => Ok(None),
                serde_json::Value::Number(number) => number.as_f64().map(Some).ok_or_else(|| {
                    dag_ml_data_core::DataError::Validation(format!(
                        "feature `{}` contains a non-f64 numeric value",
                        feature_name
                    ))
                }),
                _ => Err(dag_ml_data_core::DataError::Validation(format!(
                    "feature `{}` Arrow smoke only supports numeric or null values",
                    feature_name
                ))),
            })
            .collect::<dag_ml_data_core::Result<Vec<_>>>()?;
        child_arrays.push(Box::into_raw(Box::new(f64_array(
            numeric_values.into_iter(),
        ))));
        child_schemas.push(Box::into_raw(Box::new(field_schema(
            feature_name,
            "g",
            true,
        )?)));
    }
    Ok((
        struct_array(features.observation_ids.len(), child_arrays),
        struct_schema("coordinator_features", child_schemas)?,
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

    #[test]
    fn exports_coordinator_feature_arrow_over_abi() {
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
                "columns": ["f1"],
                "include_augmented": false
            },
            "feature_table": {
                "feature_set_id": "x",
                "representation_id": "tabular_numeric",
                "feature_names": ["f0", "f1"],
                "rows": [
                    {"observation_id": "obs.S001.base", "values": [1.0, 10.0]},
                    {"observation_id": "obs.S001.rep1", "values": [2.0, 20.0]},
                    {"observation_id": "obs.S001.aug0", "values": [3.0, 30.0]},
                    {"observation_id": "obs.S002.base", "values": [4.0, 40.0]}
                ]
            }
        });
        let request = serde_json::to_vec(&request).unwrap();
        let mut array = std::ptr::null_mut();
        let mut schema = std::ptr::null_mut();
        let mut error = DagMlDataString::default();

        let status = unsafe {
            dagmldata_coordinator_feature_arrow_json(
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
            assert_eq!((*array).length, 2);
            assert_eq!((*array).n_children, 3);
            let array_children =
                slice::from_raw_parts((*array).children, (*array).n_children as usize);
            let schema_children =
                slice::from_raw_parts((*schema).children, (*schema).n_children as usize);
            assert_eq!(
                CStr::from_ptr((*schema_children[2]).name).to_str().unwrap(),
                "f1"
            );
            assert_eq!(
                utf8_values(array_children[0]),
                vec![
                    Some("obs.S001.base".to_string()),
                    Some("obs.S001.rep1".to_string()),
                ]
            );
            assert_eq!(f64_values(array_children[2]), vec![Some(10.0), Some(20.0)]);
            dagmldata_arrow_array_free(array);
            dagmldata_arrow_schema_free(schema);
        }
    }

    #[test]
    fn inmemory_provider_vtable_materializes_views_identity_targets_and_features() {
        let envelope = include_bytes!(
            "../../../examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json"
        );
        let materialization_request = include_bytes!(
            "../../../examples/fixtures/oof_campaign/materialization_request_model_base_x.json"
        );
        let target_tables = serde_json::to_vec(&serde_json::json!([
            {
                "target_id": "y",
                "values": [
                    {"sample_id": "S001", "value": 42.0},
                    {"sample_id": "S002", "value": 7.0}
                ]
            }
        ]))
        .unwrap();
        let feature_tables = serde_json::to_vec(&serde_json::json!([
            {
                "feature_set_id": "x",
                "representation_id": "tabular_numeric",
                "feature_names": ["f0", "f1"],
                "rows": [
                    {"observation_id": "obs.S001.base", "values": [1.0, 10.0]},
                    {"observation_id": "obs.S001.rep1", "values": [2.0, 20.0]},
                    {"observation_id": "obs.S001.aug0", "values": [3.0, 30.0]},
                    {"observation_id": "obs.S002.base", "values": [4.0, 40.0]}
                ]
            },
            {
                "feature_set_id": "x_bad_representation",
                "representation_id": "dense_signal",
                "feature_names": ["f0"],
                "rows": [
                    {"observation_id": "obs.S001.base", "values": [1.0]},
                    {"observation_id": "obs.S001.rep1", "values": [2.0]},
                    {"observation_id": "obs.S001.aug0", "values": [3.0]},
                    {"observation_id": "obs.S002.base", "values": [4.0]}
                ]
            }
        ]))
        .unwrap();
        let mut vtable = empty_vtable();
        let mut error = DagMlDataString::default();

        let status = unsafe {
            dagmldata_inmemory_provider_new_with_features_json(
                envelope.as_ptr(),
                envelope.len(),
                target_tables.as_ptr(),
                target_tables.len(),
                feature_tables.as_ptr(),
                feature_tables.len(),
                &mut vtable,
                &mut error,
            )
        };

        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert!(error.ptr.is_null());
        assert!(!vtable.user_data.is_null());
        let mut data_handle = 0;
        let materialize = vtable.materialize.unwrap();
        let status = unsafe {
            materialize(
                vtable.user_data,
                0,
                DagMlDataBytesView {
                    ptr: materialization_request.as_ptr(),
                    len: materialization_request.len(),
                },
                &mut data_handle,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert_eq!(data_handle, 1);

        let view_json = serde_json::to_vec(
            &serde_json::json!({"sample_ids": ["S001"], "columns": ["f1"], "include_augmented": false}),
        )
        .unwrap();
        let mut view_handle = 0;
        let make_view = vtable.make_view.unwrap();
        let status = unsafe {
            make_view(
                vtable.user_data,
                data_handle,
                DagMlDataBytesView {
                    ptr: view_json.as_ptr(),
                    len: view_json.len(),
                },
                &mut view_handle,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert_eq!(view_handle, 2);

        let mut identity_array = std::ptr::null_mut();
        let mut identity_schema = std::ptr::null_mut();
        let view_identity = vtable.view_identity.unwrap();
        let status = unsafe {
            view_identity(
                vtable.user_data,
                view_handle,
                &mut identity_array,
                &mut identity_schema,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        unsafe {
            assert_eq!((*identity_array).length, 2);
            let array_children = slice::from_raw_parts(
                (*identity_array).children,
                (*identity_array).n_children as usize,
            );
            assert_eq!(
                utf8_values(array_children[0]),
                vec![
                    Some("obs.S001.base".to_string()),
                    Some("obs.S001.rep1".to_string()),
                ]
            );
            dagmldata_arrow_array_free(identity_array);
            dagmldata_arrow_schema_free(identity_schema);
        }

        let mut target_array = std::ptr::null_mut();
        let mut target_schema = std::ptr::null_mut();
        let target_arrow = vtable.target_arrow.unwrap();
        let target_name = b"y";
        let status = unsafe {
            target_arrow(
                vtable.user_data,
                view_handle,
                DagMlDataBytesView {
                    ptr: target_name.as_ptr(),
                    len: target_name.len(),
                },
                &mut target_array,
                &mut target_schema,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        unsafe {
            assert_eq!((*target_array).length, 1);
            let array_children = slice::from_raw_parts(
                (*target_array).children,
                (*target_array).n_children as usize,
            );
            assert_eq!(
                utf8_values(array_children[0]),
                vec![Some("S001".to_string())]
            );
            assert_eq!(f64_values(array_children[2]), vec![Some(42.0)]);
            dagmldata_arrow_array_free(target_array);
            dagmldata_arrow_schema_free(target_schema);
        }

        let mut feature_array = std::ptr::null_mut();
        let mut feature_schema = std::ptr::null_mut();
        let feature_arrow = vtable.feature_arrow.unwrap();
        let feature_set_name = b"x";
        let status = unsafe {
            feature_arrow(
                vtable.user_data,
                view_handle,
                DagMlDataBytesView {
                    ptr: feature_set_name.as_ptr(),
                    len: feature_set_name.len(),
                },
                &mut feature_array,
                &mut feature_schema,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        unsafe {
            assert_eq!((*feature_array).length, 2);
            assert_eq!((*feature_array).n_children, 3);
            let array_children = slice::from_raw_parts(
                (*feature_array).children,
                (*feature_array).n_children as usize,
            );
            assert_eq!(
                utf8_values(array_children[0]),
                vec![
                    Some("obs.S001.base".to_string()),
                    Some("obs.S001.rep1".to_string()),
                ]
            );
            assert_eq!(f64_values(array_children[2]), vec![Some(10.0), Some(20.0)]);
            dagmldata_arrow_array_free(feature_array);
            dagmldata_arrow_schema_free(feature_schema);
        }
        let bad_feature_set_name = b"x_bad_representation";
        let mut bad_feature_array = std::ptr::null_mut();
        let mut bad_feature_schema = std::ptr::null_mut();
        let status = unsafe {
            feature_arrow(
                vtable.user_data,
                view_handle,
                DagMlDataBytesView {
                    ptr: bad_feature_set_name.as_ptr(),
                    len: bad_feature_set_name.len(),
                },
                &mut bad_feature_array,
                &mut bad_feature_schema,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::ValidationError);
        assert!(bad_feature_array.is_null());
        assert!(bad_feature_schema.is_null());

        unsafe {
            vtable.release.unwrap()(vtable.user_data, view_handle);
        }
        let mut released_array = std::ptr::null_mut();
        let mut released_schema = std::ptr::null_mut();
        let status = unsafe {
            view_identity(
                vtable.user_data,
                view_handle,
                &mut released_array,
                &mut released_schema,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::ValidationError);
        assert!(released_array.is_null());
        assert!(released_schema.is_null());

        let mut child_view_handle = 0;
        let status = unsafe {
            make_view(
                vtable.user_data,
                data_handle,
                DagMlDataBytesView {
                    ptr: view_json.as_ptr(),
                    len: view_json.len(),
                },
                &mut child_view_handle,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert_ne!(child_view_handle, 0);
        unsafe {
            vtable.release.unwrap()(vtable.user_data, data_handle);
        }
        let mut child_array = std::ptr::null_mut();
        let mut child_schema = std::ptr::null_mut();
        let status = unsafe {
            view_identity(
                vtable.user_data,
                child_view_handle,
                &mut child_array,
                &mut child_schema,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::ValidationError);
        assert!(child_array.is_null());
        assert!(child_schema.is_null());

        let mut orphan_view_handle = 0;
        let status = unsafe {
            make_view(
                vtable.user_data,
                data_handle,
                DagMlDataBytesView {
                    ptr: view_json.as_ptr(),
                    len: view_json.len(),
                },
                &mut orphan_view_handle,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::ValidationError);
        assert_eq!(orphan_view_handle, 0);

        unsafe {
            dagmldata_inmemory_provider_destroy(&mut vtable);
        }
        assert!(vtable.user_data.is_null());
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
