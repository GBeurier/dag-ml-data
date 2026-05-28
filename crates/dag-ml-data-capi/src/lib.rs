use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::{c_void, CString};
use std::os::raw::c_char;
use std::slice;

#[cfg(test)]
use dag_ml_data_core::SampleId;
use dag_ml_data_core::{
    collate_feature_block, fuse_feature_blocks, schema_fingerprint, CollationPolicy,
    CoordinatorDataHandleRecord, CoordinatorDataMaterializationRequest,
    CoordinatorDataPlanEnvelope, CoordinatorFeatureBlock, CoordinatorFeatureTable,
    CoordinatorHandleArena, CoordinatorTargetBlock, CoordinatorTargetTable, DataView,
    DatasetSchema, FeatureFusionPolicy, FittedAdapterManifest, FittedAdapterMaterializationRequest,
    FittedAdapterRef, InMemoryFittedAdapterStore, NumericFeatureBufferArena,
    NumericFeatureBufferStore, NumericFeatureMatrixF64, NumericFeatureMatrixF64Columnar,
    NumericTensorBlock, ObservationId, RepresentationId, RuntimeFittedAdapterStore,
    SampleAlignmentPlan, SourceFeatureBlock, SourceId, TargetId,
};
use serde::Deserialize;

pub type DagMlDataHandle = u64;
pub const DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION: u32 = 2;

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
#[derive(Clone, Copy, Debug)]
pub struct DagMlDataFeatureMatrixF64View {
    pub feature_set_id: DagMlDataBytesView,
    pub representation_id: DagMlDataBytesView,
    pub feature_names: *const DagMlDataBytesView,
    pub feature_names_len: usize,
    pub observation_ids: *const DagMlDataBytesView,
    pub observation_ids_len: usize,
    pub values: *const f64,
    pub values_len: usize,
    pub validity_mask: *const u8,
    pub validity_mask_len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DagMlDataF64ColumnView {
    pub values: *const f64,
    pub values_len: usize,
    pub validity_mask: *const u8,
    pub validity_mask_len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DagMlDataFeatureMatrixF64ColumnarView {
    pub feature_set_id: DagMlDataBytesView,
    pub representation_id: DagMlDataBytesView,
    pub feature_names: *const DagMlDataBytesView,
    pub feature_names_len: usize,
    pub observation_ids: *const DagMlDataBytesView,
    pub observation_ids_len: usize,
    pub columns: *const DagMlDataF64ColumnView,
    pub columns_len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DagMlDataStringArray {
    pub ptr: *mut DagMlDataString,
    pub len: usize,
}

impl Default for DagMlDataStringArray {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            len: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DagMlDataUSizeArray {
    pub ptr: *mut usize,
    pub len: usize,
}

impl Default for DagMlDataUSizeArray {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            len: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DagMlDataF64Array {
    pub ptr: *mut f64,
    pub len: usize,
}

impl Default for DagMlDataF64Array {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            len: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DagMlDataF32Array {
    pub ptr: *mut f32,
    pub len: usize,
}

impl Default for DagMlDataF32Array {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            len: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DagMlDataU8Array {
    pub ptr: *mut u8,
    pub len: usize,
}

impl Default for DagMlDataU8Array {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            len: 0,
        }
    }
}

pub const DAG_ML_DATA_TENSOR_F64_ABI_VERSION: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DagMlDataTensorF64 {
    pub abi_version: u32,
    pub block_id: DagMlDataString,
    pub representation_id: DagMlDataString,
    pub batch_container: DagMlDataString,
    pub observation_ids: DagMlDataStringArray,
    pub sample_ids: DagMlDataStringArray,
    pub shape: DagMlDataUSizeArray,
    pub values: DagMlDataF64Array,
    pub presence_mask: DagMlDataU8Array,
    pub validity_mask: DagMlDataU8Array,
    pub feature_names: DagMlDataStringArray,
}

impl Default for DagMlDataTensorF64 {
    fn default() -> Self {
        Self {
            abi_version: DAG_ML_DATA_TENSOR_F64_ABI_VERSION,
            block_id: DagMlDataString::default(),
            representation_id: DagMlDataString::default(),
            batch_container: DagMlDataString::default(),
            observation_ids: DagMlDataStringArray::default(),
            sample_ids: DagMlDataStringArray::default(),
            shape: DagMlDataUSizeArray::default(),
            values: DagMlDataF64Array::default(),
            presence_mask: DagMlDataU8Array::default(),
            validity_mask: DagMlDataU8Array::default(),
            feature_names: DagMlDataStringArray::default(),
        }
    }
}

pub const DAG_ML_DATA_TENSOR_F32_ABI_VERSION: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DagMlDataTensorF32 {
    pub abi_version: u32,
    pub block_id: DagMlDataString,
    pub representation_id: DagMlDataString,
    pub batch_container: DagMlDataString,
    pub observation_ids: DagMlDataStringArray,
    pub sample_ids: DagMlDataStringArray,
    pub shape: DagMlDataUSizeArray,
    pub values: DagMlDataF32Array,
    pub presence_mask: DagMlDataU8Array,
    pub validity_mask: DagMlDataU8Array,
    pub feature_names: DagMlDataStringArray,
}

impl Default for DagMlDataTensorF32 {
    fn default() -> Self {
        Self {
            abi_version: DAG_ML_DATA_TENSOR_F32_ABI_VERSION,
            block_id: DagMlDataString::default(),
            representation_id: DagMlDataString::default(),
            batch_container: DagMlDataString::default(),
            observation_ids: DagMlDataStringArray::default(),
            sample_ids: DagMlDataStringArray::default(),
            shape: DagMlDataUSizeArray::default(),
            values: DagMlDataF32Array::default(),
            presence_mask: DagMlDataU8Array::default(),
            validity_mask: DagMlDataU8Array::default(),
            feature_names: DagMlDataStringArray::default(),
        }
    }
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

/// Releases a tensor allocated by DAG-ML-DATA.
///
/// # Safety
///
/// Every pointer inside `tensor` must either be null or come from a
/// DAG-ML-DATA C ABI function returning `DagMlDataTensorF64`. Passing nested
/// pointers from any other allocator, or freeing the same tensor twice, is
/// undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_tensor_f64_free(tensor: DagMlDataTensorF64) {
    dagmldata_string_free(tensor.block_id);
    dagmldata_string_free(tensor.representation_id);
    dagmldata_string_free(tensor.batch_container);
    free_string_array(tensor.observation_ids);
    free_string_array(tensor.sample_ids);
    free_usize_array(tensor.shape);
    free_f64_array(tensor.values);
    free_u8_array(tensor.presence_mask);
    free_u8_array(tensor.validity_mask);
    free_string_array(tensor.feature_names);
}

/// Releases an f32 tensor allocated by DAG-ML-DATA.
///
/// # Safety
///
/// Every pointer inside `tensor` must either be null or come from a
/// DAG-ML-DATA C ABI function returning `DagMlDataTensorF32`. Passing nested
/// pointers from any other allocator, or freeing the same tensor twice, is
/// undefined behavior.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_tensor_f32_free(tensor: DagMlDataTensorF32) {
    dagmldata_string_free(tensor.block_id);
    dagmldata_string_free(tensor.representation_id);
    dagmldata_string_free(tensor.batch_container);
    free_string_array(tensor.observation_ids);
    free_string_array(tensor.sample_ids);
    free_usize_array(tensor.shape);
    free_f32_array(tensor.values);
    free_u8_array(tensor.presence_mask);
    free_u8_array(tensor.validity_mask);
    free_string_array(tensor.feature_names);
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

/// Opaque handle to a Rust-owned in-memory fitted-adapter store. Callers
/// receive this from `dagmldata_inmemory_fitted_adapter_store_new` and must
/// eventually release it with `dagmldata_inmemory_fitted_adapter_store_destroy`.
///
/// The underlying Rust store is internally synchronized with a
/// `std::sync::Mutex`, so the same handle value can be shared across host
/// threads and concurrent `register`/`materialize`/`release` calls serialize
/// safely instead of racing. `destroy` is the only operation that must be
/// externally synchronized: no other call on the handle may overlap with
/// `destroy`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DagMlDataFittedAdapterStoreHandle {
    pub ptr: *mut c_void,
}

impl Default for DagMlDataFittedAdapterStoreHandle {
    fn default() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
        }
    }
}

/// Creates a Rust-owned in-memory fitted-adapter store. The returned handle
/// must be released with `dagmldata_inmemory_fitted_adapter_store_destroy`.
///
/// # Safety
///
/// `out_store` must point to writable memory for one
/// `DagMlDataFittedAdapterStoreHandle`.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_inmemory_fitted_adapter_store_new(
    out_store: *mut DagMlDataFittedAdapterStoreHandle,
) -> DagMlDataStatusCode {
    if out_store.is_null() {
        return DagMlDataStatusCode::InvalidArgument;
    }
    let store = Box::new(InMemoryFittedAdapterStore::new());
    *out_store = DagMlDataFittedAdapterStoreHandle {
        ptr: Box::into_raw(store).cast::<c_void>(),
    };
    DagMlDataStatusCode::Ok
}

/// Destroys a fitted-adapter store handle previously returned by
/// `dagmldata_inmemory_fitted_adapter_store_new`.
///
/// # Safety
///
/// `store.ptr` must either be null or point to a store handle returned by
/// `dagmldata_inmemory_fitted_adapter_store_new` and not already destroyed.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_inmemory_fitted_adapter_store_destroy(
    store: DagMlDataFittedAdapterStoreHandle,
) {
    if !store.ptr.is_null() {
        drop(Box::from_raw(
            store.ptr.cast::<InMemoryFittedAdapterStore>(),
        ));
    }
}

/// Registers a JSON `FittedAdapterRef` in the store. On success writes the
/// allocated u64 handle to `out_handle`. Returns `ValidationError` if the ref
/// fails validation or if an adapter with the same `adapter_id` is already
/// registered.
///
/// # Safety
///
/// `store.ptr` must point to a live store handle. `json_ptr` must point to
/// `json_len` readable bytes. `out_handle` and `error_out` may be null; any
/// returned error string must be released with `dagmldata_string_free`.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_inmemory_fitted_adapter_store_register_json(
    store: DagMlDataFittedAdapterStoreHandle,
    json_ptr: *const u8,
    json_len: usize,
    out_handle: *mut u64,
    error_out: *mut DagMlDataString,
) -> DagMlDataStatusCode {
    clear_string(error_out);
    if store.ptr.is_null() || json_ptr.is_null() {
        set_string(error_out, "store or json pointer is null");
        return DagMlDataStatusCode::InvalidArgument;
    }
    let store_ref = &*store.ptr.cast::<InMemoryFittedAdapterStore>();
    let json = slice::from_raw_parts(json_ptr, json_len);
    let value: FittedAdapterRef = match serde_json::from_slice(json) {
        Ok(value) => value,
        Err(error) => {
            set_string(error_out, error.to_string());
            return DagMlDataStatusCode::ValidationError;
        }
    };
    match store_ref.register(value) {
        Ok(record) => {
            if !out_handle.is_null() {
                *out_handle = record.handle;
            }
            DagMlDataStatusCode::Ok
        }
        Err(error) => {
            set_string(error_out, error.to_string());
            DagMlDataStatusCode::ValidationError
        }
    }
}

/// Materializes an opaque handle for a previously registered fitted adapter
/// from a JSON `FittedAdapterMaterializationRequest`. Writes the handle id to
/// `out_handle` on success. Returns `ValidationError` if the request fails
/// validation, the adapter is missing, or the request's `params_fingerprint`
/// does not match the registered ref's fingerprint.
///
/// # Safety
///
/// Same pointer ownership rules as
/// `dagmldata_inmemory_fitted_adapter_store_register_json`.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_inmemory_fitted_adapter_store_materialize_json(
    store: DagMlDataFittedAdapterStoreHandle,
    json_ptr: *const u8,
    json_len: usize,
    out_handle: *mut u64,
    error_out: *mut DagMlDataString,
) -> DagMlDataStatusCode {
    clear_string(error_out);
    if store.ptr.is_null() || json_ptr.is_null() {
        set_string(error_out, "store or json pointer is null");
        return DagMlDataStatusCode::InvalidArgument;
    }
    let store_ref = &*store.ptr.cast::<InMemoryFittedAdapterStore>();
    let json = slice::from_raw_parts(json_ptr, json_len);
    let request: FittedAdapterMaterializationRequest = match serde_json::from_slice(json) {
        Ok(request) => request,
        Err(error) => {
            set_string(error_out, error.to_string());
            return DagMlDataStatusCode::ValidationError;
        }
    };
    match store_ref.materialize(&request) {
        Ok(handle) => {
            if !out_handle.is_null() {
                *out_handle = handle;
            }
            DagMlDataStatusCode::Ok
        }
        Err(error) => {
            set_string(error_out, error.to_string());
            DagMlDataStatusCode::ValidationError
        }
    }
}

/// Releases a registered fitted-adapter handle by `adapter_id`. Returns
/// `Ok` whether the adapter existed or not; callers can inspect
/// `out_released` (non-zero if a record was removed) when non-null.
///
/// # Safety
///
/// `store.ptr` must point to a live store handle. `adapter_id` must point to
/// `adapter_id_len` UTF-8 bytes for the duration of the call. `out_released`
/// may be null.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_inmemory_fitted_adapter_store_release(
    store: DagMlDataFittedAdapterStoreHandle,
    adapter_id: *const u8,
    adapter_id_len: usize,
    out_released: *mut u8,
) -> DagMlDataStatusCode {
    if store.ptr.is_null() || adapter_id.is_null() {
        return DagMlDataStatusCode::InvalidArgument;
    }
    let store_ref = &*store.ptr.cast::<InMemoryFittedAdapterStore>();
    let adapter_id_bytes = slice::from_raw_parts(adapter_id, adapter_id_len);
    let adapter_id_str = match std::str::from_utf8(adapter_id_bytes) {
        Ok(value) => value,
        Err(_) => return DagMlDataStatusCode::InvalidArgument,
    };
    let released = store_ref.release(adapter_id_str);
    if !out_released.is_null() {
        *out_released = u8::from(released);
    }
    DagMlDataStatusCode::Ok
}

/// Attaches a fitted-adapter store handle to a Rust-owned in-memory provider
/// vtable so the provider's controllers can materialize fitted-adapter
/// handles through the same lifecycle as the data/view handles. The provider
/// holds a borrowed pointer to the store; the host must keep the store alive
/// while it stays attached and must clear the attachment (by attaching a
/// null handle) before calling `dagmldata_inmemory_fitted_adapter_store_destroy`
/// on the store. Passing `store.ptr == null` detaches any previously
/// attached store without freeing it.
///
/// # Safety
///
/// `vtable` must point to a live `DagMlDataVTable` returned by one of the
/// `dagmldata_inmemory_provider_new*` constructors. `store.ptr` must either
/// be null or a live store handle returned by
/// `dagmldata_inmemory_fitted_adapter_store_new`.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_inmemory_provider_attach_fitted_adapter_store(
    vtable: *const DagMlDataVTable,
    store: DagMlDataFittedAdapterStoreHandle,
    error_out: *mut DagMlDataString,
) -> DagMlDataStatusCode {
    clear_string(error_out);
    if vtable.is_null() || (*vtable).user_data.is_null() {
        set_string(error_out, "provider vtable or user_data is null");
        return DagMlDataStatusCode::InvalidArgument;
    }
    let provider = &*((*vtable).user_data.cast::<InMemoryProvider>());
    let store_ptr = if store.ptr.is_null() {
        std::ptr::null()
    } else {
        store.ptr.cast::<InMemoryFittedAdapterStore>() as *const _
    };
    match provider.fitted_adapter_store.lock() {
        Ok(mut slot) => {
            *slot = store_ptr;
            DagMlDataStatusCode::Ok
        }
        Err(error) => {
            set_string(
                error_out,
                format!("provider fitted-adapter mutex poisoned: {error}"),
            );
            DagMlDataStatusCode::ValidationError
        }
    }
}

/// Materializes a fitted-adapter handle through the provider's attached
/// store. Returns `InvalidArgument` if no fitted-adapter store is attached,
/// `ValidationError` if the request fails, otherwise writes the opaque u64
/// handle to `out_handle` and returns `Ok`.
///
/// # Safety
///
/// `vtable` must point to a live provider vtable. `json_ptr` must point to
/// `json_len` readable bytes encoding a `FittedAdapterMaterializationRequest`.
/// `out_handle` and `error_out` may be null; returned error strings must be
/// released with `dagmldata_string_free`.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_inmemory_provider_materialize_fitted_adapter_json(
    vtable: *const DagMlDataVTable,
    json_ptr: *const u8,
    json_len: usize,
    out_handle: *mut u64,
    error_out: *mut DagMlDataString,
) -> DagMlDataStatusCode {
    clear_string(error_out);
    if vtable.is_null() || (*vtable).user_data.is_null() || json_ptr.is_null() {
        set_string(
            error_out,
            "provider vtable, user_data or json pointer is null",
        );
        return DagMlDataStatusCode::InvalidArgument;
    }
    let provider = &*((*vtable).user_data.cast::<InMemoryProvider>());
    let store_ptr = match provider.fitted_adapter_store.lock() {
        Ok(guard) => *guard,
        Err(error) => {
            set_string(
                error_out,
                format!("provider fitted-adapter mutex poisoned: {error}"),
            );
            return DagMlDataStatusCode::ValidationError;
        }
    };
    if store_ptr.is_null() {
        set_string(
            error_out,
            "provider has no fitted-adapter store attached; \
             call dagmldata_inmemory_provider_attach_fitted_adapter_store first",
        );
        return DagMlDataStatusCode::InvalidArgument;
    }
    let store = &*store_ptr;
    let json = slice::from_raw_parts(json_ptr, json_len);
    let request: FittedAdapterMaterializationRequest = match serde_json::from_slice(json) {
        Ok(request) => request,
        Err(error) => {
            set_string(error_out, error.to_string());
            return DagMlDataStatusCode::ValidationError;
        }
    };
    match store.materialize(&request) {
        Ok(handle) => {
            if !out_handle.is_null() {
                *out_handle = handle;
            }
            DagMlDataStatusCode::Ok
        }
        Err(error) => {
            set_string(error_out, error.to_string());
            DagMlDataStatusCode::ValidationError
        }
    }
}

/// Validates a JSON `FittedAdapterRef` payload against the published v1
/// contract (`fitted_adapter_ref.v1`). Returns `Ok` if the payload parses,
/// passes shape validation and either inline or portable validation as
/// requested; otherwise returns `ValidationError` and writes an error string.
///
/// When `require_portable` is non-zero, the ref must also carry backend, safe
/// relative URI and content fingerprint (matching `validate_portable`).
///
/// # Safety
///
/// When `json_ptr` is non-null it must point to `json_len` readable bytes for
/// the duration of the call. `error_out` may be null; when non-null it must
/// point to writable memory for one `DagMlDataString`. Returned strings must
/// be released with `dagmldata_string_free`.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_fitted_adapter_ref_validate_json(
    json_ptr: *const u8,
    json_len: usize,
    require_portable: u8,
    error_out: *mut DagMlDataString,
) -> DagMlDataStatusCode {
    clear_string(error_out);
    if json_ptr.is_null() {
        set_string(error_out, "json pointer is null");
        return DagMlDataStatusCode::InvalidArgument;
    }
    let json = slice::from_raw_parts(json_ptr, json_len);
    let value: FittedAdapterRef = match serde_json::from_slice(json) {
        Ok(value) => value,
        Err(error) => {
            set_string(error_out, error.to_string());
            return DagMlDataStatusCode::ValidationError;
        }
    };
    let result = if require_portable != 0 {
        value.validate_portable()
    } else {
        value.validate()
    };
    match result {
        Ok(()) => DagMlDataStatusCode::Ok,
        Err(error) => {
            set_string(error_out, error.to_string());
            DagMlDataStatusCode::ValidationError
        }
    }
}

/// Validates a JSON `FittedAdapterManifest` payload against the published v1
/// contract. Same `require_portable` semantics as
/// `dagmldata_fitted_adapter_ref_validate_json`, applied to every manifest
/// entry; the manifest enforces unique adapter ids and key-vs-ref consistency.
///
/// # Safety
///
/// Same pointer ownership rules as `dagmldata_fitted_adapter_ref_validate_json`.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_fitted_adapter_manifest_validate_json(
    json_ptr: *const u8,
    json_len: usize,
    require_portable: u8,
    error_out: *mut DagMlDataString,
) -> DagMlDataStatusCode {
    clear_string(error_out);
    if json_ptr.is_null() {
        set_string(error_out, "json pointer is null");
        return DagMlDataStatusCode::InvalidArgument;
    }
    let json = slice::from_raw_parts(json_ptr, json_len);
    let manifest: FittedAdapterManifest = match serde_json::from_slice(json) {
        Ok(manifest) => manifest,
        Err(error) => {
            set_string(error_out, error.to_string());
            return DagMlDataStatusCode::ValidationError;
        }
    };
    let result = if require_portable != 0 {
        manifest.validate_portable()
    } else {
        manifest.validate()
    };
    match result {
        Ok(()) => DagMlDataStatusCode::Ok,
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

/// Builds an Arrow C Data fused feature table from already materialized
/// coordinator feature blocks.
///
/// The request JSON shape is `{ feature_set_id, sources, alignment, policy? }`,
/// where `sources` is a list of `{ source_id, block }`. The returned table has
/// `observation_id`, `sample_id` and one numeric column per fused feature.
/// Reference-source repetitions are preserved; singleton non-reference rows are
/// broadcast; ambiguous repeated non-reference rows are rejected.
///
/// # Safety
///
/// When `json_ptr` is non-null it must point to `json_len` readable bytes for
/// the duration of the call. `out_arrow_array`, `out_arrow_schema` and
/// `error_out` may be null. Returned Arrow pointers must be released with
/// `dagmldata_arrow_array_free` and `dagmldata_arrow_schema_free`.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_coordinator_feature_fusion_arrow_json(
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
    match serde_json::from_slice::<CoordinatorFeatureFusionArrowRequest>(json) {
        Ok(request) => match fuse_feature_blocks(
            request.feature_set_id,
            &request.sources,
            &request.alignment,
            &request.policy,
        )
        .and_then(|block| build_feature_arrow(&block))
        {
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

/// Builds a JSON row-major tensor from a coordinator feature block.
///
/// The request JSON shape is `{ feature_block, policy? }`. The output JSON is a
/// `NumericTensorBlock`: it preserves observation/sample identity and emits
/// deterministic shape, values, optional presence mask and optional validity
/// mask.
///
/// # Safety
///
/// When `json_ptr` is non-null it must point to `json_len` readable bytes for
/// the duration of the call. `out_json` and `error_out` may be null. Any
/// returned string must be released with `dagmldata_string_free`.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_coordinator_feature_collation_json(
    json_ptr: *const u8,
    json_len: usize,
    out_json: *mut DagMlDataString,
    error_out: *mut DagMlDataString,
) -> DagMlDataStatusCode {
    clear_string(out_json);
    clear_string(error_out);
    if json_ptr.is_null() {
        set_string(error_out, "json pointer is null");
        return DagMlDataStatusCode::InvalidArgument;
    }

    let json = slice::from_raw_parts(json_ptr, json_len);
    match serde_json::from_slice::<CoordinatorFeatureCollationJsonRequest>(json) {
        Ok(request) => {
            match collate_feature_block(&request.feature_block, &request.policy)
                .and_then(|tensor| serde_json::to_string(&tensor).map_err(Into::into))
            {
                Ok(tensor_json) => {
                    set_string(out_json, tensor_json);
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

/// Builds an owned row-major f64 tensor from a coordinator feature block.
///
/// The request JSON shape is `{ feature_block, policy? }`. The returned tensor
/// owns contiguous f64 values, shape, identity arrays and optional masks. It
/// must be released with `dagmldata_tensor_f64_free`.
///
/// # Safety
///
/// When `json_ptr` is non-null it must point to `json_len` readable bytes for
/// the duration of the call. `out_tensor` must point to writable memory for one
/// `DagMlDataTensorF64`. `error_out` may be null.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_coordinator_feature_collation_tensor_f64_json(
    json_ptr: *const u8,
    json_len: usize,
    out_tensor: *mut DagMlDataTensorF64,
    error_out: *mut DagMlDataString,
) -> DagMlDataStatusCode {
    clear_tensor(out_tensor);
    clear_string(error_out);
    if json_ptr.is_null() {
        set_string(error_out, "json pointer is null");
        return DagMlDataStatusCode::InvalidArgument;
    }
    if out_tensor.is_null() {
        set_string(error_out, "tensor output pointer is null");
        return DagMlDataStatusCode::InvalidArgument;
    }

    let json = slice::from_raw_parts(json_ptr, json_len);
    match serde_json::from_slice::<CoordinatorFeatureCollationJsonRequest>(json) {
        Ok(request) => match collate_feature_block(&request.feature_block, &request.policy) {
            Ok(tensor) => {
                *out_tensor = tensor_to_c(tensor);
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

/// Builds an owned row-major f32 tensor from a coordinator feature block.
///
/// The request JSON shape is `{ feature_block, policy? }`, identical to the
/// f64 entry point. The collation kernel still operates on f64 to preserve the
/// canonical numeric semantics; each value is cast to f32 at the ABI boundary
/// and the call is rejected with `ValidationError` if any padded value, finite
/// input or padding fallback does not round-trip into a finite f32 (overflow
/// to infinity, or non-finite input). The returned tensor must be released
/// with `dagmldata_tensor_f32_free`.
///
/// # Safety
///
/// When `json_ptr` is non-null it must point to `json_len` readable bytes for
/// the duration of the call. `out_tensor` must point to writable memory for one
/// `DagMlDataTensorF32`. `error_out` may be null.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_coordinator_feature_collation_tensor_f32_json(
    json_ptr: *const u8,
    json_len: usize,
    out_tensor: *mut DagMlDataTensorF32,
    error_out: *mut DagMlDataString,
) -> DagMlDataStatusCode {
    clear_tensor_f32(out_tensor);
    clear_string(error_out);
    if json_ptr.is_null() {
        set_string(error_out, "json pointer is null");
        return DagMlDataStatusCode::InvalidArgument;
    }
    if out_tensor.is_null() {
        set_string(error_out, "tensor output pointer is null");
        return DagMlDataStatusCode::InvalidArgument;
    }

    let json = slice::from_raw_parts(json_ptr, json_len);
    match serde_json::from_slice::<CoordinatorFeatureCollationJsonRequest>(json) {
        Ok(request) => match collate_feature_block(&request.feature_block, &request.policy)
            .and_then(tensor_to_c_f32)
        {
            Ok(tensor) => {
                *out_tensor = tensor;
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
    match InMemoryProvider::new(
        envelope,
        target_tables,
        NumericFeatureBufferStore::default(),
    ) {
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

/// Creates a Rust-owned in-memory provider with typed row-major f64 feature matrices.
///
/// `feature_matrices_ptr/feature_matrices_len` may be null/zero, or a JSON
/// array of `NumericFeatureMatrixF64` values. Unlike
/// `dagmldata_inmemory_provider_new_with_features_json`, this path avoids
/// per-cell `serde_json::Value` parsing for numeric feature values.
///
/// # Safety
///
/// Non-null byte pointers must point to readable memory for the duration of the
/// call. `out_vtable` may be null only if the caller is probing error handling.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_inmemory_provider_new_with_f64_features_json(
    envelope_ptr: *const u8,
    envelope_len: usize,
    target_tables_ptr: *const u8,
    target_tables_len: usize,
    feature_matrices_ptr: *const u8,
    feature_matrices_len: usize,
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
    let feature_store = match parse_f64_feature_matrices(feature_matrices_ptr, feature_matrices_len)
    {
        Ok(feature_store) => feature_store,
        Err(error) => {
            set_string(error_out, error.to_string());
            return DagMlDataStatusCode::ValidationError;
        }
    };
    match InMemoryProvider::new(envelope, target_tables, feature_store) {
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

/// Creates a Rust-owned in-memory provider with borrowed C f64 feature matrices.
///
/// The borrowed `DagMlDataFeatureMatrixF64View` descriptors are copied into
/// Rust-owned buffers during this call. Callers keep ownership of all input
/// pointers and may release them after the function returns.
///
/// # Safety
///
/// Non-null byte pointers and matrix pointers must point to readable memory for
/// the duration of the call. `out_vtable` may be null only if the caller is
/// probing error handling.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_inmemory_provider_new_with_f64_feature_views(
    envelope_ptr: *const u8,
    envelope_len: usize,
    target_tables_ptr: *const u8,
    target_tables_len: usize,
    feature_matrices_ptr: *const DagMlDataFeatureMatrixF64View,
    feature_matrices_len: usize,
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
    let feature_store =
        match parse_f64_feature_matrix_views(feature_matrices_ptr, feature_matrices_len) {
            Ok(feature_store) => feature_store,
            Err(error) => {
                set_string(error_out, error.to_string());
                return DagMlDataStatusCode::ValidationError;
            }
        };
    match InMemoryProvider::new(envelope, target_tables, feature_store) {
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

/// Creates a Rust-owned in-memory provider with borrowed C column-major f64 feature matrices.
///
/// `DagMlDataFeatureMatrixF64ColumnarView` mirrors the production columnar
/// layout of Arrow/Parquet/NumPy column ndarrays: one f64 slice per feature
/// column, with optional per-column validity bitmaps. The column slices and
/// optional validity masks are copied into Rust-owned buffers during this
/// call, so callers may release them after the function returns and pay no
/// row-major transpose on the hot ingestion path.
///
/// # Safety
///
/// Non-null byte pointers and matrix/column pointers must point to readable
/// memory for the duration of the call. `out_vtable` may be null only if the
/// caller is probing error handling.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_inmemory_provider_new_with_f64_feature_columns(
    envelope_ptr: *const u8,
    envelope_len: usize,
    target_tables_ptr: *const u8,
    target_tables_len: usize,
    feature_matrices_ptr: *const DagMlDataFeatureMatrixF64ColumnarView,
    feature_matrices_len: usize,
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
    let feature_store =
        match parse_f64_feature_matrix_columnar_views(feature_matrices_ptr, feature_matrices_len) {
            Ok(feature_store) => feature_store,
            Err(error) => {
                set_string(error_out, error.to_string());
                return DagMlDataStatusCode::ValidationError;
            }
        };
    match InMemoryProvider::new(envelope, target_tables, feature_store) {
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

/// Creates a Rust-owned in-memory provider whose feature buffers are loaded
/// from a deterministic `.n4d` binary file produced by the dag-ml-data file
/// store serializer. Lets a host persist a provider's feature buffers across
/// processes without re-ingesting the source data.
///
/// `path_ptr/path_len` must encode a UTF-8 filesystem path. Loading errors
/// (truncation, bad magic, unsupported version, SHA-256 mismatch) bubble up
/// as `ValidationError` with the error string filled in.
///
/// # Safety
///
/// Non-null byte pointers must point to readable memory for the duration of
/// the call. `out_vtable` may be null only if the caller is probing error
/// handling.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_inmemory_provider_new_from_file(
    envelope_ptr: *const u8,
    envelope_len: usize,
    target_tables_ptr: *const u8,
    target_tables_len: usize,
    path_ptr: *const u8,
    path_len: usize,
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
    if path_ptr.is_null() {
        set_string(error_out, "buffer-store path pointer is null");
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
    let path_bytes = slice::from_raw_parts(path_ptr, path_len);
    let path_str = match std::str::from_utf8(path_bytes) {
        Ok(value) => value,
        Err(error) => {
            set_string(
                error_out,
                format!("buffer-store path is not valid utf-8: {error}"),
            );
            return DagMlDataStatusCode::ValidationError;
        }
    };
    let path = std::path::Path::new(path_str);
    let feature_store = match dag_ml_data_core::buffer_file_store::read_store_from_path(path) {
        Ok(store) => store,
        Err(error) => {
            set_string(error_out, error.to_string());
            return DagMlDataStatusCode::ValidationError;
        }
    };
    match InMemoryProvider::new(envelope, target_tables, feature_store) {
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

/// Creates a Rust-owned in-memory provider whose feature buffers are loaded
/// from an Apache Arrow IPC file on disk. Only available when the
/// `arrow-ipc` feature is enabled — host bindings opt into the `arrow`
/// dependency by depending on `dag-ml-data-capi` with that feature on.
///
/// `path_ptr/path_len` must encode a UTF-8 filesystem path to an Arrow IPC
/// file. Each top-level `RecordBatch` becomes one feature buffer; its
/// schema must declare `dag_ml_data.feature_set_id` and
/// `dag_ml_data.representation_id` metadata, and expose an
/// `observation_id` Utf8 column. See `dag-ml-data-arrow` for the full
/// mapping contract.
///
/// # Safety
///
/// Non-null byte pointers must point to readable memory for the duration of
/// the call. `out_vtable` may be null only if the caller is probing error
/// handling.
#[cfg(feature = "arrow-ipc")]
#[no_mangle]
pub unsafe extern "C" fn dagmldata_inmemory_provider_new_from_arrow_ipc(
    envelope_ptr: *const u8,
    envelope_len: usize,
    target_tables_ptr: *const u8,
    target_tables_len: usize,
    path_ptr: *const u8,
    path_len: usize,
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
    if path_ptr.is_null() {
        set_string(error_out, "arrow IPC path pointer is null");
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
    let path_bytes = slice::from_raw_parts(path_ptr, path_len);
    let path_str = match std::str::from_utf8(path_bytes) {
        Ok(value) => value,
        Err(error) => {
            set_string(
                error_out,
                format!("arrow IPC path is not valid utf-8: {error}"),
            );
            return DagMlDataStatusCode::ValidationError;
        }
    };
    let path = std::path::Path::new(path_str);
    let feature_store = match dag_ml_data_arrow::read_buffers_from_ipc_path(path) {
        Ok(store) => store,
        Err(error) => {
            set_string(error_out, error.to_string());
            return DagMlDataStatusCode::ValidationError;
        }
    };
    match InMemoryProvider::new(envelope, target_tables, feature_store) {
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

/// Returns the provider-owned numeric feature-buffer manifests as JSON.
///
/// The JSON is an array of `NumericFeatureBufferManifest` values. It is a
/// conformance/debug export for bindings that need to verify which typed
/// buffers are loaded before creating views, feature Arrow arrays or tensors.
///
/// # Safety
///
/// `vtable` must point to a live vtable returned by
/// any `dagmldata_inmemory_provider_new*` constructor (JSON, typed f64,
/// borrowed views, columnar borrowed views, or file-backed). Returned
/// strings must be released with `dagmldata_string_free`.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_inmemory_provider_feature_buffer_manifest_json(
    vtable: *const DagMlDataVTable,
    out_json: *mut DagMlDataString,
    error_out: *mut DagMlDataString,
) -> DagMlDataStatusCode {
    clear_string(out_json);
    clear_string(error_out);
    if vtable.is_null() || (*vtable).user_data.is_null() {
        set_string(error_out, "provider vtable or user_data is null");
        return DagMlDataStatusCode::InvalidArgument;
    }

    let provider = &*((*vtable).user_data.cast::<InMemoryProvider>());
    match provider
        .feature_arena
        .borrow()
        .manifests()
        .and_then(|manifests| serde_json::to_string(&manifests).map_err(Into::into))
    {
        Ok(json) => {
            set_string(out_json, json);
            DagMlDataStatusCode::Ok
        }
        Err(error) => {
            set_string(error_out, error.to_string());
            DagMlDataStatusCode::ValidationError
        }
    }
}

/// Returns feature-buffer bindings for one live materialized data handle.
///
/// Unlike the provider-wide manifest export, this reports only buffers whose
/// representation and observation coverage are compatible with the scoped
/// coordinator relations owned by `data_handle`.
///
/// # Safety
///
/// `vtable` must point to a live vtable returned by
/// any `dagmldata_inmemory_provider_new*` constructor (JSON, typed f64,
/// borrowed views, columnar borrowed views, or file-backed). Returned
/// strings must be released with `dagmldata_string_free`.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_inmemory_provider_data_feature_buffer_manifest_json(
    vtable: *const DagMlDataVTable,
    data_handle: DagMlDataHandle,
    out_json: *mut DagMlDataString,
    error_out: *mut DagMlDataString,
) -> DagMlDataStatusCode {
    clear_string(out_json);
    clear_string(error_out);
    if vtable.is_null() || (*vtable).user_data.is_null() {
        set_string(error_out, "provider vtable or user_data is null");
        return DagMlDataStatusCode::InvalidArgument;
    }

    let provider = &*((*vtable).user_data.cast::<InMemoryProvider>());
    match provider
        .feature_arena
        .borrow()
        .bindings_for_data_handle(data_handle)
        .and_then(|bindings| serde_json::to_string(&bindings).map_err(Into::into))
    {
        Ok(json) => {
            set_string(out_json, json);
            DagMlDataStatusCode::Ok
        }
        Err(error) => {
            set_string(error_out, error.to_string());
            DagMlDataStatusCode::ValidationError
        }
    }
}

/// Builds a JSON row-major tensor from feature buffers owned by the Rust
/// in-memory provider.
///
/// The selector JSON shape is `{ feature_set_id, policy? }` for a single
/// provider feature table, or `{ fusion, policy? }` where `fusion` is a feature
/// fusion selector accepted by the provider. The output JSON is a
/// `NumericTensorBlock`.
///
/// # Safety
///
/// `vtable` must point to a live vtable returned by
/// any `dagmldata_inmemory_provider_new*` constructor; `selector_json.ptr`
/// must point to `selector_json.len` readable bytes. Returned strings must be
/// released with `dagmldata_string_free`.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_inmemory_provider_feature_collation_json(
    vtable: *const DagMlDataVTable,
    view: DagMlDataHandle,
    selector_json: DagMlDataBytesView,
    out_json: *mut DagMlDataString,
    error_out: *mut DagMlDataString,
) -> DagMlDataStatusCode {
    clear_string(out_json);
    clear_string(error_out);
    if vtable.is_null() || (*vtable).user_data.is_null() || selector_json.ptr.is_null() {
        set_string(
            error_out,
            "provider vtable, user_data or selector pointer is null",
        );
        return DagMlDataStatusCode::InvalidArgument;
    }

    let provider = &*((*vtable).user_data.cast::<InMemoryProvider>());
    let selector = slice::from_raw_parts(selector_json.ptr, selector_json.len);
    match serde_json::from_slice::<ProviderFeatureCollationJsonRequest>(selector) {
        Ok(request) => match provider_feature_collation_block(provider, view, &request)
            .and_then(|block| collate_feature_block(&block, &request.policy))
            .and_then(|tensor| serde_json::to_string(&tensor).map_err(Into::into))
        {
            Ok(tensor_json) => {
                set_string(out_json, tensor_json);
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

/// Builds an owned row-major f64 tensor from feature buffers owned by the Rust
/// in-memory provider.
///
/// The selector JSON shape is `{ feature_set_id, policy? }` for a single
/// provider feature table, or `{ fusion, policy? }` where `fusion` is a feature
/// fusion selector accepted by the provider. The returned tensor must be
/// released with `dagmldata_tensor_f64_free`.
///
/// # Safety
///
/// `vtable` must point to a live vtable returned by
/// any `dagmldata_inmemory_provider_new*` constructor; `selector_json.ptr`
/// must point to `selector_json.len` readable bytes. `out_tensor` must point to
/// writable memory for one `DagMlDataTensorF64`. `error_out` may be null.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_inmemory_provider_feature_collation_tensor_f64_json(
    vtable: *const DagMlDataVTable,
    view: DagMlDataHandle,
    selector_json: DagMlDataBytesView,
    out_tensor: *mut DagMlDataTensorF64,
    error_out: *mut DagMlDataString,
) -> DagMlDataStatusCode {
    clear_tensor(out_tensor);
    clear_string(error_out);
    if vtable.is_null()
        || (*vtable).user_data.is_null()
        || selector_json.ptr.is_null()
        || out_tensor.is_null()
    {
        set_string(
            error_out,
            "provider vtable, user_data, selector pointer or tensor output is null",
        );
        return DagMlDataStatusCode::InvalidArgument;
    }

    let provider = &*((*vtable).user_data.cast::<InMemoryProvider>());
    let selector = slice::from_raw_parts(selector_json.ptr, selector_json.len);
    match serde_json::from_slice::<ProviderFeatureCollationJsonRequest>(selector) {
        Ok(request) => match provider_feature_collation_block(provider, view, &request)
            .and_then(|block| collate_feature_block(&block, &request.policy))
        {
            Ok(tensor) => {
                *out_tensor = tensor_to_c(tensor);
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

/// Builds an owned row-major f32 tensor from feature buffers owned by the Rust
/// in-memory provider.
///
/// The selector JSON shape matches the f64 entry point. The collation kernel
/// runs in f64 and each value is cast to f32 at the ABI boundary; the call is
/// rejected with `ValidationError` if any padded value, finite input or padding
/// fallback does not round-trip into a finite f32. The returned tensor must be
/// released with `dagmldata_tensor_f32_free`.
///
/// # Safety
///
/// `vtable` must point to a live vtable returned by
/// any `dagmldata_inmemory_provider_new*` constructor; `selector_json.ptr`
/// must point to `selector_json.len` readable bytes. `out_tensor` must point to
/// writable memory for one `DagMlDataTensorF32`. `error_out` may be null.
#[no_mangle]
pub unsafe extern "C" fn dagmldata_inmemory_provider_feature_collation_tensor_f32_json(
    vtable: *const DagMlDataVTable,
    view: DagMlDataHandle,
    selector_json: DagMlDataBytesView,
    out_tensor: *mut DagMlDataTensorF32,
    error_out: *mut DagMlDataString,
) -> DagMlDataStatusCode {
    clear_tensor_f32(out_tensor);
    clear_string(error_out);
    if vtable.is_null()
        || (*vtable).user_data.is_null()
        || selector_json.ptr.is_null()
        || out_tensor.is_null()
    {
        set_string(
            error_out,
            "provider vtable, user_data, selector pointer or tensor output is null",
        );
        return DagMlDataStatusCode::InvalidArgument;
    }

    let provider = &*((*vtable).user_data.cast::<InMemoryProvider>());
    let selector = slice::from_raw_parts(selector_json.ptr, selector_json.len);
    match serde_json::from_slice::<ProviderFeatureCollationJsonRequest>(selector) {
        Ok(request) => match provider_feature_collation_block(provider, view, &request)
            .and_then(|block| collate_feature_block(&block, &request.policy))
            .and_then(tensor_to_c_f32)
        {
            Ok(tensor) => {
                *out_tensor = tensor;
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

unsafe fn clear_string(out: *mut DagMlDataString) {
    if !out.is_null() {
        *out = DagMlDataString::default();
    }
}

unsafe fn clear_tensor(out: *mut DagMlDataTensorF64) {
    if !out.is_null() {
        *out = DagMlDataTensorF64::default();
    }
}

unsafe fn clear_tensor_f32(out: *mut DagMlDataTensorF32) {
    if !out.is_null() {
        *out = DagMlDataTensorF32::default();
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
    *out = owned_string(value);
}

fn owned_string(value: impl Into<String>) -> DagMlDataString {
    let sanitized = value.into().replace('\0', "\\0");
    let c_string = CString::new(sanitized).expect("nul bytes were sanitized");
    let len = c_string.as_bytes().len();
    DagMlDataString {
        ptr: c_string.into_raw(),
        len,
    }
}

fn tensor_to_c(tensor: NumericTensorBlock) -> DagMlDataTensorF64 {
    DagMlDataTensorF64 {
        abi_version: DAG_ML_DATA_TENSOR_F64_ABI_VERSION,
        block_id: owned_string(tensor.block_id),
        representation_id: owned_string(tensor.representation_id.as_str()),
        batch_container: owned_string(tensor.batch_container),
        observation_ids: owned_string_array(
            tensor
                .observation_ids
                .iter()
                .map(|observation_id| observation_id.as_str()),
        ),
        sample_ids: owned_string_array(
            tensor.sample_ids.iter().map(|sample_id| sample_id.as_str()),
        ),
        shape: owned_usize_array(tensor.shape),
        values: owned_f64_array(tensor.values),
        presence_mask: owned_bool_array(tensor.presence_mask),
        validity_mask: owned_bool_array(tensor.validity_mask),
        feature_names: tensor
            .feature_names
            .map(owned_string_array)
            .unwrap_or_default(),
    }
}

fn tensor_to_c_f32(tensor: NumericTensorBlock) -> dag_ml_data_core::Result<DagMlDataTensorF32> {
    // f64 → f32 cast: reject only when the result is non-finite (overflow to
    // ±inf, or upstream NaN that survived collation). Subnormal f32 values
    // are intentionally passed through; consumers running on hardware with
    // flush-to-zero (GPU default, SSE `DAZ`) should be aware that values
    // below ~1.2e-38 may flush at use time.
    let values = tensor
        .values
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            let cast = *value as f32;
            if cast.is_finite() {
                Ok(cast)
            } else {
                Err(dag_ml_data_core::DataError::Validation(format!(
                    "f32 tensor `{}` value at index {idx} ({value}) does not round-trip into a finite f32",
                    tensor.block_id
                )))
            }
        })
        .collect::<dag_ml_data_core::Result<Vec<f32>>>()?;
    Ok(DagMlDataTensorF32 {
        abi_version: DAG_ML_DATA_TENSOR_F32_ABI_VERSION,
        block_id: owned_string(tensor.block_id),
        representation_id: owned_string(tensor.representation_id.as_str()),
        batch_container: owned_string(tensor.batch_container),
        observation_ids: owned_string_array(
            tensor
                .observation_ids
                .iter()
                .map(|observation_id| observation_id.as_str()),
        ),
        sample_ids: owned_string_array(
            tensor.sample_ids.iter().map(|sample_id| sample_id.as_str()),
        ),
        shape: owned_usize_array(tensor.shape),
        values: owned_f32_array(values),
        presence_mask: owned_bool_array(tensor.presence_mask),
        validity_mask: owned_bool_array(tensor.validity_mask),
        feature_names: tensor
            .feature_names
            .map(owned_string_array)
            .unwrap_or_default(),
    })
}

fn owned_string_array<I, S>(values: I) -> DagMlDataStringArray
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let values = values.into_iter().map(owned_string).collect::<Vec<_>>();
    let (ptr, len) = boxed_slice_parts(values);
    DagMlDataStringArray { ptr, len }
}

fn owned_usize_array(values: Vec<usize>) -> DagMlDataUSizeArray {
    let (ptr, len) = boxed_slice_parts(values);
    DagMlDataUSizeArray { ptr, len }
}

fn owned_f64_array(values: Vec<f64>) -> DagMlDataF64Array {
    let (ptr, len) = boxed_slice_parts(values);
    DagMlDataF64Array { ptr, len }
}

fn owned_f32_array(values: Vec<f32>) -> DagMlDataF32Array {
    let (ptr, len) = boxed_slice_parts(values);
    DagMlDataF32Array { ptr, len }
}

fn owned_bool_array(values: Option<Vec<bool>>) -> DagMlDataU8Array {
    let Some(values) = values else {
        return DagMlDataU8Array::default();
    };
    let values = values.into_iter().map(u8::from).collect::<Vec<_>>();
    let (ptr, len) = boxed_slice_parts(values);
    DagMlDataU8Array { ptr, len }
}

fn boxed_slice_parts<T>(values: Vec<T>) -> (*mut T, usize) {
    if values.is_empty() {
        return (std::ptr::null_mut(), 0);
    }
    let mut boxed = values.into_boxed_slice();
    let ptr = boxed.as_mut_ptr();
    let len = boxed.len();
    std::mem::forget(boxed);
    (ptr, len)
}

unsafe fn free_string_array(array: DagMlDataStringArray) {
    if array.ptr.is_null() {
        return;
    }
    for idx in 0..array.len {
        let value = std::ptr::read(array.ptr.add(idx));
        dagmldata_string_free(value);
    }
    drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
        array.ptr, array.len,
    )));
}

unsafe fn free_usize_array(array: DagMlDataUSizeArray) {
    free_boxed_slice(array.ptr, array.len);
}

unsafe fn free_f64_array(array: DagMlDataF64Array) {
    free_boxed_slice(array.ptr, array.len);
}

unsafe fn free_f32_array(array: DagMlDataF32Array) {
    free_boxed_slice(array.ptr, array.len);
}

unsafe fn free_u8_array(array: DagMlDataU8Array) {
    free_boxed_slice(array.ptr, array.len);
}

unsafe fn free_boxed_slice<T>(ptr: *mut T, len: usize) {
    if ptr.is_null() {
        return;
    }
    drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)));
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

#[derive(Debug, Deserialize)]
struct CoordinatorFeatureFusionArrowRequest {
    feature_set_id: String,
    sources: Vec<SourceFeatureBlock>,
    alignment: SampleAlignmentPlan,
    #[serde(default)]
    policy: FeatureFusionPolicy,
}

fn default_owner_controller() -> String {
    "controller:data.provider".to_string()
}

struct InMemoryProvider {
    arena: CoordinatorHandleArena,
    envelope: CoordinatorDataPlanEnvelope,
    target_tables: BTreeMap<TargetId, CoordinatorTargetTable>,
    feature_arena: RefCell<NumericFeatureBufferArena>,
    // Borrowed pointer to a host-owned `InMemoryFittedAdapterStore`. The host
    // is responsible for keeping the store alive while it stays attached to
    // the provider and for clearing the attachment (by attaching a null
    // handle) before destroying the store. The pointer is read inside a
    // `Mutex` guard so concurrent attach/detach from multiple host threads
    // never race the materialize path.
    fitted_adapter_store: std::sync::Mutex<*const InMemoryFittedAdapterStore>,
}

// The fitted-adapter store pointer is only ever dereferenced while the
// `Mutex` is held, and `InMemoryFittedAdapterStore` is itself `Send + Sync`,
// so the borrowed pointer is safe to share across threads.
unsafe impl Send for InMemoryProvider {}
unsafe impl Sync for InMemoryProvider {}

impl InMemoryProvider {
    fn new(
        envelope: CoordinatorDataPlanEnvelope,
        target_tables: BTreeMap<TargetId, CoordinatorTargetTable>,
        feature_store: NumericFeatureBufferStore,
    ) -> dag_ml_data_core::Result<Self> {
        envelope.validate()?;
        Ok(Self {
            arena: CoordinatorHandleArena::new(default_owner_controller())?,
            envelope,
            target_tables,
            feature_arena: RefCell::new(NumericFeatureBufferArena::new(feature_store)),
            fitted_adapter_store: std::sync::Mutex::new(std::ptr::null()),
        })
    }
}

#[derive(Debug, Deserialize)]
struct CoordinatorFeatureCollationJsonRequest {
    feature_block: CoordinatorFeatureBlock,
    #[serde(default)]
    policy: CollationPolicy,
}

#[derive(Debug, Deserialize)]
struct ProviderFeatureFusionSelector {
    feature_set_id: String,
    sources: Vec<ProviderFeatureFusionSource>,
    alignment: SampleAlignmentPlan,
    #[serde(default)]
    policy: FeatureFusionPolicy,
}

#[derive(Debug, Deserialize)]
struct ProviderFeatureCollationJsonRequest {
    #[serde(default)]
    feature_set_id: Option<String>,
    #[serde(default)]
    fusion: Option<ProviderFeatureFusionSelector>,
    #[serde(default)]
    policy: CollationPolicy,
}

#[derive(Debug, Deserialize)]
struct ProviderFeatureFusionSource {
    source_id: SourceId,
    feature_set_id: String,
    #[serde(default)]
    columns: Option<Vec<String>>,
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
) -> dag_ml_data_core::Result<NumericFeatureBufferStore> {
    if feature_tables_ptr.is_null() {
        if feature_tables_len != 0 {
            return Err(dag_ml_data_core::DataError::Validation(
                "feature tables pointer is null".to_string(),
            ));
        }
        return Ok(NumericFeatureBufferStore::default());
    }
    if feature_tables_len == 0 {
        return Ok(NumericFeatureBufferStore::default());
    }
    let json = unsafe { slice::from_raw_parts(feature_tables_ptr, feature_tables_len) };
    let tables = serde_json::from_slice::<Vec<CoordinatorFeatureTable>>(json).map_err(|error| {
        dag_ml_data_core::DataError::Validation(format!(
            "failed to parse feature tables JSON: {error}"
        ))
    })?;
    NumericFeatureBufferStore::from_feature_tables(tables)
}

fn parse_f64_feature_matrices(
    feature_matrices_ptr: *const u8,
    feature_matrices_len: usize,
) -> dag_ml_data_core::Result<NumericFeatureBufferStore> {
    if feature_matrices_ptr.is_null() {
        if feature_matrices_len != 0 {
            return Err(dag_ml_data_core::DataError::Validation(
                "f64 feature matrices pointer is null".to_string(),
            ));
        }
        return Ok(NumericFeatureBufferStore::default());
    }
    if feature_matrices_len == 0 {
        return Ok(NumericFeatureBufferStore::default());
    }
    let json = unsafe { slice::from_raw_parts(feature_matrices_ptr, feature_matrices_len) };
    let matrices =
        serde_json::from_slice::<Vec<NumericFeatureMatrixF64>>(json).map_err(|error| {
            dag_ml_data_core::DataError::Validation(format!(
                "failed to parse f64 feature matrices JSON: {error}"
            ))
        })?;
    NumericFeatureBufferStore::from_f64_matrices(matrices)
}

unsafe fn parse_f64_feature_matrix_views(
    feature_matrices_ptr: *const DagMlDataFeatureMatrixF64View,
    feature_matrices_len: usize,
) -> dag_ml_data_core::Result<NumericFeatureBufferStore> {
    if feature_matrices_ptr.is_null() {
        if feature_matrices_len != 0 {
            return Err(dag_ml_data_core::DataError::Validation(
                "f64 feature matrix views pointer is null".to_string(),
            ));
        }
        return Ok(NumericFeatureBufferStore::default());
    }
    if feature_matrices_len == 0 {
        return Ok(NumericFeatureBufferStore::default());
    }
    let views = slice::from_raw_parts(feature_matrices_ptr, feature_matrices_len);
    let matrices = views
        .iter()
        .enumerate()
        .map(|(idx, view)| f64_feature_matrix_view_to_core(*view, idx))
        .collect::<dag_ml_data_core::Result<Vec<_>>>()?;
    NumericFeatureBufferStore::from_f64_matrices(matrices)
}

unsafe fn f64_feature_matrix_view_to_core(
    view: DagMlDataFeatureMatrixF64View,
    index: usize,
) -> dag_ml_data_core::Result<NumericFeatureMatrixF64> {
    let feature_set_id = bytes_view_to_string(
        view.feature_set_id,
        &format!("f64 feature matrix view {index} feature_set_id"),
    )?;
    let representation = bytes_view_to_string(
        view.representation_id,
        &format!("f64 feature matrix view {index} representation_id"),
    )?;
    let representation_id = RepresentationId::new(&representation)?;
    let feature_names = bytes_view_array_to_strings(
        view.feature_names,
        view.feature_names_len,
        &format!("f64 feature matrix view {index} feature_names"),
    )?;
    let observation_ids = bytes_view_array_to_strings(
        view.observation_ids,
        view.observation_ids_len,
        &format!("f64 feature matrix view {index} observation_ids"),
    )?
    .into_iter()
    .map(|observation_id| ObservationId::new(&observation_id))
    .collect::<dag_ml_data_core::Result<Vec<_>>>()?;
    let values = f64_view_to_vec(
        view.values,
        view.values_len,
        &format!("f64 feature matrix view {index} values"),
    )?;
    let validity_mask = u8_validity_mask_to_vec(
        view.validity_mask,
        view.validity_mask_len,
        &format!("f64 feature matrix view {index} validity_mask"),
    )?;
    Ok(NumericFeatureMatrixF64 {
        feature_set_id,
        representation_id,
        feature_names,
        observation_ids,
        values,
        validity_mask,
    })
}

unsafe fn parse_f64_feature_matrix_columnar_views(
    feature_matrices_ptr: *const DagMlDataFeatureMatrixF64ColumnarView,
    feature_matrices_len: usize,
) -> dag_ml_data_core::Result<NumericFeatureBufferStore> {
    if feature_matrices_ptr.is_null() {
        if feature_matrices_len != 0 {
            return Err(dag_ml_data_core::DataError::Validation(
                "f64 columnar feature matrix views pointer is null".to_string(),
            ));
        }
        return Ok(NumericFeatureBufferStore::default());
    }
    if feature_matrices_len == 0 {
        return Ok(NumericFeatureBufferStore::default());
    }
    let views = slice::from_raw_parts(feature_matrices_ptr, feature_matrices_len);
    let matrices = views
        .iter()
        .enumerate()
        .map(|(idx, view)| f64_feature_matrix_columnar_view_to_core(*view, idx))
        .collect::<dag_ml_data_core::Result<Vec<_>>>()?;
    NumericFeatureBufferStore::from_f64_column_matrices(matrices)
}

unsafe fn f64_feature_matrix_columnar_view_to_core(
    view: DagMlDataFeatureMatrixF64ColumnarView,
    index: usize,
) -> dag_ml_data_core::Result<NumericFeatureMatrixF64Columnar> {
    let feature_set_id = bytes_view_to_string(
        view.feature_set_id,
        &format!("f64 columnar feature matrix view {index} feature_set_id"),
    )?;
    let representation = bytes_view_to_string(
        view.representation_id,
        &format!("f64 columnar feature matrix view {index} representation_id"),
    )?;
    let representation_id = RepresentationId::new(&representation)?;
    let feature_names = bytes_view_array_to_strings(
        view.feature_names,
        view.feature_names_len,
        &format!("f64 columnar feature matrix view {index} feature_names"),
    )?;
    let observation_ids = bytes_view_array_to_strings(
        view.observation_ids,
        view.observation_ids_len,
        &format!("f64 columnar feature matrix view {index} observation_ids"),
    )?
    .into_iter()
    .map(|observation_id| ObservationId::new(&observation_id))
    .collect::<dag_ml_data_core::Result<Vec<_>>>()?;
    let (columns, validity_masks) = f64_column_views_to_vecs(
        view.columns,
        view.columns_len,
        &format!("f64 columnar feature matrix view {index}"),
    )?;
    Ok(NumericFeatureMatrixF64Columnar {
        feature_set_id,
        representation_id,
        feature_names,
        observation_ids,
        columns,
        validity_masks,
    })
}

type ColumnarParseOutput = (Vec<Vec<f64>>, Option<Vec<Vec<bool>>>);

unsafe fn f64_column_views_to_vecs(
    columns_ptr: *const DagMlDataF64ColumnView,
    columns_len: usize,
    label: &str,
) -> dag_ml_data_core::Result<ColumnarParseOutput> {
    if columns_ptr.is_null() {
        if columns_len != 0 {
            return Err(dag_ml_data_core::DataError::Validation(format!(
                "{label} columns pointer is null"
            )));
        }
        return Ok((Vec::new(), None));
    }
    let column_views = slice::from_raw_parts(columns_ptr, columns_len);
    let mut columns = Vec::with_capacity(columns_len);
    let mut masks = Vec::with_capacity(columns_len);
    let mut any_mask = false;
    for (idx, column) in column_views.iter().enumerate() {
        let values = f64_view_to_vec(
            column.values,
            column.values_len,
            &format!("{label} column {idx} values"),
        )?;
        columns.push(values);
        let mask = u8_validity_mask_to_vec(
            column.validity_mask,
            column.validity_mask_len,
            &format!("{label} column {idx} validity_mask"),
        )?;
        if let Some(mask) = mask {
            any_mask = true;
            masks.push(mask);
        } else {
            masks.push(Vec::new());
        }
    }
    if any_mask {
        for (idx, (mask, column)) in masks.iter().zip(columns.iter()).enumerate() {
            if mask.is_empty() {
                return Err(dag_ml_data_core::DataError::Validation(format!(
                    "{label} column {idx} validity_mask is missing but other columns supply one; \
                     either all columns must supply a validity_mask or none"
                )));
            }
            if mask.len() != column.len() {
                return Err(dag_ml_data_core::DataError::Validation(format!(
                    "{label} column {idx} validity_mask has {} values for {} rows",
                    mask.len(),
                    column.len()
                )));
            }
        }
        Ok((columns, Some(masks)))
    } else {
        Ok((columns, None))
    }
}

unsafe fn bytes_view_to_string(
    view: DagMlDataBytesView,
    label: &str,
) -> dag_ml_data_core::Result<String> {
    if view.ptr.is_null() {
        return Err(dag_ml_data_core::DataError::Validation(format!(
            "{label} pointer is null"
        )));
    }
    let bytes = slice::from_raw_parts(view.ptr, view.len);
    std::str::from_utf8(bytes)
        .map(str::to_string)
        .map_err(|error| {
            dag_ml_data_core::DataError::Validation(format!("{label} is not valid UTF-8: {error}"))
        })
}

unsafe fn bytes_view_array_to_strings(
    ptr: *const DagMlDataBytesView,
    len: usize,
    label: &str,
) -> dag_ml_data_core::Result<Vec<String>> {
    if ptr.is_null() {
        if len != 0 {
            return Err(dag_ml_data_core::DataError::Validation(format!(
                "{label} pointer is null"
            )));
        }
        return Ok(Vec::new());
    }
    slice::from_raw_parts(ptr, len)
        .iter()
        .enumerate()
        .map(|(idx, view)| bytes_view_to_string(*view, &format!("{label}[{idx}]")))
        .collect()
}

unsafe fn f64_view_to_vec(
    ptr: *const f64,
    len: usize,
    label: &str,
) -> dag_ml_data_core::Result<Vec<f64>> {
    if ptr.is_null() {
        if len != 0 {
            return Err(dag_ml_data_core::DataError::Validation(format!(
                "{label} pointer is null"
            )));
        }
        return Ok(Vec::new());
    }
    Ok(slice::from_raw_parts(ptr, len).to_vec())
}

unsafe fn u8_validity_mask_to_vec(
    ptr: *const u8,
    len: usize,
    label: &str,
) -> dag_ml_data_core::Result<Option<Vec<bool>>> {
    if ptr.is_null() {
        if len != 0 {
            return Err(dag_ml_data_core::DataError::Validation(format!(
                "{label} pointer is null"
            )));
        }
        return Ok(None);
    }
    if len == 0 {
        return Ok(None);
    }
    slice::from_raw_parts(ptr, len)
        .iter()
        .enumerate()
        .map(|(idx, value)| match *value {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(dag_ml_data_core::DataError::Validation(format!(
                "{label}[{idx}] must be 0 or 1"
            ))),
        })
        .collect::<dag_ml_data_core::Result<Vec<_>>>()
        .map(Some)
}

fn provider_vtable(user_data: *mut c_void) -> DagMlDataVTable {
    DagMlDataVTable {
        abi_version: DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION,
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
        abi_version: DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION,
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
    match provider
        .arena
        .materialize(&provider.envelope, &request)
        .and_then(|record| {
            bind_data_feature_buffers(provider, &record)?;
            Ok(record)
        }) {
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
        Ok(feature_set_name) => feature_set_name,
        Err(_) => return DagMlDataStatusCode::ValidationError,
    };
    let result = if feature_set_name.trim_start().starts_with('{') {
        serde_json::from_str::<ProviderFeatureFusionSelector>(feature_set_name)
            .map_err(|error| {
                dag_ml_data_core::DataError::Validation(format!(
                    "failed to parse feature fusion selector JSON: {error}"
                ))
            })
            .and_then(|selector| provider_feature_fusion_block(provider, view, &selector))
            .and_then(|features| build_feature_arrow(&features))
    } else {
        if feature_set_name.trim().is_empty() {
            return DagMlDataStatusCode::ValidationError;
        }
        provider_feature_block(provider, view, feature_set_name)
            .and_then(|features| build_feature_arrow(&features))
    };
    match result {
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
    provider
        .feature_arena
        .borrow_mut()
        .release_data_handle(handle);
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

fn bind_data_feature_buffers(
    provider: &InMemoryProvider,
    record: &CoordinatorDataHandleRecord,
) -> dag_ml_data_core::Result<()> {
    if let Ok(relations) = provider.arena.data_identity(record.handle.handle) {
        provider.feature_arena.borrow_mut().bind_data_handle(
            record.handle.handle,
            &relations,
            &record.output_representation,
        )?;
    }
    Ok(())
}

fn provider_feature_block(
    provider: &InMemoryProvider,
    view_handle: u64,
    feature_set_id: &str,
) -> dag_ml_data_core::Result<CoordinatorFeatureBlock> {
    provider_feature_block_filtered(provider, view_handle, feature_set_id, None, None)
}

fn provider_feature_block_filtered(
    provider: &InMemoryProvider,
    view_handle: u64,
    feature_set_id: &str,
    source_id: Option<&SourceId>,
    columns: Option<&[String]>,
) -> dag_ml_data_core::Result<CoordinatorFeatureBlock> {
    let view_record = provider.arena.view_record(view_handle).ok_or_else(|| {
        dag_ml_data_core::DataError::Validation(format!("unknown view handle `{view_handle}`"))
    })?;
    let parent_record = provider
        .arena
        .handle_record(view_record.parent_handle.handle)
        .ok_or_else(|| {
            dag_ml_data_core::DataError::Validation(format!(
                "view `{view_handle}` parent data handle `{}` is not live",
                view_record.parent_handle.handle
            ))
        })?;
    let relations = provider.arena.view_identity(view_handle)?;
    let selected_columns = if source_id.is_some() {
        columns
    } else {
        columns.or(view_record.view.columns.as_deref())
    };
    provider.feature_arena.borrow().project_bound_relations(
        parent_record.handle.handle,
        feature_set_id,
        &relations,
        source_id,
        selected_columns,
    )
}

fn provider_feature_fusion_block(
    provider: &InMemoryProvider,
    view_handle: u64,
    selector: &ProviderFeatureFusionSelector,
) -> dag_ml_data_core::Result<CoordinatorFeatureBlock> {
    if selector.sources.is_empty() {
        return Err(dag_ml_data_core::DataError::Validation(
            "feature fusion selector requires at least one source".to_string(),
        ));
    }
    let mut sources = Vec::with_capacity(selector.sources.len());
    for source in &selector.sources {
        let block = provider_feature_block_filtered(
            provider,
            view_handle,
            &source.feature_set_id,
            Some(&source.source_id),
            source.columns.as_deref(),
        )?;
        sources.push(SourceFeatureBlock {
            source_id: source.source_id.clone(),
            block,
        });
    }
    fuse_feature_blocks(
        selector.feature_set_id.clone(),
        &sources,
        &selector.alignment,
        &selector.policy,
    )
}

fn provider_feature_collation_block(
    provider: &InMemoryProvider,
    view_handle: u64,
    selector: &ProviderFeatureCollationJsonRequest,
) -> dag_ml_data_core::Result<CoordinatorFeatureBlock> {
    match (&selector.feature_set_id, &selector.fusion) {
        (Some(feature_set_id), None) => {
            if feature_set_id.trim().is_empty() {
                return Err(dag_ml_data_core::DataError::Validation(
                    "feature collation selector feature_set_id is empty".to_string(),
                ));
            }
            provider_feature_block(provider, view_handle, feature_set_id)
        }
        (None, Some(fusion)) => provider_feature_fusion_block(provider, view_handle, fusion),
        (Some(_), Some(_)) => Err(dag_ml_data_core::DataError::Validation(
            "feature collation selector must not specify both feature_set_id and fusion"
                .to_string(),
        )),
        (None, None) => Err(dag_ml_data_core::DataError::Validation(
            "feature collation selector requires feature_set_id or fusion".to_string(),
        )),
    }
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
    fn validates_fitted_adapter_ref_through_abi() {
        let payload = br#"{
  "schema_version": 1,
  "adapter_id": "snv",
  "adapter_version": "1.0.0",
  "params_fingerprint": "12121212121212121212121212121212121212121212121212121212121212ab"
}"#;
        let mut error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_fitted_adapter_ref_validate_json(
                payload.as_ptr(),
                payload.len(),
                0,
                &mut error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert!(error.ptr.is_null());

        let mut portable_error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_fitted_adapter_ref_validate_json(
                payload.as_ptr(),
                payload.len(),
                1,
                &mut portable_error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::ValidationError);
        let message = unsafe { string_value(portable_error) };
        assert!(message.contains("is not portable"), "unexpected: {message}");
    }

    #[test]
    fn fitted_adapter_store_register_materialize_release_round_trips_through_abi() {
        let mut store = DagMlDataFittedAdapterStoreHandle::default();
        let status = unsafe { dagmldata_inmemory_fitted_adapter_store_new(&mut store) };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert!(!store.ptr.is_null());

        let ref_payload = br#"{
  "schema_version": 1,
  "adapter_id": "snv",
  "adapter_version": "1.0.0",
  "params_fingerprint": "12121212121212121212121212121212121212121212121212121212121212ab"
}"#;
        let mut register_handle: u64 = 0;
        let mut error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_fitted_adapter_store_register_json(
                store,
                ref_payload.as_ptr(),
                ref_payload.len(),
                &mut register_handle,
                &mut error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert!(error.ptr.is_null());
        assert_eq!(register_handle, 1);

        let request_payload = br#"{
  "adapter_id": "snv",
  "params_fingerprint": "12121212121212121212121212121212121212121212121212121212121212ab"
}"#;
        let mut materialize_handle: u64 = 0;
        let mut materialize_error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_fitted_adapter_store_materialize_json(
                store,
                request_payload.as_ptr(),
                request_payload.len(),
                &mut materialize_handle,
                &mut materialize_error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert!(materialize_error.ptr.is_null());
        assert_eq!(materialize_handle, register_handle);

        let bad_request = br#"{
  "adapter_id": "snv",
  "params_fingerprint": "5555555555555555555555555555555555555555555555555555555555555555"
}"#;
        let mut bad_handle: u64 = 0;
        let mut bad_error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_fitted_adapter_store_materialize_json(
                store,
                bad_request.as_ptr(),
                bad_request.len(),
                &mut bad_handle,
                &mut bad_error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::ValidationError);
        let message = unsafe { string_value(bad_error) };
        assert!(
            message.contains("params fingerprint mismatch"),
            "unexpected: {message}"
        );

        let mut released: u8 = 0;
        let status = unsafe {
            dagmldata_inmemory_fitted_adapter_store_release(
                store,
                b"snv".as_ptr(),
                3,
                &mut released,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert_eq!(released, 1);

        unsafe { dagmldata_inmemory_fitted_adapter_store_destroy(store) };
    }

    #[test]
    fn provider_attached_fitted_adapter_store_materializes_handles() {
        let envelope = include_bytes!(
            "../../../examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json"
        );
        let mut provider_vtable = empty_vtable();
        let mut error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_provider_new_json(
                envelope.as_ptr(),
                envelope.len(),
                b"[]".as_ptr(),
                2,
                &mut provider_vtable,
                &mut error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);

        let mut store = DagMlDataFittedAdapterStoreHandle::default();
        let status = unsafe { dagmldata_inmemory_fitted_adapter_store_new(&mut store) };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert!(!store.ptr.is_null());

        let ref_payload = br#"{
  "schema_version": 1,
  "adapter_id": "snv",
  "adapter_version": "1.0.0",
  "params_fingerprint": "12121212121212121212121212121212121212121212121212121212121212ab"
}"#;
        let mut adapter_handle: u64 = 0;
        let status = unsafe {
            dagmldata_inmemory_fitted_adapter_store_register_json(
                store,
                ref_payload.as_ptr(),
                ref_payload.len(),
                &mut adapter_handle,
                &mut error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert_eq!(adapter_handle, 1);

        // Materializing through the provider BEFORE attaching the store is rejected.
        let request_payload = br#"{
  "adapter_id": "snv",
  "params_fingerprint": "12121212121212121212121212121212121212121212121212121212121212ab"
}"#;
        let mut materialize_handle: u64 = 0;
        let status = unsafe {
            dagmldata_inmemory_provider_materialize_fitted_adapter_json(
                &provider_vtable,
                request_payload.as_ptr(),
                request_payload.len(),
                &mut materialize_handle,
                &mut error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::InvalidArgument);
        let message = unsafe { string_value(error) };
        assert!(
            message.contains("no fitted-adapter store attached"),
            "unexpected: {message}"
        );

        // Attach the store, then materialize through the provider succeeds.
        let mut attach_error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_provider_attach_fitted_adapter_store(
                &provider_vtable,
                store,
                &mut attach_error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert!(attach_error.ptr.is_null());

        let mut materialize_error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_provider_materialize_fitted_adapter_json(
                &provider_vtable,
                request_payload.as_ptr(),
                request_payload.len(),
                &mut materialize_handle,
                &mut materialize_error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert_eq!(materialize_handle, 1);
        assert!(materialize_error.ptr.is_null());

        // Detach by passing a null handle.
        let null_handle = DagMlDataFittedAdapterStoreHandle::default();
        let mut detach_error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_provider_attach_fitted_adapter_store(
                &provider_vtable,
                null_handle,
                &mut detach_error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);

        let mut detached_error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_provider_materialize_fitted_adapter_json(
                &provider_vtable,
                request_payload.as_ptr(),
                request_payload.len(),
                &mut materialize_handle,
                &mut detached_error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::InvalidArgument);
        let detached_message = unsafe { string_value(detached_error) };
        assert!(detached_message.contains("no fitted-adapter store attached"));

        unsafe {
            dagmldata_inmemory_fitted_adapter_store_destroy(store);
            dagmldata_inmemory_provider_destroy(&mut provider_vtable);
        }
    }

    #[test]
    fn rejects_bad_fitted_adapter_manifest_through_abi() {
        let payload = br#"{
  "schema_version": 1,
  "entries": [
    {
      "adapter_id": "snv",
      "fitted_adapter": {
        "schema_version": 1,
        "adapter_id": "snv",
        "adapter_version": "1.0.0",
        "params_fingerprint": "bad"
      }
    }
  ]
}"#;
        let mut error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_fitted_adapter_manifest_validate_json(
                payload.as_ptr(),
                payload.len(),
                0,
                &mut error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::ValidationError);
        let message = unsafe { string_value(error) };
        assert!(
            message.contains("params fingerprint"),
            "unexpected: {message}"
        );
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
    fn exports_coordinator_feature_fusion_arrow_over_abi() {
        let request = serde_json::json!({
            "schema_version": 1,
            "feature_set_id": "fused",
            "sources": [
                {
                    "source_id": "nir",
                    "block": {
                        "feature_set_id": "nir_x",
                        "representation_id": "tabular_numeric",
                        "feature_names": ["n0"],
                        "observation_ids": ["obs.S001.r1", "obs.S001.r2", "obs.S002.r1"],
                        "sample_ids": ["S001", "S001", "S002"],
                        "values": [[1.0], [2.0], [3.0]]
                    }
                },
                {
                    "source_id": "chem",
                    "block": {
                        "feature_set_id": "chem_x",
                        "representation_id": "tabular_numeric",
                        "feature_names": ["c0"],
                        "observation_ids": ["chem.S001", "chem.S002"],
                        "sample_ids": ["S001", "S002"],
                        "values": [[10.0], [20.0]]
                    }
                }
            ],
            "alignment": {
                "mode": "inner",
                "sample_ids": ["S001", "S002"],
                "masks": [
                    {"source_id": "nir", "sample_ids": ["S001", "S002"], "present": [true, true]},
                    {"source_id": "chem", "sample_ids": ["S001", "S002"], "present": [true, true]}
                ]
            }
        });
        let request = serde_json::to_vec(&request).unwrap();
        let mut array = std::ptr::null_mut();
        let mut schema = std::ptr::null_mut();
        let mut error = DagMlDataString::default();

        let status = unsafe {
            dagmldata_coordinator_feature_fusion_arrow_json(
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
            assert_eq!((*array).length, 3);
            assert_eq!((*array).n_children, 4);
            let array_children =
                slice::from_raw_parts((*array).children, (*array).n_children as usize);
            let schema_children =
                slice::from_raw_parts((*schema).children, (*schema).n_children as usize);
            assert_eq!(
                utf8_values(array_children[0]),
                vec![
                    Some("obs.S001.r1".to_string()),
                    Some("obs.S001.r2".to_string()),
                    Some("obs.S002.r1".to_string()),
                ]
            );
            assert_eq!(
                CStr::from_ptr((*schema_children[2]).name).to_str().unwrap(),
                "nir.n0"
            );
            assert_eq!(
                CStr::from_ptr((*schema_children[3]).name).to_str().unwrap(),
                "chem.c0"
            );
            assert_eq!(
                f64_values(array_children[2]),
                vec![Some(1.0), Some(2.0), Some(3.0)]
            );
            assert_eq!(
                f64_values(array_children[3]),
                vec![Some(10.0), Some(10.0), Some(20.0)]
            );
            dagmldata_arrow_array_free(array);
            dagmldata_arrow_schema_free(schema);
        }
    }

    #[test]
    fn exports_coordinator_feature_collation_json_over_abi() {
        let request = serde_json::json!({
            "feature_block": {
                "feature_set_id": "x",
                "representation_id": "tabular_numeric",
                "feature_names": ["f0", "f1"],
                "observation_ids": ["obs.S001", "obs.S002"],
                "sample_ids": ["S001", "S002"],
                "values": [[1.0, null], [3.0, 4.0]]
            },
            "policy": {
                "emit_mask": true
            }
        });
        let request = serde_json::to_vec(&request).unwrap();
        let mut out_json = DagMlDataString::default();
        let mut error = DagMlDataString::default();

        let status = unsafe {
            dagmldata_coordinator_feature_collation_json(
                request.as_ptr(),
                request.len(),
                &mut out_json,
                &mut error,
            )
        };

        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert!(error.ptr.is_null());
        assert!(!out_json.ptr.is_null());
        let tensor_json = unsafe { string_value(out_json) };
        let tensor: serde_json::Value = serde_json::from_str(&tensor_json).unwrap();
        assert_eq!(tensor["shape"], serde_json::json!([2, 2]));
        assert_eq!(tensor["values"], serde_json::json!([1.0, 0.0, 3.0, 4.0]));
        assert_eq!(
            tensor["presence_mask"],
            serde_json::json!([true, true, true, true])
        );
        assert_eq!(
            tensor["validity_mask"],
            serde_json::json!([true, false, true, true])
        );
    }

    #[test]
    fn exports_coordinator_feature_collation_tensor_f64_over_abi() {
        let request = serde_json::json!({
            "feature_block": {
                "feature_set_id": "x",
                "representation_id": "tabular_numeric",
                "feature_names": ["f0", "f1"],
                "observation_ids": ["obs.S001", "obs.S002"],
                "sample_ids": ["S001", "S002"],
                "values": [[1.0, null], [3.0, 4.0]]
            },
            "policy": {
                "emit_mask": true
            }
        });
        let request = serde_json::to_vec(&request).unwrap();
        let mut tensor = DagMlDataTensorF64::default();
        let mut error = DagMlDataString::default();

        let status = unsafe {
            dagmldata_coordinator_feature_collation_tensor_f64_json(
                request.as_ptr(),
                request.len(),
                &mut tensor,
                &mut error,
            )
        };

        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert!(error.ptr.is_null());
        unsafe {
            assert_eq!(tensor.abi_version, DAG_ML_DATA_TENSOR_F64_ABI_VERSION);
            assert_eq!(borrowed_string(&tensor.block_id), "x");
            assert_eq!(
                borrowed_string(&tensor.representation_id),
                "tabular_numeric"
            );
            assert_eq!(
                string_array_values(tensor.observation_ids),
                vec!["obs.S001", "obs.S002"]
            );
            assert_eq!(string_array_values(tensor.sample_ids), vec!["S001", "S002"]);
            assert_eq!(usize_array_values(tensor.shape), vec![2, 2]);
            assert_eq!(f64_array_values(tensor.values), vec![1.0, 0.0, 3.0, 4.0]);
            assert_eq!(u8_array_values(tensor.presence_mask), vec![1, 1, 1, 1]);
            assert_eq!(u8_array_values(tensor.validity_mask), vec![1, 0, 1, 1]);
            assert_eq!(string_array_values(tensor.feature_names), vec!["f0", "f1"]);
            dagmldata_tensor_f64_free(tensor);
        }
    }

    #[test]
    fn exports_coordinator_feature_collation_tensor_f32_over_abi() {
        let request = serde_json::json!({
            "feature_block": {
                "feature_set_id": "x",
                "representation_id": "tabular_numeric",
                "feature_names": ["f0", "f1"],
                "observation_ids": ["obs.S001", "obs.S002"],
                "sample_ids": ["S001", "S002"],
                "values": [[1.0, null], [3.0, 4.5]]
            },
            "policy": {
                "emit_mask": true
            }
        });
        let request = serde_json::to_vec(&request).unwrap();
        let mut tensor = DagMlDataTensorF32::default();
        let mut error = DagMlDataString::default();

        let status = unsafe {
            dagmldata_coordinator_feature_collation_tensor_f32_json(
                request.as_ptr(),
                request.len(),
                &mut tensor,
                &mut error,
            )
        };

        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert!(error.ptr.is_null());
        unsafe {
            assert_eq!(tensor.abi_version, DAG_ML_DATA_TENSOR_F32_ABI_VERSION);
            assert_eq!(borrowed_string(&tensor.block_id), "x");
            assert_eq!(
                borrowed_string(&tensor.representation_id),
                "tabular_numeric"
            );
            assert_eq!(
                string_array_values(tensor.observation_ids),
                vec!["obs.S001", "obs.S002"]
            );
            assert_eq!(string_array_values(tensor.sample_ids), vec!["S001", "S002"]);
            assert_eq!(usize_array_values(tensor.shape), vec![2, 2]);
            assert_eq!(
                f32_array_values(tensor.values),
                vec![1.0_f32, 0.0, 3.0, 4.5]
            );
            assert_eq!(u8_array_values(tensor.presence_mask), vec![1, 1, 1, 1]);
            assert_eq!(u8_array_values(tensor.validity_mask), vec![1, 0, 1, 1]);
            assert_eq!(string_array_values(tensor.feature_names), vec!["f0", "f1"]);
            dagmldata_tensor_f32_free(tensor);
        }

        let overflow_request = serde_json::json!({
            "feature_block": {
                "feature_set_id": "x",
                "representation_id": "tabular_numeric",
                "feature_names": ["f0"],
                "observation_ids": ["obs.S001"],
                "sample_ids": ["S001"],
                "values": [[1.0e40]]
            },
            "policy": {"emit_mask": false}
        });
        let overflow_request = serde_json::to_vec(&overflow_request).unwrap();
        let mut overflow_tensor = DagMlDataTensorF32::default();
        let mut overflow_error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_coordinator_feature_collation_tensor_f32_json(
                overflow_request.as_ptr(),
                overflow_request.len(),
                &mut overflow_tensor,
                &mut overflow_error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::ValidationError);
        assert_eq!(
            overflow_tensor.abi_version,
            DAG_ML_DATA_TENSOR_F32_ABI_VERSION
        );
        assert!(overflow_tensor.values.ptr.is_null());
        let overflow_message = unsafe { string_value(overflow_error) };
        assert!(
            overflow_message.contains("does not round-trip into a finite f32"),
            "unexpected overflow error: {overflow_message}"
        );
    }

    #[test]
    fn inmemory_provider_feature_arrow_accepts_fusion_selector_json() {
        let (envelope, materialization_request) = multisource_provider_fixture();
        let target_tables = b"[]";
        let feature_tables = serde_json::to_vec(&serde_json::json!([
            {
                "feature_set_id": "nir_x",
                "representation_id": "tabular_numeric",
                "feature_names": ["n0"],
                "rows": [
                    {"observation_id": "obs.S001.r1", "values": [1.0]},
                    {"observation_id": "obs.S001.r2", "values": [2.0]},
                    {"observation_id": "obs.S002.r1", "values": [3.0]}
                ]
            },
            {
                "feature_set_id": "chem_x",
                "representation_id": "tabular_numeric",
                "feature_names": ["c0"],
                "rows": [
                    {"observation_id": "chem.S001", "values": [10.0]},
                    {"observation_id": "chem.S002", "values": [20.0]}
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

        let view_json = serde_json::to_vec(&serde_json::json!({
            "sample_ids": ["S001", "S002"],
            "include_augmented": false
        }))
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

        let fusion_selector = serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "feature_set_id": "fused",
            "sources": [
                {"source_id": "nir", "feature_set_id": "nir_x"},
                {"source_id": "chem", "feature_set_id": "chem_x"}
            ],
            "alignment": {
                "mode": "inner",
                "sample_ids": ["S001", "S002"],
                "masks": [
                    {"source_id": "nir", "sample_ids": ["S001", "S002"], "present": [true, true]},
                    {"source_id": "chem", "sample_ids": ["S001", "S002"], "present": [true, true]}
                ]
            }
        }))
        .unwrap();
        let mut feature_array = std::ptr::null_mut();
        let mut feature_schema = std::ptr::null_mut();
        let feature_arrow = vtable.feature_arrow.unwrap();
        let status = unsafe {
            feature_arrow(
                vtable.user_data,
                view_handle,
                DagMlDataBytesView {
                    ptr: fusion_selector.as_ptr(),
                    len: fusion_selector.len(),
                },
                &mut feature_array,
                &mut feature_schema,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        unsafe {
            assert_eq!((*feature_array).length, 3);
            assert_eq!((*feature_array).n_children, 4);
            let array_children = slice::from_raw_parts(
                (*feature_array).children,
                (*feature_array).n_children as usize,
            );
            let schema_children = slice::from_raw_parts(
                (*feature_schema).children,
                (*feature_schema).n_children as usize,
            );
            assert_eq!(
                utf8_values(array_children[0]),
                vec![
                    Some("obs.S001.r1".to_string()),
                    Some("obs.S001.r2".to_string()),
                    Some("obs.S002.r1".to_string()),
                ]
            );
            assert_eq!(
                CStr::from_ptr((*schema_children[2]).name).to_str().unwrap(),
                "nir.n0"
            );
            assert_eq!(
                CStr::from_ptr((*schema_children[3]).name).to_str().unwrap(),
                "chem.c0"
            );
            assert_eq!(
                f64_values(array_children[2]),
                vec![Some(1.0), Some(2.0), Some(3.0)]
            );
            assert_eq!(
                f64_values(array_children[3]),
                vec![Some(10.0), Some(10.0), Some(20.0)]
            );
            dagmldata_arrow_array_free(feature_array);
            dagmldata_arrow_schema_free(feature_schema);
            vtable.release.unwrap()(vtable.user_data, view_handle);
            vtable.release.unwrap()(vtable.user_data, data_handle);
            dagmldata_inmemory_provider_destroy(&mut vtable);
        }
    }

    #[test]
    fn inmemory_provider_feature_collation_uses_provider_buffers_and_fusion_selector() {
        let (envelope, materialization_request) = multisource_provider_fixture();
        let target_tables = b"[]";
        let feature_tables = serde_json::to_vec(&serde_json::json!([
            {
                "feature_set_id": "nir_x",
                "representation_id": "tabular_numeric",
                "feature_names": ["n0"],
                "rows": [
                    {"observation_id": "obs.S001.r1", "values": [1.0]},
                    {"observation_id": "obs.S001.r2", "values": [2.0]},
                    {"observation_id": "obs.S002.r1", "values": [3.0]}
                ]
            },
            {
                "feature_set_id": "chem_x",
                "representation_id": "tabular_numeric",
                "feature_names": ["c0"],
                "rows": [
                    {"observation_id": "chem.S001", "values": [10.0]},
                    {"observation_id": "chem.S002", "values": [20.0]}
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

        let mut data_handle = 0;
        let status = unsafe {
            vtable.materialize.unwrap()(
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

        let view_json = serde_json::to_vec(&serde_json::json!({
            "sample_ids": ["S001", "S002"],
            "include_augmented": false
        }))
        .unwrap();
        let mut view_handle = 0;
        let status = unsafe {
            vtable.make_view.unwrap()(
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

        let selector = serde_json::to_vec(&serde_json::json!({
            "fusion": {
                "schema_version": 1,
                "feature_set_id": "fused",
                "sources": [
                    {"source_id": "nir", "feature_set_id": "nir_x"},
                    {"source_id": "chem", "feature_set_id": "chem_x"}
                ],
                "alignment": {
                    "mode": "inner",
                    "sample_ids": ["S001", "S002"],
                    "masks": [
                        {"source_id": "nir", "sample_ids": ["S001", "S002"], "present": [true, true]},
                        {"source_id": "chem", "sample_ids": ["S001", "S002"], "present": [true, true]}
                    ]
                }
            },
            "policy": {
                "emit_mask": true
            }
        }))
        .unwrap();
        let mut out_json = DagMlDataString::default();
        let mut error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_provider_feature_collation_json(
                &vtable,
                view_handle,
                DagMlDataBytesView {
                    ptr: selector.as_ptr(),
                    len: selector.len(),
                },
                &mut out_json,
                &mut error,
            )
        };

        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert!(error.ptr.is_null());
        let tensor_json = unsafe { string_value(out_json) };
        let tensor: serde_json::Value = serde_json::from_str(&tensor_json).unwrap();
        assert_eq!(tensor["shape"], serde_json::json!([3, 2]));
        assert_eq!(
            tensor["observation_ids"],
            serde_json::json!(["obs.S001.r1", "obs.S001.r2", "obs.S002.r1"])
        );
        assert_eq!(
            tensor["values"],
            serde_json::json!([1.0, 10.0, 2.0, 10.0, 3.0, 20.0])
        );
        assert_eq!(
            tensor["feature_names"],
            serde_json::json!(["nir.n0", "chem.c0"])
        );
        assert_eq!(
            tensor["presence_mask"],
            serde_json::json!([true, true, true, true, true, true])
        );

        let mut tensor = DagMlDataTensorF64::default();
        let mut error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_provider_feature_collation_tensor_f64_json(
                &vtable,
                view_handle,
                DagMlDataBytesView {
                    ptr: selector.as_ptr(),
                    len: selector.len(),
                },
                &mut tensor,
                &mut error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert!(error.ptr.is_null());
        unsafe {
            assert_eq!(usize_array_values(tensor.shape), vec![3, 2]);
            assert_eq!(
                string_array_values(tensor.observation_ids),
                vec!["obs.S001.r1", "obs.S001.r2", "obs.S002.r1"]
            );
            assert_eq!(
                f64_array_values(tensor.values),
                vec![1.0, 10.0, 2.0, 10.0, 3.0, 20.0]
            );
            assert_eq!(
                string_array_values(tensor.feature_names),
                vec!["nir.n0", "chem.c0"]
            );
            assert_eq!(
                u8_array_values(tensor.presence_mask),
                vec![1, 1, 1, 1, 1, 1]
            );
            dagmldata_tensor_f64_free(tensor);
        }

        unsafe {
            vtable.release.unwrap()(vtable.user_data, view_handle);
            vtable.release.unwrap()(vtable.user_data, data_handle);
            dagmldata_inmemory_provider_destroy(&mut vtable);
        }
    }

    #[test]
    fn inmemory_provider_refuses_unbound_feature_buffers_for_source_scoped_handle() {
        let (envelope, materialization_request) = multisource_provider_fixture();
        let mut materialization_request: serde_json::Value =
            serde_json::from_slice(&materialization_request).unwrap();
        materialization_request["source_ids"] = serde_json::json!(["nir"]);
        let materialization_request = serde_json::to_vec(&materialization_request).unwrap();
        let target_tables = b"[]";
        let feature_tables = serde_json::to_vec(&serde_json::json!([
            {
                "feature_set_id": "nir_x",
                "representation_id": "tabular_numeric",
                "feature_names": ["n0"],
                "rows": [
                    {"observation_id": "obs.S001.r1", "values": [1.0]},
                    {"observation_id": "obs.S001.r2", "values": [2.0]},
                    {"observation_id": "obs.S002.r1", "values": [3.0]}
                ]
            },
            {
                "feature_set_id": "chem_x",
                "representation_id": "tabular_numeric",
                "feature_names": ["c0"],
                "rows": [
                    {"observation_id": "chem.S001", "values": [10.0]},
                    {"observation_id": "chem.S002", "values": [20.0]}
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

        let mut data_handle = 0;
        let status = unsafe {
            vtable.materialize.unwrap()(
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

        let mut data_manifest_json = DagMlDataString::default();
        let mut data_manifest_error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_provider_data_feature_buffer_manifest_json(
                &vtable,
                data_handle,
                &mut data_manifest_json,
                &mut data_manifest_error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert!(data_manifest_error.ptr.is_null());
        let manifests: serde_json::Value =
            serde_json::from_str(&unsafe { string_value(data_manifest_json) }).unwrap();
        assert_eq!(manifests.as_array().unwrap().len(), 1);
        assert_eq!(manifests[0]["feature_set_id"], serde_json::json!("nir_x"));
        assert_eq!(manifests[0]["source_ids"], serde_json::json!(["nir"]));

        let view_json = serde_json::to_vec(&serde_json::json!({
            "sample_ids": ["S001", "S002"],
            "include_augmented": false
        }))
        .unwrap();
        let mut view_handle = 0;
        let status = unsafe {
            vtable.make_view.unwrap()(
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

        let mut chem_array = std::ptr::null_mut();
        let mut chem_schema = std::ptr::null_mut();
        let feature_set_name = b"chem_x";
        let status = unsafe {
            vtable.feature_arrow.unwrap()(
                vtable.user_data,
                view_handle,
                DagMlDataBytesView {
                    ptr: feature_set_name.as_ptr(),
                    len: feature_set_name.len(),
                },
                &mut chem_array,
                &mut chem_schema,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::ValidationError);
        assert!(chem_array.is_null());
        assert!(chem_schema.is_null());

        let selector = serde_json::to_vec(&serde_json::json!({
            "fusion": {
                "schema_version": 1,
                "feature_set_id": "bad_fused",
                "sources": [
                    {"source_id": "chem", "feature_set_id": "chem_x"}
                ],
                "alignment": {
                    "mode": "inner",
                    "sample_ids": ["S001", "S002"],
                    "masks": [
                        {"source_id": "chem", "sample_ids": ["S001", "S002"], "present": [true, true]}
                    ]
                }
            },
            "policy": {
                "emit_mask": true
            }
        }))
        .unwrap();
        let mut out_json = DagMlDataString::default();
        let mut error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_provider_feature_collation_json(
                &vtable,
                view_handle,
                DagMlDataBytesView {
                    ptr: selector.as_ptr(),
                    len: selector.len(),
                },
                &mut out_json,
                &mut error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::ValidationError);
        assert!(out_json.ptr.is_null());
        unsafe {
            dagmldata_string_free(error);
        }

        let mut tensor = DagMlDataTensorF64::default();
        let mut error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_provider_feature_collation_tensor_f64_json(
                &vtable,
                view_handle,
                DagMlDataBytesView {
                    ptr: selector.as_ptr(),
                    len: selector.len(),
                },
                &mut tensor,
                &mut error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::ValidationError);
        assert!(tensor.values.ptr.is_null());
        unsafe {
            dagmldata_string_free(error);
            vtable.release.unwrap()(vtable.user_data, view_handle);
            vtable.release.unwrap()(vtable.user_data, data_handle);
            dagmldata_inmemory_provider_destroy(&mut vtable);
        }
    }

    #[test]
    fn inmemory_provider_accepts_typed_f64_feature_matrices() {
        let envelope = include_bytes!(
            "../../../examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json"
        );
        let materialization_request = include_bytes!(
            "../../../examples/fixtures/oof_campaign/materialization_request_model_base_x.json"
        );
        let target_tables = b"[]";
        let feature_matrices = serde_json::to_vec(&serde_json::json!([
            {
                "feature_set_id": "x",
                "representation_id": "tabular_numeric",
                "feature_names": ["f0", "f1"],
                "observation_ids": [
                    "obs.S001.base",
                    "obs.S001.rep1",
                    "obs.S001.aug0",
                    "obs.S002.base"
                ],
                "values": [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 0.0],
                "validity_mask": [true, true, true, true, true, true, true, false]
            }
        ]))
        .unwrap();
        let mut vtable = empty_vtable();
        let mut error = DagMlDataString::default();

        let status = unsafe {
            dagmldata_inmemory_provider_new_with_f64_features_json(
                envelope.as_ptr(),
                envelope.len(),
                target_tables.as_ptr(),
                target_tables.len(),
                feature_matrices.as_ptr(),
                feature_matrices.len(),
                &mut vtable,
                &mut error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert!(error.ptr.is_null());

        let mut data_handle = 0;
        let status = unsafe {
            vtable.materialize.unwrap()(
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

        let view_json = serde_json::to_vec(&serde_json::json!({
            "sample_ids": ["S002", "S001"],
            "include_augmented": false
        }))
        .unwrap();
        let mut view_handle = 0;
        let status = unsafe {
            vtable.make_view.unwrap()(
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

        let mut feature_array = std::ptr::null_mut();
        let mut feature_schema = std::ptr::null_mut();
        let feature_set_name = b"x";
        let status = unsafe {
            vtable.feature_arrow.unwrap()(
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
            let array_children = slice::from_raw_parts(
                (*feature_array).children,
                (*feature_array).n_children as usize,
            );
            assert_eq!(
                utf8_values(array_children[0]),
                vec![
                    Some("obs.S002.base".to_string()),
                    Some("obs.S001.base".to_string()),
                    Some("obs.S001.rep1".to_string()),
                ]
            );
            assert_eq!(
                f64_values(array_children[2]),
                vec![Some(4.0), Some(1.0), Some(2.0)]
            );
            assert_eq!(
                f64_values(array_children[3]),
                vec![None, Some(10.0), Some(20.0)]
            );
            dagmldata_arrow_array_free(feature_array);
            dagmldata_arrow_schema_free(feature_schema);
        }

        let selector = serde_json::to_vec(&serde_json::json!({
            "feature_set_id": "x",
            "policy": {
                "emit_mask": true
            }
        }))
        .unwrap();
        let mut tensor = DagMlDataTensorF64::default();
        let mut error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_provider_feature_collation_tensor_f64_json(
                &vtable,
                view_handle,
                DagMlDataBytesView {
                    ptr: selector.as_ptr(),
                    len: selector.len(),
                },
                &mut tensor,
                &mut error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert!(error.ptr.is_null());
        unsafe {
            assert_eq!(usize_array_values(tensor.shape), vec![3, 2]);
            assert_eq!(
                f64_array_values(tensor.values),
                vec![4.0, 0.0, 1.0, 10.0, 2.0, 20.0]
            );
            assert_eq!(
                u8_array_values(tensor.validity_mask),
                vec![1, 0, 1, 1, 1, 1]
            );
            dagmldata_tensor_f64_free(tensor);
            vtable.release.unwrap()(vtable.user_data, view_handle);
            vtable.release.unwrap()(vtable.user_data, data_handle);
            dagmldata_inmemory_provider_destroy(&mut vtable);
        }
    }

    #[test]
    fn inmemory_provider_accepts_borrowed_f64_feature_matrix_views() {
        let envelope = include_bytes!(
            "../../../examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json"
        );
        let materialization_request = include_bytes!(
            "../../../examples/fixtures/oof_campaign/materialization_request_model_base_x.json"
        );
        let target_tables = b"[]";
        let feature_names = [bytes_view(b"f0"), bytes_view(b"f1")];
        let observation_ids = [
            bytes_view(b"obs.S001.base"),
            bytes_view(b"obs.S001.rep1"),
            bytes_view(b"obs.S001.aug0"),
            bytes_view(b"obs.S002.base"),
        ];
        let values = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 0.0];
        let validity_mask = [1_u8, 1, 1, 1, 1, 1, 1, 0];
        let matrices = [DagMlDataFeatureMatrixF64View {
            feature_set_id: bytes_view(b"x"),
            representation_id: bytes_view(b"tabular_numeric"),
            feature_names: feature_names.as_ptr(),
            feature_names_len: feature_names.len(),
            observation_ids: observation_ids.as_ptr(),
            observation_ids_len: observation_ids.len(),
            values: values.as_ptr(),
            values_len: values.len(),
            validity_mask: validity_mask.as_ptr(),
            validity_mask_len: validity_mask.len(),
        }];
        let mut vtable = empty_vtable();
        let mut error = DagMlDataString::default();

        let status = unsafe {
            dagmldata_inmemory_provider_new_with_f64_feature_views(
                envelope.as_ptr(),
                envelope.len(),
                target_tables.as_ptr(),
                target_tables.len(),
                matrices.as_ptr(),
                matrices.len(),
                &mut vtable,
                &mut error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert!(error.ptr.is_null());

        let mut data_handle = 0;
        let status = unsafe {
            vtable.materialize.unwrap()(
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

        let mut manifest_json = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_provider_data_feature_buffer_manifest_json(
                &vtable,
                data_handle,
                &mut manifest_json,
                &mut error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        let manifests: serde_json::Value =
            serde_json::from_str(&unsafe { string_value(manifest_json) }).unwrap();
        assert_eq!(manifests[0]["feature_set_id"], serde_json::json!("x"));
        assert_eq!(manifests[0]["source_ids"], serde_json::json!(["nir"]));

        let mut invalid_vtable = empty_vtable();
        let invalid_mask = [2_u8];
        let invalid = [DagMlDataFeatureMatrixF64View {
            validity_mask: invalid_mask.as_ptr(),
            validity_mask_len: invalid_mask.len(),
            ..matrices[0]
        }];
        let status = unsafe {
            dagmldata_inmemory_provider_new_with_f64_feature_views(
                envelope.as_ptr(),
                envelope.len(),
                target_tables.as_ptr(),
                target_tables.len(),
                invalid.as_ptr(),
                invalid.len(),
                &mut invalid_vtable,
                &mut error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::ValidationError);
        assert!(invalid_vtable.user_data.is_null());
        unsafe {
            dagmldata_string_free(error);
            vtable.release.unwrap()(vtable.user_data, data_handle);
            dagmldata_inmemory_provider_destroy(&mut vtable);
        }
    }

    #[test]
    fn inmemory_provider_accepts_borrowed_f64_feature_matrix_columnar_views() {
        let envelope = include_bytes!(
            "../../../examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json"
        );
        let materialization_request = include_bytes!(
            "../../../examples/fixtures/oof_campaign/materialization_request_model_base_x.json"
        );
        let target_tables = b"[]";
        let feature_names = [bytes_view(b"f0"), bytes_view(b"f1")];
        let observation_ids = [
            bytes_view(b"obs.S001.base"),
            bytes_view(b"obs.S001.rep1"),
            bytes_view(b"obs.S001.aug0"),
            bytes_view(b"obs.S002.base"),
        ];
        let f0_values = [1.0_f64, 2.0, 3.0, 4.0];
        let f1_values = [10.0_f64, 20.0, 30.0, 0.0];
        let f0_mask = [1_u8, 1, 1, 1];
        let f1_mask = [1_u8, 1, 1, 0];
        let columns = [
            DagMlDataF64ColumnView {
                values: f0_values.as_ptr(),
                values_len: f0_values.len(),
                validity_mask: f0_mask.as_ptr(),
                validity_mask_len: f0_mask.len(),
            },
            DagMlDataF64ColumnView {
                values: f1_values.as_ptr(),
                values_len: f1_values.len(),
                validity_mask: f1_mask.as_ptr(),
                validity_mask_len: f1_mask.len(),
            },
        ];
        let matrices = [DagMlDataFeatureMatrixF64ColumnarView {
            feature_set_id: bytes_view(b"x"),
            representation_id: bytes_view(b"tabular_numeric"),
            feature_names: feature_names.as_ptr(),
            feature_names_len: feature_names.len(),
            observation_ids: observation_ids.as_ptr(),
            observation_ids_len: observation_ids.len(),
            columns: columns.as_ptr(),
            columns_len: columns.len(),
        }];
        let mut vtable = empty_vtable();
        let mut error = DagMlDataString::default();

        let status = unsafe {
            dagmldata_inmemory_provider_new_with_f64_feature_columns(
                envelope.as_ptr(),
                envelope.len(),
                target_tables.as_ptr(),
                target_tables.len(),
                matrices.as_ptr(),
                matrices.len(),
                &mut vtable,
                &mut error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert!(error.ptr.is_null());

        let mut data_handle = 0;
        let status = unsafe {
            vtable.materialize.unwrap()(
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

        let mut manifest_json = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_provider_data_feature_buffer_manifest_json(
                &vtable,
                data_handle,
                &mut manifest_json,
                &mut error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        let manifests: serde_json::Value =
            serde_json::from_str(&unsafe { string_value(manifest_json) }).unwrap();
        assert_eq!(manifests[0]["feature_set_id"], serde_json::json!("x"));
        assert_eq!(manifests[0]["source_ids"], serde_json::json!(["nir"]));

        let mut invalid_vtable = empty_vtable();
        let short_column = [1.0_f64, 2.0];
        let invalid_columns = [DagMlDataF64ColumnView {
            values: short_column.as_ptr(),
            values_len: short_column.len(),
            validity_mask: std::ptr::null(),
            validity_mask_len: 0,
        }];
        let invalid = [DagMlDataFeatureMatrixF64ColumnarView {
            feature_set_id: bytes_view(b"x"),
            representation_id: bytes_view(b"tabular_numeric"),
            feature_names: feature_names.as_ptr(),
            feature_names_len: 1,
            observation_ids: observation_ids.as_ptr(),
            observation_ids_len: observation_ids.len(),
            columns: invalid_columns.as_ptr(),
            columns_len: invalid_columns.len(),
        }];
        let status = unsafe {
            dagmldata_inmemory_provider_new_with_f64_feature_columns(
                envelope.as_ptr(),
                envelope.len(),
                target_tables.as_ptr(),
                target_tables.len(),
                invalid.as_ptr(),
                invalid.len(),
                &mut invalid_vtable,
                &mut error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::ValidationError);
        assert!(invalid_vtable.user_data.is_null());

        let mut invalid_mask_vtable = empty_vtable();
        let bad_mask = [2_u8, 1, 1, 1];
        let bad_mask_columns = [
            DagMlDataF64ColumnView {
                values: f0_values.as_ptr(),
                values_len: f0_values.len(),
                validity_mask: bad_mask.as_ptr(),
                validity_mask_len: bad_mask.len(),
            },
            DagMlDataF64ColumnView {
                values: f1_values.as_ptr(),
                values_len: f1_values.len(),
                validity_mask: f1_mask.as_ptr(),
                validity_mask_len: f1_mask.len(),
            },
        ];
        let bad_mask_matrices = [DagMlDataFeatureMatrixF64ColumnarView {
            feature_set_id: bytes_view(b"x"),
            representation_id: bytes_view(b"tabular_numeric"),
            feature_names: feature_names.as_ptr(),
            feature_names_len: feature_names.len(),
            observation_ids: observation_ids.as_ptr(),
            observation_ids_len: observation_ids.len(),
            columns: bad_mask_columns.as_ptr(),
            columns_len: bad_mask_columns.len(),
        }];
        let status = unsafe {
            dagmldata_inmemory_provider_new_with_f64_feature_columns(
                envelope.as_ptr(),
                envelope.len(),
                target_tables.as_ptr(),
                target_tables.len(),
                bad_mask_matrices.as_ptr(),
                bad_mask_matrices.len(),
                &mut invalid_mask_vtable,
                &mut error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::ValidationError);
        assert!(invalid_mask_vtable.user_data.is_null());

        let mut mixed_mask_vtable = empty_vtable();
        let mixed_mask_columns = [
            DagMlDataF64ColumnView {
                values: f0_values.as_ptr(),
                values_len: f0_values.len(),
                validity_mask: std::ptr::null(),
                validity_mask_len: 0,
            },
            DagMlDataF64ColumnView {
                values: f1_values.as_ptr(),
                values_len: f1_values.len(),
                validity_mask: f1_mask.as_ptr(),
                validity_mask_len: f1_mask.len(),
            },
        ];
        let mixed_mask_matrices = [DagMlDataFeatureMatrixF64ColumnarView {
            feature_set_id: bytes_view(b"x"),
            representation_id: bytes_view(b"tabular_numeric"),
            feature_names: feature_names.as_ptr(),
            feature_names_len: feature_names.len(),
            observation_ids: observation_ids.as_ptr(),
            observation_ids_len: observation_ids.len(),
            columns: mixed_mask_columns.as_ptr(),
            columns_len: mixed_mask_columns.len(),
        }];
        let status = unsafe {
            dagmldata_inmemory_provider_new_with_f64_feature_columns(
                envelope.as_ptr(),
                envelope.len(),
                target_tables.as_ptr(),
                target_tables.len(),
                mixed_mask_matrices.as_ptr(),
                mixed_mask_matrices.len(),
                &mut mixed_mask_vtable,
                &mut error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::ValidationError);
        assert!(mixed_mask_vtable.user_data.is_null());
        let mixed_mask_error = unsafe { string_value(error) };
        assert!(
            mixed_mask_error.contains("either all columns must supply a validity_mask or none"),
            "unexpected mixed-mask error: {mixed_mask_error}"
        );

        unsafe {
            vtable.release.unwrap()(vtable.user_data, data_handle);
            dagmldata_inmemory_provider_destroy(&mut vtable);
        }
    }

    #[test]
    fn file_backed_provider_loads_persisted_buffer_store() {
        use dag_ml_data_core::buffer::{
            NumericFeatureBufferStore, NumericFeatureMatrixF64Columnar,
        };
        use dag_ml_data_core::ids::{ObservationId, RepresentationId};

        let matrix = NumericFeatureMatrixF64Columnar {
            feature_set_id: "x".to_string(),
            representation_id: RepresentationId::new("tabular_numeric").unwrap(),
            feature_names: vec!["f0".to_string(), "f1".to_string()],
            observation_ids: vec![
                ObservationId::new("obs.S001.base").unwrap(),
                ObservationId::new("obs.S001.rep1").unwrap(),
                ObservationId::new("obs.S001.aug0").unwrap(),
                ObservationId::new("obs.S002.base").unwrap(),
            ],
            columns: vec![vec![1.0, 2.0, 3.0, 4.0], vec![10.0, 20.0, 30.0, 40.0]],
            validity_masks: None,
        };
        let store = NumericFeatureBufferStore::from_f64_column_matrices(vec![matrix]).unwrap();
        let path = std::env::temp_dir().join(format!(
            "dag_ml_data_file_backed_provider_{}.n4d",
            std::process::id()
        ));
        dag_ml_data_core::buffer_file_store::write_store_to_path(&store, &path).unwrap();

        let envelope = include_bytes!(
            "../../../examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json"
        );
        let materialization_request = include_bytes!(
            "../../../examples/fixtures/oof_campaign/materialization_request_model_base_x.json"
        );
        let target_tables = b"[]";
        let path_str = path.to_string_lossy().into_owned();
        let path_bytes = path_str.as_bytes();
        let mut vtable = empty_vtable();
        let mut error = DagMlDataString::default();

        let status = unsafe {
            dagmldata_inmemory_provider_new_from_file(
                envelope.as_ptr(),
                envelope.len(),
                target_tables.as_ptr(),
                target_tables.len(),
                path_bytes.as_ptr(),
                path_bytes.len(),
                &mut vtable,
                &mut error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert!(error.ptr.is_null());

        let mut data_handle = 0;
        let status = unsafe {
            vtable.materialize.unwrap()(
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

        let mut manifest_json = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_provider_data_feature_buffer_manifest_json(
                &vtable,
                data_handle,
                &mut manifest_json,
                &mut error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        let manifests: serde_json::Value =
            serde_json::from_str(&unsafe { string_value(manifest_json) }).unwrap();
        assert_eq!(manifests[0]["feature_set_id"], serde_json::json!("x"));

        unsafe {
            vtable.release.unwrap()(vtable.user_data, data_handle);
            dagmldata_inmemory_provider_destroy(&mut vtable);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn file_backed_provider_rejects_missing_path() {
        let envelope = include_bytes!(
            "../../../examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json"
        );
        let missing_path = b"/tmp/dag_ml_data_definitely_missing.n4d";
        let mut vtable = empty_vtable();
        let mut error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_provider_new_from_file(
                envelope.as_ptr(),
                envelope.len(),
                b"[]".as_ptr(),
                2,
                missing_path.as_ptr(),
                missing_path.len(),
                &mut vtable,
                &mut error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::ValidationError);
        assert!(vtable.user_data.is_null());
        let message = unsafe { string_value(error) };
        assert!(
            message.contains("failed to read feature buffer store"),
            "unexpected: {message}"
        );
    }

    #[cfg(feature = "arrow-ipc")]
    #[test]
    fn arrow_ipc_provider_loads_buffer_store_from_disk() {
        use arrow_array::{Float64Array, RecordBatch, StringArray};
        use arrow_ipc::writer::FileWriter;
        use arrow_schema::{DataType, Field, Schema};
        use std::collections::HashMap;
        use std::sync::Arc;

        let mut metadata = HashMap::new();
        metadata.insert("dag_ml_data.feature_set_id".to_string(), "x".to_string());
        metadata.insert(
            "dag_ml_data.representation_id".to_string(),
            "tabular_numeric".to_string(),
        );
        let schema = Arc::new(Schema::new_with_metadata(
            vec![
                Field::new("observation_id", DataType::Utf8, false),
                Field::new("f0", DataType::Float64, true),
                Field::new("f1", DataType::Float64, true),
            ],
            metadata,
        ));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![
                    "obs.S001.base",
                    "obs.S001.rep1",
                    "obs.S001.aug0",
                    "obs.S002.base",
                ])),
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0])),
            ],
        )
        .unwrap();

        let path = std::env::temp_dir().join(format!(
            "dag_ml_data_arrow_ipc_provider_{}.arrow",
            std::process::id()
        ));
        {
            let mut file = std::fs::File::create(&path).unwrap();
            let mut writer = FileWriter::try_new(&mut file, &schema).unwrap();
            writer.write(&batch).unwrap();
            writer.finish().unwrap();
        }

        let envelope = include_bytes!(
            "../../../examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json"
        );
        let materialization_request = include_bytes!(
            "../../../examples/fixtures/oof_campaign/materialization_request_model_base_x.json"
        );
        let target_tables = b"[]";
        let path_str = path.to_string_lossy().into_owned();
        let path_bytes = path_str.as_bytes();
        let mut vtable = empty_vtable();
        let mut error = DagMlDataString::default();

        let status = unsafe {
            dagmldata_inmemory_provider_new_from_arrow_ipc(
                envelope.as_ptr(),
                envelope.len(),
                target_tables.as_ptr(),
                target_tables.len(),
                path_bytes.as_ptr(),
                path_bytes.len(),
                &mut vtable,
                &mut error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert!(error.ptr.is_null());

        let mut data_handle = 0;
        let status = unsafe {
            vtable.materialize.unwrap()(
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
        let mut manifest_json = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_provider_data_feature_buffer_manifest_json(
                &vtable,
                data_handle,
                &mut manifest_json,
                &mut error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        let manifests: serde_json::Value =
            serde_json::from_str(&unsafe { string_value(manifest_json) }).unwrap();
        assert_eq!(manifests[0]["feature_set_id"], serde_json::json!("x"));

        unsafe {
            vtable.release.unwrap()(vtable.user_data, data_handle);
            dagmldata_inmemory_provider_destroy(&mut vtable);
        }
        let _ = std::fs::remove_file(&path);
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

        let mut manifest_json = DagMlDataString::default();
        let mut manifest_error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_provider_feature_buffer_manifest_json(
                &vtable,
                &mut manifest_json,
                &mut manifest_error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert!(manifest_error.ptr.is_null());
        let manifests: serde_json::Value =
            serde_json::from_str(&unsafe { string_value(manifest_json) }).unwrap();
        assert_eq!(manifests.as_array().unwrap().len(), 2);
        assert_eq!(manifests[0]["schema_version"], serde_json::json!(1));
        assert_eq!(manifests[0]["feature_set_id"], serde_json::json!("x"));
        assert_eq!(manifests[0]["row_count"], serde_json::json!(4));
        assert_eq!(manifests[0]["feature_count"], serde_json::json!(2));
        assert_eq!(manifests[0]["estimated_value_bytes"], serde_json::json!(64));
        assert_eq!(
            manifests[0]["buffer_fingerprint"].as_str().unwrap().len(),
            64
        );

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
        let mut data_manifest_json = DagMlDataString::default();
        let mut data_manifest_error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_provider_data_feature_buffer_manifest_json(
                &vtable,
                data_handle,
                &mut data_manifest_json,
                &mut data_manifest_error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::Ok);
        assert!(data_manifest_error.ptr.is_null());
        let data_manifests: serde_json::Value =
            serde_json::from_str(&unsafe { string_value(data_manifest_json) }).unwrap();
        assert_eq!(data_manifests.as_array().unwrap().len(), 1);
        assert_eq!(data_manifests[0]["feature_set_id"], serde_json::json!("x"));
        assert_eq!(data_manifests[0]["source_ids"], serde_json::json!(["nir"]));

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

        let bad_selector = serde_json::to_vec(&serde_json::json!({
            "feature_set_id": "x_bad_representation",
            "policy": {
                "emit_mask": true
            }
        }))
        .unwrap();
        let mut bad_tensor_json = DagMlDataString::default();
        let mut bad_tensor_error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_provider_feature_collation_json(
                &vtable,
                view_handle,
                DagMlDataBytesView {
                    ptr: bad_selector.as_ptr(),
                    len: bad_selector.len(),
                },
                &mut bad_tensor_json,
                &mut bad_tensor_error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::ValidationError);
        assert!(bad_tensor_json.ptr.is_null());
        unsafe {
            dagmldata_string_free(bad_tensor_error);
        }

        let mut bad_tensor = DagMlDataTensorF64::default();
        let mut bad_tensor_error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_provider_feature_collation_tensor_f64_json(
                &vtable,
                view_handle,
                DagMlDataBytesView {
                    ptr: bad_selector.as_ptr(),
                    len: bad_selector.len(),
                },
                &mut bad_tensor,
                &mut bad_tensor_error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::ValidationError);
        assert!(bad_tensor.values.ptr.is_null());
        unsafe {
            dagmldata_string_free(bad_tensor_error);
        }

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
        let mut stale_manifest_json = DagMlDataString::default();
        let mut stale_manifest_error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_provider_data_feature_buffer_manifest_json(
                &vtable,
                data_handle,
                &mut stale_manifest_json,
                &mut stale_manifest_error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::ValidationError);
        assert!(stale_manifest_json.ptr.is_null());
        unsafe {
            dagmldata_string_free(stale_manifest_error);
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

        let mut stale_feature_array = std::ptr::null_mut();
        let mut stale_feature_schema = std::ptr::null_mut();
        let status = unsafe {
            feature_arrow(
                vtable.user_data,
                child_view_handle,
                DagMlDataBytesView {
                    ptr: feature_set_name.as_ptr(),
                    len: feature_set_name.len(),
                },
                &mut stale_feature_array,
                &mut stale_feature_schema,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::ValidationError);
        assert!(stale_feature_array.is_null());
        assert!(stale_feature_schema.is_null());

        let selector = serde_json::to_vec(&serde_json::json!({
            "feature_set_id": "x",
            "policy": {
                "emit_mask": true
            }
        }))
        .unwrap();
        let mut stale_tensor_json = DagMlDataString::default();
        let mut stale_tensor_error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_provider_feature_collation_json(
                &vtable,
                child_view_handle,
                DagMlDataBytesView {
                    ptr: selector.as_ptr(),
                    len: selector.len(),
                },
                &mut stale_tensor_json,
                &mut stale_tensor_error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::ValidationError);
        assert!(stale_tensor_json.ptr.is_null());
        unsafe {
            dagmldata_string_free(stale_tensor_error);
        }

        let mut stale_tensor = DagMlDataTensorF64::default();
        let mut stale_tensor_error = DagMlDataString::default();
        let status = unsafe {
            dagmldata_inmemory_provider_feature_collation_tensor_f64_json(
                &vtable,
                child_view_handle,
                DagMlDataBytesView {
                    ptr: selector.as_ptr(),
                    len: selector.len(),
                },
                &mut stale_tensor,
                &mut stale_tensor_error,
            )
        };
        assert_eq!(status, DagMlDataStatusCode::ValidationError);
        assert!(stale_tensor.values.ptr.is_null());
        unsafe {
            dagmldata_string_free(stale_tensor_error);
        }

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

    #[test]
    fn inmemory_provider_rejects_non_numeric_feature_buffers_on_creation() {
        let envelope = include_bytes!(
            "../../../examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json"
        );
        let feature_tables = serde_json::to_vec(&serde_json::json!([
            {
                "feature_set_id": "x",
                "representation_id": "tabular_numeric",
                "feature_names": ["f0"],
                "rows": [
                    {"observation_id": "obs.S001.base", "values": ["bad"]}
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
                std::ptr::null(),
                0,
                feature_tables.as_ptr(),
                feature_tables.len(),
                &mut vtable,
                &mut error,
            )
        };

        assert_eq!(status, DagMlDataStatusCode::ValidationError);
        assert!(vtable.user_data.is_null());
        assert!(!error.ptr.is_null());
        unsafe {
            dagmldata_string_free(error);
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

    fn bytes_view(bytes: &'static [u8]) -> DagMlDataBytesView {
        DagMlDataBytesView {
            ptr: bytes.as_ptr(),
            len: bytes.len(),
        }
    }

    unsafe fn string_value(value: DagMlDataString) -> String {
        let bytes = slice::from_raw_parts(value.ptr.cast::<u8>(), value.len);
        let out = String::from_utf8(bytes.to_vec()).unwrap();
        dagmldata_string_free(value);
        out
    }

    unsafe fn borrowed_string(value: &DagMlDataString) -> String {
        assert!(!value.ptr.is_null());
        let bytes = slice::from_raw_parts(value.ptr.cast::<u8>(), value.len);
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    unsafe fn string_array_values(array: DagMlDataStringArray) -> Vec<String> {
        if array.ptr.is_null() {
            return Vec::new();
        }
        slice::from_raw_parts(array.ptr, array.len)
            .iter()
            .map(|value| borrowed_string(value))
            .collect()
    }

    unsafe fn usize_array_values(array: DagMlDataUSizeArray) -> Vec<usize> {
        if array.ptr.is_null() {
            return Vec::new();
        }
        slice::from_raw_parts(array.ptr, array.len).to_vec()
    }

    unsafe fn f64_array_values(array: DagMlDataF64Array) -> Vec<f64> {
        if array.ptr.is_null() {
            return Vec::new();
        }
        slice::from_raw_parts(array.ptr, array.len).to_vec()
    }

    unsafe fn f32_array_values(array: DagMlDataF32Array) -> Vec<f32> {
        if array.ptr.is_null() {
            return Vec::new();
        }
        slice::from_raw_parts(array.ptr, array.len).to_vec()
    }

    unsafe fn u8_array_values(array: DagMlDataU8Array) -> Vec<u8> {
        if array.ptr.is_null() {
            return Vec::new();
        }
        slice::from_raw_parts(array.ptr, array.len).to_vec()
    }

    fn multisource_provider_fixture() -> (Vec<u8>, Vec<u8>) {
        use std::collections::BTreeMap;

        use dag_ml_data_core::{
            data_plan_fingerprint, CoordinatorDataMaterializationRequest,
            CoordinatorDataPlanEnvelope, CoordinatorRelation, CoordinatorRelationSet, DataPlan,
            DataPlanStep, DataPlanStepKind, FitScope,
        };

        let tabular = RepresentationId::new("tabular_numeric").unwrap();
        let plan = DataPlan {
            id: "nir-chem-tabular".to_string(),
            steps: vec![
                DataPlanStep {
                    kind: DataPlanStepKind::Materialize,
                    source_id: Some(SourceId::new("nir").unwrap()),
                    adapter_id: None,
                    input_representation: None,
                    output_representation: Some(tabular.clone()),
                    fit_scope: FitScope::Stateless,
                    requires_user_choice: false,
                    metadata: BTreeMap::new(),
                },
                DataPlanStep {
                    kind: DataPlanStepKind::Materialize,
                    source_id: Some(SourceId::new("chem").unwrap()),
                    adapter_id: None,
                    input_representation: None,
                    output_representation: Some(tabular.clone()),
                    fit_scope: FitScope::Stateless,
                    requires_user_choice: false,
                    metadata: BTreeMap::new(),
                },
                DataPlanStep {
                    kind: DataPlanStepKind::Align,
                    source_id: None,
                    adapter_id: None,
                    input_representation: Some(tabular.clone()),
                    output_representation: Some(tabular.clone()),
                    fit_scope: FitScope::Stateless,
                    requires_user_choice: false,
                    metadata: BTreeMap::new(),
                },
                DataPlanStep {
                    kind: DataPlanStepKind::Join,
                    source_id: None,
                    adapter_id: None,
                    input_representation: Some(tabular.clone()),
                    output_representation: Some(tabular.clone()),
                    fit_scope: FitScope::Stateless,
                    requires_user_choice: false,
                    metadata: BTreeMap::new(),
                },
            ],
            output_representation: tabular.clone(),
            issues: Vec::new(),
        };
        let plan_fingerprint = data_plan_fingerprint(&plan).unwrap();
        let schema_fingerprint = "a".repeat(64);
        let envelope = CoordinatorDataPlanEnvelope {
            schema_version: dag_ml_data_core::COORDINATOR_DATA_PLAN_ENVELOPE_SCHEMA_VERSION,
            schema_fingerprint: schema_fingerprint.clone(),
            plan_fingerprint: plan_fingerprint.clone(),
            relation_fingerprint: None,
            plan,
            coordinator_relations: Some(CoordinatorRelationSet {
                records: vec![
                    CoordinatorRelation {
                        observation_id: ObservationId::new("obs.S001.r1").unwrap(),
                        sample_id: SampleId::new("S001").unwrap(),
                        target_id: None,
                        group_id: None,
                        origin_sample_id: None,
                        source_id: Some(SourceId::new("nir").unwrap()),
                        is_augmented: false,
                    },
                    CoordinatorRelation {
                        observation_id: ObservationId::new("obs.S001.r2").unwrap(),
                        sample_id: SampleId::new("S001").unwrap(),
                        target_id: None,
                        group_id: None,
                        origin_sample_id: None,
                        source_id: Some(SourceId::new("nir").unwrap()),
                        is_augmented: false,
                    },
                    CoordinatorRelation {
                        observation_id: ObservationId::new("obs.S002.r1").unwrap(),
                        sample_id: SampleId::new("S002").unwrap(),
                        target_id: None,
                        group_id: None,
                        origin_sample_id: None,
                        source_id: Some(SourceId::new("nir").unwrap()),
                        is_augmented: false,
                    },
                    CoordinatorRelation {
                        observation_id: ObservationId::new("chem.S001").unwrap(),
                        sample_id: SampleId::new("S001").unwrap(),
                        target_id: None,
                        group_id: None,
                        origin_sample_id: None,
                        source_id: Some(SourceId::new("chem").unwrap()),
                        is_augmented: false,
                    },
                    CoordinatorRelation {
                        observation_id: ObservationId::new("chem.S002").unwrap(),
                        sample_id: SampleId::new("S002").unwrap(),
                        target_id: None,
                        group_id: None,
                        origin_sample_id: None,
                        source_id: Some(SourceId::new("chem").unwrap()),
                        is_augmented: false,
                    },
                ],
            }),
            metadata: BTreeMap::new(),
        };
        envelope.validate().unwrap();
        let request = CoordinatorDataMaterializationRequest {
            run_id: "run:test".to_string(),
            node_id: "node:model".to_string(),
            input_name: "X".to_string(),
            phase: "fit".to_string(),
            variant_id: None,
            fold_id: None,
            request_id: "req:test".to_string(),
            schema_fingerprint,
            plan_fingerprint,
            relation_fingerprint: None,
            output_representation: tabular,
            source_ids: Vec::new(),
            require_relations: false,
        };
        request.validate().unwrap();
        (
            serde_json::to_vec(&envelope).unwrap(),
            serde_json::to_vec(&request).unwrap(),
        )
    }
}
