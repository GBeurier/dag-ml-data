use std::ffi::{c_void, CString};
use std::os::raw::c_char;
use std::slice;

use dag_ml_data_core::{schema_fingerprint, DatasetSchema};

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
            out_arrow_array: *mut *mut c_void,
            out_arrow_schema: *mut *mut c_void,
        ) -> DagMlDataStatusCode,
    >,
    pub target_arrow: Option<
        unsafe extern "C" fn(
            user_data: *mut c_void,
            view: DagMlDataHandle,
            target_name: DagMlDataBytesView,
            out_arrow_array: *mut *mut c_void,
            out_arrow_schema: *mut *mut c_void,
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

unsafe fn clear_string(out: *mut DagMlDataString) {
    if !out.is_null() {
        *out = DagMlDataString::default();
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
