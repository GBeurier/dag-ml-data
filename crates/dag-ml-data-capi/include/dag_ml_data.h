#ifndef DAG_ML_DATA_H
#define DAG_ML_DATA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t DagMlDataHandle;

typedef enum DagMlDataStatusCode {
    DAG_ML_DATA_STATUS_OK = 0,
    DAG_ML_DATA_STATUS_INVALID_ARGUMENT = 1,
    DAG_ML_DATA_STATUS_VALIDATION_ERROR = 2,
    DAG_ML_DATA_STATUS_PANIC = 255
} DagMlDataStatusCode;

typedef struct DagMlDataVersion {
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
} DagMlDataVersion;

typedef struct DagMlDataString {
    char *ptr;
    size_t len;
} DagMlDataString;

typedef struct DagMlDataBytesView {
    const uint8_t *ptr;
    size_t len;
} DagMlDataBytesView;

#define DAG_ML_DATA_TENSOR_F64_ABI_VERSION 1u

typedef struct DagMlDataStringArray {
    DagMlDataString *ptr;
    size_t len;
} DagMlDataStringArray;

typedef struct DagMlDataUSizeArray {
    size_t *ptr;
    size_t len;
} DagMlDataUSizeArray;

typedef struct DagMlDataF64Array {
    double *ptr;
    size_t len;
} DagMlDataF64Array;

typedef struct DagMlDataU8Array {
    uint8_t *ptr;
    size_t len;
} DagMlDataU8Array;

typedef struct DagMlDataTensorF64 {
    uint32_t abi_version;
    DagMlDataString block_id;
    DagMlDataString representation_id;
    DagMlDataString batch_container;
    DagMlDataStringArray observation_ids;
    DagMlDataStringArray sample_ids;
    DagMlDataUSizeArray shape;
    DagMlDataF64Array values;
    DagMlDataU8Array presence_mask;
    DagMlDataU8Array validity_mask;
    DagMlDataStringArray feature_names;
} DagMlDataTensorF64;

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

typedef struct ArrowArray {
    int64_t length;
    int64_t null_count;
    int64_t offset;
    int64_t n_buffers;
    int64_t n_children;
    const void **buffers;
    struct ArrowArray **children;
    struct ArrowArray *dictionary;
    void (*release)(struct ArrowArray *array);
    void *private_data;
} ArrowArray;

typedef struct ArrowSchema {
    const char *format;
    const char *name;
    const char *metadata;
    int64_t flags;
    int64_t n_children;
    struct ArrowSchema **children;
    struct ArrowSchema *dictionary;
    void (*release)(struct ArrowSchema *schema);
    void *private_data;
} ArrowSchema;

#endif

#ifndef DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION
#define DAG_ML_DATA_PROVIDER_VTABLE_ABI_VERSION 2u
#endif

#ifndef DAG_ML_DATA_VTABLE_DEFINED
#define DAG_ML_DATA_VTABLE_DEFINED
typedef struct DagMlDataVTable {
    uint32_t abi_version;
    void *user_data;
    DagMlDataStatusCode (*materialize)(void *user_data, DagMlDataHandle dataset, DagMlDataBytesView request_json, DagMlDataHandle *out_handle);
    DagMlDataStatusCode (*make_view)(void *user_data, DagMlDataHandle dataset, DagMlDataBytesView selector_json, DagMlDataHandle *out_view);
    DagMlDataStatusCode (*view_identity)(void *user_data, DagMlDataHandle view, ArrowArray **out_arrow_array, ArrowSchema **out_arrow_schema);
    DagMlDataStatusCode (*target_arrow)(void *user_data, DagMlDataHandle view, DagMlDataBytesView target_name, ArrowArray **out_arrow_array, ArrowSchema **out_arrow_schema);
    DagMlDataStatusCode (*feature_arrow)(void *user_data, DagMlDataHandle view, DagMlDataBytesView feature_set_name, ArrowArray **out_arrow_array, ArrowSchema **out_arrow_schema);
    void (*release)(void *user_data, DagMlDataHandle handle);
    void (*destroy)(void *user_data);
} DagMlDataVTable;
#endif

DagMlDataVersion dagmldata_version(void);
void dagmldata_string_free(DagMlDataString value);
void dagmldata_tensor_f64_free(DagMlDataTensorF64 tensor);
void dagmldata_arrow_array_free(ArrowArray *array);
void dagmldata_arrow_schema_free(ArrowSchema *schema);
DagMlDataStatusCode dagmldata_schema_fingerprint_json(const uint8_t *json_ptr, size_t json_len, DagMlDataString *fingerprint_out, DagMlDataString *error_out);
DagMlDataStatusCode dagmldata_coordinator_identity_arrow_json(const uint8_t *json_ptr, size_t json_len, ArrowArray **out_arrow_array, ArrowSchema **out_arrow_schema, DagMlDataString *error_out);
DagMlDataStatusCode dagmldata_coordinator_target_arrow_json(const uint8_t *json_ptr, size_t json_len, ArrowArray **out_arrow_array, ArrowSchema **out_arrow_schema, DagMlDataString *error_out);
DagMlDataStatusCode dagmldata_coordinator_feature_arrow_json(const uint8_t *json_ptr, size_t json_len, ArrowArray **out_arrow_array, ArrowSchema **out_arrow_schema, DagMlDataString *error_out);
DagMlDataStatusCode dagmldata_coordinator_feature_fusion_arrow_json(const uint8_t *json_ptr, size_t json_len, ArrowArray **out_arrow_array, ArrowSchema **out_arrow_schema, DagMlDataString *error_out);
DagMlDataStatusCode dagmldata_coordinator_feature_collation_json(const uint8_t *json_ptr, size_t json_len, DagMlDataString *out_json, DagMlDataString *error_out);
DagMlDataStatusCode dagmldata_coordinator_feature_collation_tensor_f64_json(const uint8_t *json_ptr, size_t json_len, DagMlDataTensorF64 *out_tensor, DagMlDataString *error_out);
DagMlDataStatusCode dagmldata_inmemory_provider_new_json(const uint8_t *envelope_ptr, size_t envelope_len, const uint8_t *target_tables_ptr, size_t target_tables_len, DagMlDataVTable *out_vtable, DagMlDataString *error_out);
DagMlDataStatusCode dagmldata_inmemory_provider_new_with_features_json(const uint8_t *envelope_ptr, size_t envelope_len, const uint8_t *target_tables_ptr, size_t target_tables_len, const uint8_t *feature_tables_ptr, size_t feature_tables_len, DagMlDataVTable *out_vtable, DagMlDataString *error_out);
DagMlDataStatusCode dagmldata_inmemory_provider_feature_buffer_manifest_json(const DagMlDataVTable *vtable, DagMlDataString *out_json, DagMlDataString *error_out);
DagMlDataStatusCode dagmldata_inmemory_provider_data_feature_buffer_manifest_json(const DagMlDataVTable *vtable, DagMlDataHandle data_handle, DagMlDataString *out_json, DagMlDataString *error_out);
DagMlDataStatusCode dagmldata_inmemory_provider_feature_collation_json(const DagMlDataVTable *vtable, DagMlDataHandle view, DagMlDataBytesView selector_json, DagMlDataString *out_json, DagMlDataString *error_out);
DagMlDataStatusCode dagmldata_inmemory_provider_feature_collation_tensor_f64_json(const DagMlDataVTable *vtable, DagMlDataHandle view, DagMlDataBytesView selector_json, DagMlDataTensorF64 *out_tensor, DagMlDataString *error_out);
void dagmldata_inmemory_provider_destroy(DagMlDataVTable *vtable);

#ifdef __cplusplus
}
#endif

#endif
