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

typedef struct DagMlDataVTable {
    uint32_t abi_version;
    void *user_data;
    DagMlDataStatusCode (*materialize)(void *user_data, DagMlDataHandle dataset, DagMlDataBytesView request_json, DagMlDataHandle *out_handle);
    DagMlDataStatusCode (*make_view)(void *user_data, DagMlDataHandle dataset, DagMlDataBytesView selector_json, DagMlDataHandle *out_view);
    DagMlDataStatusCode (*view_identity)(void *user_data, DagMlDataHandle view, ArrowArray **out_arrow_array, ArrowSchema **out_arrow_schema);
    DagMlDataStatusCode (*target_arrow)(void *user_data, DagMlDataHandle view, DagMlDataBytesView target_name, ArrowArray **out_arrow_array, ArrowSchema **out_arrow_schema);
    void (*release)(void *user_data, DagMlDataHandle handle);
    void (*destroy)(void *user_data);
} DagMlDataVTable;

DagMlDataVersion dagmldata_version(void);
void dagmldata_string_free(DagMlDataString value);
void dagmldata_arrow_array_free(ArrowArray *array);
void dagmldata_arrow_schema_free(ArrowSchema *schema);
DagMlDataStatusCode dagmldata_schema_fingerprint_json(const uint8_t *json_ptr, size_t json_len, DagMlDataString *fingerprint_out, DagMlDataString *error_out);
DagMlDataStatusCode dagmldata_coordinator_identity_arrow_json(const uint8_t *json_ptr, size_t json_len, ArrowArray **out_arrow_array, ArrowSchema **out_arrow_schema, DagMlDataString *error_out);

#ifdef __cplusplus
}
#endif

#endif
