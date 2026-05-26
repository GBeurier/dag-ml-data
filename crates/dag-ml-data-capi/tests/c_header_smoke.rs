use std::fs;
use std::process::Command;

#[test]
fn c_header_exposes_provider_vtable_with_arrow_types() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let source = r#"
#include <stddef.h>
#include <stdint.h>
#include "dag_ml_data.h"

static DagMlDataStatusCode materialize(void *user_data, DagMlDataHandle dataset, DagMlDataBytesView request_json, DagMlDataHandle *out_handle) {
    (void)user_data;
    (void)dataset;
    (void)request_json;
    if (out_handle != NULL) {
        *out_handle = 1;
    }
    return DAG_ML_DATA_STATUS_OK;
}

static DagMlDataStatusCode make_view(void *user_data, DagMlDataHandle data, DagMlDataBytesView selector_json, DagMlDataHandle *out_view) {
    (void)user_data;
    (void)data;
    (void)selector_json;
    if (out_view != NULL) {
        *out_view = 2;
    }
    return DAG_ML_DATA_STATUS_OK;
}

static DagMlDataStatusCode view_identity(void *user_data, DagMlDataHandle view, ArrowArray **out_arrow_array, ArrowSchema **out_arrow_schema) {
    (void)user_data;
    (void)view;
    if (out_arrow_array != NULL) {
        *out_arrow_array = NULL;
    }
    if (out_arrow_schema != NULL) {
        *out_arrow_schema = NULL;
    }
    return DAG_ML_DATA_STATUS_OK;
}

static DagMlDataStatusCode target_arrow(void *user_data, DagMlDataHandle view, DagMlDataBytesView target_name, ArrowArray **out_arrow_array, ArrowSchema **out_arrow_schema) {
    (void)user_data;
    (void)view;
    (void)target_name;
    if (out_arrow_array != NULL) {
        *out_arrow_array = NULL;
    }
    if (out_arrow_schema != NULL) {
        *out_arrow_schema = NULL;
    }
    return DAG_ML_DATA_STATUS_OK;
}

int main(void) {
    DagMlDataVTable table = {0};
    DagMlDataString error = {0};
    ArrowArray *array = NULL;
    ArrowSchema *schema = NULL;

    table.abi_version = 1;
    table.materialize = materialize;
    table.make_view = make_view;
    table.view_identity = view_identity;
    table.target_arrow = target_arrow;

    (void)dagmldata_version();
    (void)dagmldata_inmemory_provider_new_json((const uint8_t*)"{}", 2, NULL, 0, &table, &error);
    dagmldata_arrow_array_free(array);
    dagmldata_arrow_schema_free(schema);
    dagmldata_inmemory_provider_destroy(&table);
    return 0;
}
"#;
    let path = std::env::temp_dir().join(format!(
        "dag_ml_data_c_header_smoke_{}.c",
        std::process::id()
    ));
    fs::write(&path, source).expect("write C header smoke source");

    let output = Command::new("cc")
        .arg("-std=c11")
        .arg("-fsyntax-only")
        .arg("-I")
        .arg(format!("{manifest_dir}/include"))
        .arg(&path)
        .output()
        .expect("run C compiler");
    let _ = fs::remove_file(&path);

    assert!(
        output.status.success(),
        "C header smoke failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
