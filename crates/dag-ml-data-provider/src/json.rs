//! JSON transport facade over [`InMemoryProvider`].
//!
//! [`JsonInMemoryProvider`] is an additive convenience wrapper that takes and
//! returns UTF-8 JSON (and decimal-string handles) instead of typed Rust
//! contract values. It is shared by the non-C-ABI language bindings (WASM via
//! wasm-bindgen, R via extendr) so each binding is a tiny adapter rather than a
//! re-implementation. It depends only on `serde_json` — no wasm / R / FFI deps,
//! so the provider crate stays target-neutral. The typed
//! [`crate::DagMlDataProvider`] trait and [`InMemoryProvider`] remain the
//! contract-typed surface; this is strictly a string transport on top.
//!
//! Handles cross as DECIMAL STRINGS because JS (and R) numbers cannot represent
//! the full `u64` range.

use std::collections::BTreeMap;

use dag_ml_data_core::{
    collate_feature_block, CoordinatorDataMaterializationRequest, CoordinatorDataPlanEnvelope,
    CoordinatorFeatureTable, CoordinatorTargetTable, DataError, DataView,
    NumericFeatureBufferStore, NumericFeatureMatrixF64, Result, TargetId,
};
use serde::de::DeserializeOwned;

use crate::{DagMlDataProvider, InMemoryProvider, ProviderFeatureCollationRequest};

/// JSON facade over [`InMemoryProvider`]: JSON in, JSON out, decimal-string
/// handles.
pub struct JsonInMemoryProvider {
    provider: InMemoryProvider,
}

impl JsonInMemoryProvider {
    /// Builds the provider from JSON. At most one of `feature_tables_json` or
    /// `f64_feature_matrices_json` may be a non-empty array.
    pub fn from_json(
        envelope_json: &str,
        target_tables_json: Option<&str>,
        feature_tables_json: Option<&str>,
        f64_feature_matrices_json: Option<&str>,
    ) -> Result<Self> {
        let envelope: CoordinatorDataPlanEnvelope =
            serde_json::from_str(envelope_json).map_err(DataError::Serialization)?;
        let target_tables = parse_target_tables(target_tables_json)?;
        let feature_store = parse_feature_store(feature_tables_json, f64_feature_matrices_json)?;
        Ok(Self {
            provider: InMemoryProvider::new(envelope, target_tables, feature_store)?,
        })
    }

    /// Materializes a data handle and returns it as a decimal string.
    pub fn materialize(&self, request_json: &str) -> Result<String> {
        let request: CoordinatorDataMaterializationRequest =
            serde_json::from_str(request_json).map_err(DataError::Serialization)?;
        let record = self.provider.materialize(&request)?;
        Ok(record.handle.handle.to_string())
    }

    /// Creates a view over `data_handle` and returns the view handle string.
    pub fn make_view(&self, data_handle: &str, view_json: &str) -> Result<String> {
        let view: DataView = serde_json::from_str(view_json).map_err(DataError::Serialization)?;
        let record = self.provider.make_view(parse_handle(data_handle)?, &view)?;
        Ok(record.handle.handle.to_string())
    }

    /// Returns the view identity rows as a `CoordinatorRelationSet` JSON.
    pub fn view_identity(&self, view_handle: &str) -> Result<String> {
        let relations = self.provider.view_identity(parse_handle(view_handle)?)?;
        serde_json::to_string(&relations).map_err(DataError::Serialization)
    }

    /// Returns the target block JSON for `target_id`.
    pub fn target_block(&self, view_handle: &str, target_id: &str) -> Result<String> {
        let target_id = TargetId::new(target_id)?;
        let block = self
            .provider
            .target_block(parse_handle(view_handle)?, &target_id)?;
        serde_json::to_string(&block).map_err(DataError::Serialization)
    }

    /// Returns the feature block JSON for `feature_set_id`.
    pub fn feature_block(&self, view_handle: &str, feature_set_id: &str) -> Result<String> {
        let block = self
            .provider
            .feature_block(parse_handle(view_handle)?, feature_set_id)?;
        serde_json::to_string(&block).map_err(DataError::Serialization)
    }

    /// Collates a selector (single feature set or multi-source fusion) into a
    /// row-major `NumericTensorBlock` JSON, honoring the selector's collation
    /// policy.
    pub fn feature_collation(&self, view_handle: &str, selector_json: &str) -> Result<String> {
        let selector: ProviderFeatureCollationRequest =
            serde_json::from_str(selector_json).map_err(DataError::Serialization)?;
        let block = self
            .provider
            .feature_collation_block(parse_handle(view_handle)?, &selector)?;
        let tensor = collate_feature_block(&block, &selector.policy)?;
        serde_json::to_string(&tensor).map_err(DataError::Serialization)
    }

    /// Returns the provider-wide feature-buffer manifests JSON.
    pub fn feature_buffer_manifests(&self) -> Result<String> {
        let manifests = self.provider.feature_buffer_manifests()?;
        serde_json::to_string(&manifests).map_err(DataError::Serialization)
    }

    /// Returns the feature-buffer bindings JSON scoped to a data handle.
    pub fn data_feature_buffer_bindings(&self, data_handle: &str) -> Result<String> {
        let bindings = self
            .provider
            .data_feature_buffer_bindings(parse_handle(data_handle)?)?;
        serde_json::to_string(&bindings).map_err(DataError::Serialization)
    }

    /// Releases a data or view handle; returns whether anything was released.
    pub fn release(&self, handle: &str) -> Result<bool> {
        Ok(self.provider.release(parse_handle(handle)?))
    }
}

fn parse_handle(handle: &str) -> Result<u64> {
    handle.parse::<u64>().map_err(|_| {
        DataError::Validation(format!("handle `{handle}` is not a u64 decimal string"))
    })
}

fn parse_target_tables(json: Option<&str>) -> Result<BTreeMap<TargetId, CoordinatorTargetTable>> {
    let tables: Vec<CoordinatorTargetTable> = parse_json_array(json)?;
    let mut by_target = BTreeMap::new();
    for table in tables {
        table.validate()?;
        let target_id = table.target_id.clone();
        if by_target.insert(target_id.clone(), table).is_some() {
            return Err(DataError::Validation(format!(
                "duplicate target table `{target_id}`"
            )));
        }
    }
    Ok(by_target)
}

fn parse_feature_store(
    feature_tables_json: Option<&str>,
    f64_feature_matrices_json: Option<&str>,
) -> Result<NumericFeatureBufferStore> {
    // Parse each candidate; a semantically-empty array (e.g. "[]", "[ ]") yields
    // an empty Vec and is treated as "no payload" — so an empty branch never
    // trips the both-inputs rejection.
    let feature_tables: Vec<CoordinatorFeatureTable> = parse_json_array(feature_tables_json)?;
    let f64_matrices: Vec<NumericFeatureMatrixF64> = parse_json_array(f64_feature_matrices_json)?;
    match (feature_tables.is_empty(), f64_matrices.is_empty()) {
        (false, false) => Err(DataError::Validation(
            "pass at most one of feature_tables or f64_feature_matrices".to_string(),
        )),
        (false, true) => NumericFeatureBufferStore::from_feature_tables(feature_tables),
        (true, false) => NumericFeatureBufferStore::from_f64_matrices(f64_matrices),
        (true, true) => Ok(NumericFeatureBufferStore::default()),
    }
}

/// Parses an optional JSON array; absent or blank input yields an empty Vec.
fn parse_json_array<T: DeserializeOwned>(json: Option<&str>) -> Result<Vec<T>> {
    match json.map(str::trim).filter(|value| !value.is_empty()) {
        Some(json) => serde_json::from_str(json).map_err(DataError::Serialization),
        None => Ok(Vec::new()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENVELOPE: &str = include_str!(
        "../../../examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json"
    );
    const REQUEST: &str = include_str!(
        "../../../examples/fixtures/oof_campaign/materialization_request_model_base_x.json"
    );
    const F64_MATRICES: &str = r#"[
        {
            "feature_set_id": "x",
            "representation_id": "tabular_numeric",
            "feature_names": ["f0", "f1"],
            "observation_ids": ["obs.S001.base", "obs.S001.rep1", "obs.S001.aug0", "obs.S002.base"],
            "values": [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0]
        }
    ]"#;

    fn provider() -> JsonInMemoryProvider {
        JsonInMemoryProvider::from_json(ENVELOPE, None, None, Some(F64_MATRICES)).unwrap()
    }

    #[test]
    fn materialize_make_view_and_feature_block_round_trip() {
        let provider = provider();
        let data_handle = provider.materialize(REQUEST).unwrap();
        assert_eq!(data_handle, "1");
        let view = serde_json::json!({"sample_ids": ["S002", "S001"], "include_augmented": false})
            .to_string();
        let view_handle = provider.make_view(&data_handle, &view).unwrap();
        let features: serde_json::Value =
            serde_json::from_str(&provider.feature_block(&view_handle, "x").unwrap()).unwrap();
        assert_eq!(
            features["observation_ids"],
            serde_json::json!(["obs.S002.base", "obs.S001.base", "obs.S001.rep1"])
        );
        assert_eq!(features["feature_names"], serde_json::json!(["f0", "f1"]));
    }

    #[test]
    fn feature_collation_emits_tensor_with_mask() {
        let provider = provider();
        let data_handle = provider.materialize(REQUEST).unwrap();
        let view = serde_json::json!({"sample_ids": ["S002", "S001"], "include_augmented": false})
            .to_string();
        let view_handle = provider.make_view(&data_handle, &view).unwrap();
        let selector =
            serde_json::json!({"feature_set_id": "x", "policy": {"emit_mask": true}}).to_string();
        let tensor: serde_json::Value =
            serde_json::from_str(&provider.feature_collation(&view_handle, &selector).unwrap())
                .unwrap();
        assert_eq!(tensor["shape"], serde_json::json!([3, 2]));
        assert!(tensor.get("values").is_some());
        assert!(tensor.get("presence_mask").is_some());
    }

    #[test]
    fn rejects_both_feature_inputs() {
        let feature_tables = r#"[{"feature_set_id":"a","representation_id":"r","feature_names":["f0"],"rows":[{"observation_id":"o1","values":[1.0]}]}]"#;
        let f64_matrices = r#"[{"feature_set_id":"b","representation_id":"r","feature_names":["f0"],"observation_ids":["o1"],"values":[1.0]}]"#;
        let error = match JsonInMemoryProvider::from_json(
            ENVELOPE,
            None,
            Some(feature_tables),
            Some(f64_matrices),
        ) {
            Ok(_) => panic!("expected an error when both feature inputs are provided"),
            Err(error) => error,
        };
        assert!(format!("{error}").contains("at most one"));
    }

    #[test]
    fn empty_array_branch_does_not_trip_both_inputs_rejection() {
        let provider =
            JsonInMemoryProvider::from_json(ENVELOPE, None, Some("[ ]"), Some(F64_MATRICES))
                .unwrap();
        assert_eq!(provider.materialize(REQUEST).unwrap(), "1");
    }

    #[test]
    fn rejects_non_decimal_handle() {
        let error = provider().view_identity("not-a-handle").unwrap_err();
        assert!(format!("{error}").contains("not a u64"));
    }
}
