use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::coordinator::{validate_fingerprint, CoordinatorDataPlanEnvelope};
use crate::error::{DataError, Result};
use crate::ids::{RepresentationId, SourceId};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoordinatorHandleKind {
    Data,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CoordinatorHandleRef {
    pub handle: u64,
    pub kind: CoordinatorHandleKind,
    pub owner_controller: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CoordinatorDataMaterializationRequest {
    pub run_id: String,
    pub node_id: String,
    pub input_name: String,
    pub phase: String,
    #[serde(default)]
    pub variant_id: Option<String>,
    #[serde(default)]
    pub fold_id: Option<String>,
    pub request_id: String,
    pub schema_fingerprint: String,
    pub plan_fingerprint: String,
    #[serde(default)]
    pub relation_fingerprint: Option<String>,
    pub output_representation: RepresentationId,
    #[serde(default)]
    pub source_ids: Vec<SourceId>,
    #[serde(default)]
    pub require_relations: bool,
}

impl CoordinatorDataMaterializationRequest {
    pub fn validate(&self) -> Result<()> {
        validate_non_empty("run_id", &self.run_id)?;
        validate_non_empty("node_id", &self.node_id)?;
        validate_non_empty("input_name", &self.input_name)?;
        validate_non_empty("phase", &self.phase)?;
        validate_non_empty("request_id", &self.request_id)?;
        validate_fingerprint("schema", &self.schema_fingerprint)?;
        validate_fingerprint("plan", &self.plan_fingerprint)?;
        if let Some(relation_fingerprint) = &self.relation_fingerprint {
            validate_fingerprint("relation", relation_fingerprint)?;
        } else if self.require_relations {
            return Err(DataError::Validation(format!(
                "materialization request `{}` on `{}` requires relations but has no relation_fingerprint",
                self.input_name, self.node_id
            )));
        }
        let unique_sources = self.source_ids.iter().collect::<BTreeSet<_>>();
        if unique_sources.len() != self.source_ids.len() {
            return Err(DataError::Validation(format!(
                "materialization request `{}` on `{}` contains duplicate source ids",
                self.input_name, self.node_id
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CoordinatorDataHandleRecord {
    pub handle: CoordinatorHandleRef,
    pub run_id: String,
    pub node_id: String,
    pub input_name: String,
    pub phase: String,
    #[serde(default)]
    pub variant_id: Option<String>,
    #[serde(default)]
    pub fold_id: Option<String>,
    pub request_id: String,
    pub schema_fingerprint: String,
    pub plan_fingerprint: String,
    #[serde(default)]
    pub relation_fingerprint: Option<String>,
    pub plan_id: String,
    pub output_representation: RepresentationId,
    #[serde(default)]
    pub source_ids: Vec<SourceId>,
    #[serde(default)]
    pub sample_count: Option<usize>,
    #[serde(default)]
    pub relation_record_count: Option<usize>,
}

#[derive(Debug)]
pub struct CoordinatorHandleArena {
    owner_controller: String,
    next_handle: RefCell<u64>,
    records: RefCell<BTreeMap<u64, CoordinatorDataHandleRecord>>,
}

impl CoordinatorHandleArena {
    pub fn new(owner_controller: impl Into<String>) -> Result<Self> {
        let owner_controller = owner_controller.into();
        validate_non_empty("owner_controller", &owner_controller)?;
        Ok(Self {
            owner_controller,
            next_handle: RefCell::new(1),
            records: RefCell::new(BTreeMap::new()),
        })
    }

    pub fn materialize(
        &self,
        envelope: &CoordinatorDataPlanEnvelope,
        request: &CoordinatorDataMaterializationRequest,
    ) -> Result<CoordinatorDataHandleRecord> {
        envelope.validate()?;
        request.validate()?;
        validate_request_against_envelope(envelope, request)?;

        let handle = CoordinatorHandleRef {
            handle: self.next_handle(),
            kind: CoordinatorHandleKind::Data,
            owner_controller: self.owner_controller.clone(),
        };
        let record = CoordinatorDataHandleRecord {
            handle: handle.clone(),
            run_id: request.run_id.clone(),
            node_id: request.node_id.clone(),
            input_name: request.input_name.clone(),
            phase: request.phase.clone(),
            variant_id: request.variant_id.clone(),
            fold_id: request.fold_id.clone(),
            request_id: request.request_id.clone(),
            schema_fingerprint: request.schema_fingerprint.clone(),
            plan_fingerprint: request.plan_fingerprint.clone(),
            relation_fingerprint: request.relation_fingerprint.clone(),
            plan_id: envelope.plan.id.clone(),
            output_representation: request.output_representation.clone(),
            source_ids: request.source_ids.clone(),
            sample_count: envelope.coordinator_relations.as_ref().map(|relations| {
                relations
                    .records
                    .iter()
                    .map(|record| &record.sample_id)
                    .collect::<BTreeSet<_>>()
                    .len()
            }),
            relation_record_count: envelope
                .coordinator_relations
                .as_ref()
                .map(|relations| relations.records.len()),
        };
        self.records
            .borrow_mut()
            .insert(handle.handle, record.clone());
        Ok(record)
    }

    pub fn handle_record(&self, handle: u64) -> Option<CoordinatorDataHandleRecord> {
        self.records.borrow().get(&handle).cloned()
    }

    pub fn handle_records(&self) -> Vec<CoordinatorDataHandleRecord> {
        self.records.borrow().values().cloned().collect()
    }

    fn next_handle(&self) -> u64 {
        let mut next = self.next_handle.borrow_mut();
        let handle = *next;
        *next += 1;
        handle
    }
}

fn validate_request_against_envelope(
    envelope: &CoordinatorDataPlanEnvelope,
    request: &CoordinatorDataMaterializationRequest,
) -> Result<()> {
    if request.schema_fingerprint != envelope.schema_fingerprint {
        return Err(DataError::Validation(format!(
            "materialization request `{}` on `{}` schema fingerprint mismatch",
            request.input_name, request.node_id
        )));
    }
    if request.plan_fingerprint != envelope.plan_fingerprint {
        return Err(DataError::Validation(format!(
            "materialization request `{}` on `{}` plan fingerprint mismatch",
            request.input_name, request.node_id
        )));
    }
    if request.relation_fingerprint != envelope.relation_fingerprint {
        return Err(DataError::Validation(format!(
            "materialization request `{}` on `{}` relation fingerprint mismatch",
            request.input_name, request.node_id
        )));
    }
    if request.require_relations && envelope.coordinator_relations.is_none() {
        return Err(DataError::Validation(format!(
            "materialization request `{}` on `{}` requires coordinator relations",
            request.input_name, request.node_id
        )));
    }
    if request.output_representation != envelope.plan.output_representation {
        return Err(DataError::Validation(format!(
            "materialization request `{}` on `{}` output representation `{}` does not match plan output `{}`",
            request.input_name,
            request.node_id,
            request.output_representation,
            envelope.plan.output_representation
        )));
    }
    if !request.source_ids.is_empty() {
        let plan_sources = envelope
            .plan
            .steps
            .iter()
            .filter_map(|step| step.source_id.as_ref())
            .collect::<BTreeSet<_>>();
        for source_id in &request.source_ids {
            if !plan_sources.contains(source_id) {
                return Err(DataError::Validation(format!(
                    "materialization request `{}` on `{}` source `{}` is not present in data plan `{}`",
                    request.input_name, request.node_id, source_id, envelope.plan.id
                )));
            }
        }
    }
    Ok(())
}

fn validate_non_empty(label: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        return Err(DataError::Validation(format!("{label} must not be empty")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn envelope() -> CoordinatorDataPlanEnvelope {
        serde_json::from_str(include_str!(
            "../../../examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json"
        ))
        .unwrap()
    }

    fn request() -> CoordinatorDataMaterializationRequest {
        serde_json::from_str(include_str!(
            "../../../examples/fixtures/oof_campaign/materialization_request_model_base_x.json"
        ))
        .unwrap()
    }

    #[test]
    fn materializes_validated_coordinator_handle_record() {
        let arena = CoordinatorHandleArena::new("controller:data.provider").unwrap();
        let record = arena.materialize(&envelope(), &request()).unwrap();

        assert_eq!(record.handle.handle, 1);
        assert_eq!(record.handle.kind, CoordinatorHandleKind::Data);
        assert_eq!(record.input_name, "x");
        assert_eq!(record.plan_id, "nir-to-tabular");
        assert_eq!(record.sample_count, Some(2));
        assert_eq!(record.relation_record_count, Some(4));
        assert_eq!(arena.handle_record(1), Some(record));
        assert_eq!(arena.handle_records().len(), 1);
    }

    #[test]
    fn materialization_refuses_fingerprint_mismatch() {
        let arena = CoordinatorHandleArena::new("controller:data.provider").unwrap();
        let mut request = request();
        request.plan_fingerprint = "0".repeat(64);

        assert!(arena.materialize(&envelope(), &request).is_err());
    }
}
