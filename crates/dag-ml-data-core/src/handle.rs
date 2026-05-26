use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::coordinator::{
    validate_fingerprint, CoordinatorDataPlanEnvelope, CoordinatorRelation, CoordinatorRelationSet,
};
use crate::error::{DataError, Result};
use crate::ids::{RepresentationId, SampleId, SourceId, TargetId};
use crate::model::DataView;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoordinatorHandleKind {
    Data,
    View,
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CoordinatorDataViewRecord {
    pub handle: CoordinatorHandleRef,
    pub parent_handle: CoordinatorHandleRef,
    pub view: DataView,
    pub sample_count: usize,
    pub relation_record_count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoordinatorTargetValue {
    pub sample_id: SampleId,
    pub value: serde_json::Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoordinatorTargetTable {
    pub target_id: TargetId,
    pub values: Vec<CoordinatorTargetValue>,
}

impl CoordinatorTargetTable {
    pub fn validate(&self) -> Result<()> {
        if self.values.is_empty() {
            return Err(DataError::Validation(format!(
                "target table `{}` contains no values",
                self.target_id
            )));
        }
        let mut seen = BTreeSet::new();
        for value in &self.values {
            if !seen.insert(&value.sample_id) {
                return Err(DataError::Validation(format!(
                    "target table `{}` contains duplicate sample `{}`",
                    self.target_id, value.sample_id
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoordinatorTargetBlock {
    pub target_id: TargetId,
    pub sample_ids: Vec<SampleId>,
    pub values: Vec<serde_json::Value>,
}

#[derive(Debug)]
pub struct CoordinatorHandleArena {
    owner_controller: String,
    next_handle: RefCell<u64>,
    records: RefCell<BTreeMap<u64, CoordinatorDataHandleRecord>>,
    data_relations: RefCell<BTreeMap<u64, CoordinatorRelationSet>>,
    view_records: RefCell<BTreeMap<u64, CoordinatorDataViewRecord>>,
    view_relations: RefCell<BTreeMap<u64, CoordinatorRelationSet>>,
}

impl CoordinatorHandleArena {
    pub fn new(owner_controller: impl Into<String>) -> Result<Self> {
        let owner_controller = owner_controller.into();
        validate_non_empty("owner_controller", &owner_controller)?;
        Ok(Self {
            owner_controller,
            next_handle: RefCell::new(1),
            records: RefCell::new(BTreeMap::new()),
            data_relations: RefCell::new(BTreeMap::new()),
            view_records: RefCell::new(BTreeMap::new()),
            view_relations: RefCell::new(BTreeMap::new()),
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
        if let Some(relations) = envelope.coordinator_relations.clone() {
            self.data_relations
                .borrow_mut()
                .insert(handle.handle, relations);
        }
        Ok(record)
    }

    pub fn make_view(
        &self,
        data_handle: u64,
        view: &DataView,
    ) -> Result<CoordinatorDataViewRecord> {
        validate_view(view)?;
        let parent = self
            .records
            .borrow()
            .get(&data_handle)
            .cloned()
            .ok_or_else(|| DataError::Validation(format!("unknown data handle `{data_handle}`")))?;
        let relations = self
            .data_relations
            .borrow()
            .get(&data_handle)
            .cloned()
            .ok_or_else(|| {
                DataError::Validation(format!(
                    "data handle `{data_handle}` has no coordinator relations"
                ))
            })?;
        let filtered = filter_relations(&relations.records, view)?;
        let sample_count = unique_sample_count(&filtered);
        let relation_record_count = filtered.len();
        let handle = CoordinatorHandleRef {
            handle: self.next_handle(),
            kind: CoordinatorHandleKind::View,
            owner_controller: self.owner_controller.clone(),
        };
        let record = CoordinatorDataViewRecord {
            handle: handle.clone(),
            parent_handle: parent.handle,
            view: view.clone(),
            sample_count,
            relation_record_count,
        };
        self.view_records
            .borrow_mut()
            .insert(handle.handle, record.clone());
        self.view_relations
            .borrow_mut()
            .insert(handle.handle, CoordinatorRelationSet { records: filtered });
        Ok(record)
    }

    pub fn view_record(&self, handle: u64) -> Option<CoordinatorDataViewRecord> {
        self.view_records.borrow().get(&handle).cloned()
    }

    pub fn view_identity(&self, handle: u64) -> Result<CoordinatorRelationSet> {
        self.view_relations
            .borrow()
            .get(&handle)
            .cloned()
            .ok_or_else(|| DataError::Validation(format!("unknown view handle `{handle}`")))
    }

    pub fn release_handle(&self, handle: u64) -> bool {
        if self.view_records.borrow_mut().remove(&handle).is_some() {
            self.view_relations.borrow_mut().remove(&handle);
            return true;
        }
        if let Some(record) = self.records.borrow_mut().remove(&handle) {
            self.data_relations.borrow_mut().remove(&handle);
            let child_views = self
                .view_records
                .borrow()
                .iter()
                .filter_map(|(view_handle, view_record)| {
                    (view_record.parent_handle == record.handle).then_some(*view_handle)
                })
                .collect::<Vec<_>>();
            for view_handle in child_views {
                self.view_records.borrow_mut().remove(&view_handle);
                self.view_relations.borrow_mut().remove(&view_handle);
            }
            return true;
        }
        false
    }

    pub fn target_values(
        &self,
        view_handle: u64,
        target_table: &CoordinatorTargetTable,
    ) -> Result<CoordinatorTargetBlock> {
        target_table.validate()?;
        let relations = self.view_identity(view_handle)?;
        let values_by_sample = target_table
            .values
            .iter()
            .map(|value| (&value.sample_id, &value.value))
            .collect::<BTreeMap<_, _>>();
        let mut seen_samples = BTreeSet::new();
        let mut sample_ids = Vec::new();
        let mut values = Vec::new();
        for relation in relations.records.iter().filter(|relation| {
            relation
                .target_id
                .as_ref()
                .map(|target_id| target_id == &target_table.target_id)
                .unwrap_or(true)
        }) {
            if !seen_samples.insert(&relation.sample_id) {
                continue;
            }
            let value = values_by_sample.get(&relation.sample_id).ok_or_else(|| {
                DataError::Validation(format!(
                    "target table `{}` has no value for sample `{}`",
                    target_table.target_id, relation.sample_id
                ))
            })?;
            sample_ids.push(relation.sample_id.clone());
            values.push((*value).clone());
        }
        if sample_ids.is_empty() {
            return Err(DataError::Validation(format!(
                "view `{view_handle}` contains no samples for target `{}`",
                target_table.target_id
            )));
        }
        Ok(CoordinatorTargetBlock {
            target_id: target_table.target_id.clone(),
            sample_ids,
            values,
        })
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

fn validate_view(view: &DataView) -> Result<()> {
    if let Some(samples) = &view.sample_ids {
        let unique = samples.iter().collect::<BTreeSet<_>>();
        if unique.len() != samples.len() {
            return Err(DataError::Validation(
                "data view contains duplicate sample ids".to_string(),
            ));
        }
    }
    if let Some(sources) = &view.source_ids {
        let unique = sources.iter().collect::<BTreeSet<_>>();
        if unique.len() != sources.len() {
            return Err(DataError::Validation(
                "data view contains duplicate source ids".to_string(),
            ));
        }
    }
    if let Some(columns) = &view.columns {
        let unique = columns.iter().collect::<BTreeSet<_>>();
        if unique.len() != columns.len() {
            return Err(DataError::Validation(
                "data view contains duplicate columns".to_string(),
            ));
        }
        if columns.iter().any(|column| column.trim().is_empty()) {
            return Err(DataError::Validation(
                "data view contains an empty column".to_string(),
            ));
        }
    }
    Ok(())
}

fn filter_relations(
    relations: &[CoordinatorRelation],
    view: &DataView,
) -> Result<Vec<CoordinatorRelation>> {
    let sample_filter = view
        .sample_ids
        .as_ref()
        .map(|sample_ids| sample_ids.iter().collect::<BTreeSet<_>>());
    let source_filter = view
        .source_ids
        .as_ref()
        .map(|source_ids| source_ids.iter().collect::<BTreeSet<_>>());
    let filtered = relations
        .iter()
        .filter(|relation| {
            sample_filter
                .as_ref()
                .map(|samples| samples.contains(&relation.sample_id))
                .unwrap_or(true)
        })
        .filter(|relation| {
            source_filter
                .as_ref()
                .map(|sources| {
                    relation
                        .source_id
                        .as_ref()
                        .map(|source_id| sources.contains(source_id))
                        .unwrap_or(false)
                })
                .unwrap_or(true)
        })
        .filter(|relation| view.include_augmented || !relation.is_augmented)
        .cloned()
        .collect::<Vec<_>>();
    if filtered.is_empty() {
        return Err(DataError::Validation(
            "data view selected no coordinator relations".to_string(),
        ));
    }
    Ok(filtered)
}

fn unique_sample_count(relations: &[CoordinatorRelation]) -> usize {
    relations
        .iter()
        .map(|relation| &relation.sample_id)
        .collect::<BTreeSet<_>>()
        .len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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

    #[test]
    fn view_filters_augmented_rows_and_preserves_repetition_identity() {
        let arena = CoordinatorHandleArena::new("controller:data.provider").unwrap();
        let data = arena.materialize(&envelope(), &request()).unwrap();
        let view = DataView {
            sample_ids: Some(vec![SampleId::new("S001").unwrap()]),
            include_augmented: false,
            ..Default::default()
        };

        let view_record = arena.make_view(data.handle.handle, &view).unwrap();
        let identity = arena.view_identity(view_record.handle.handle).unwrap();

        assert_eq!(view_record.handle.kind, CoordinatorHandleKind::View);
        assert_eq!(view_record.sample_count, 1);
        assert_eq!(view_record.relation_record_count, 2);
        assert_eq!(identity.records.len(), 2);
        assert_eq!(identity.records[0].observation_id.as_str(), "obs.S001.base");
        assert_eq!(identity.records[1].observation_id.as_str(), "obs.S001.rep1");
        assert_eq!(
            arena.view_record(view_record.handle.handle),
            Some(view_record)
        );
    }

    #[test]
    fn target_values_are_sample_level_and_dedup_repetitions() {
        let arena = CoordinatorHandleArena::new("controller:data.provider").unwrap();
        let data = arena.materialize(&envelope(), &request()).unwrap();
        let view = DataView {
            sample_ids: Some(vec![SampleId::new("S001").unwrap()]),
            include_augmented: false,
            ..Default::default()
        };
        let view_record = arena.make_view(data.handle.handle, &view).unwrap();
        let target_table = CoordinatorTargetTable {
            target_id: TargetId::new("y").unwrap(),
            values: vec![
                CoordinatorTargetValue {
                    sample_id: SampleId::new("S001").unwrap(),
                    value: json!(42.0),
                },
                CoordinatorTargetValue {
                    sample_id: SampleId::new("S002").unwrap(),
                    value: json!(7.0),
                },
            ],
        };

        let target = arena
            .target_values(view_record.handle.handle, &target_table)
            .unwrap();

        assert_eq!(target.target_id.as_str(), "y");
        assert_eq!(target.sample_ids, vec![SampleId::new("S001").unwrap()]);
        assert_eq!(target.values, vec![json!(42.0)]);
    }

    #[test]
    fn release_data_handle_releases_child_views() {
        let arena = CoordinatorHandleArena::new("controller:data.provider").unwrap();
        let data = arena.materialize(&envelope(), &request()).unwrap();
        let view_record = arena
            .make_view(data.handle.handle, &DataView::default())
            .unwrap();

        assert!(arena.release_handle(data.handle.handle));
        assert_eq!(arena.handle_record(data.handle.handle), None);
        assert_eq!(arena.view_record(view_record.handle.handle), None);
        assert!(arena.view_identity(view_record.handle.handle).is_err());
        assert!(!arena.release_handle(data.handle.handle));
    }
}
