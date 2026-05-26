use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::coordinator::CoordinatorRelationSet;
use crate::error::{DataError, Result};
use crate::handle::{CoordinatorFeatureBlock, CoordinatorFeatureTable};
use crate::ids::{ObservationId, RepresentationId, SourceId};

pub const NUMERIC_FEATURE_BUFFER_MANIFEST_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq)]
pub struct NumericFeatureBuffer {
    pub feature_set_id: String,
    pub representation_id: RepresentationId,
    pub feature_names: Vec<String>,
    pub observation_ids: Vec<ObservationId>,
    columns: Vec<Vec<Option<f64>>>,
    row_index_by_observation: BTreeMap<ObservationId, usize>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NumericFeatureBufferManifest {
    pub schema_version: u32,
    pub feature_set_id: String,
    pub representation_id: RepresentationId,
    pub feature_names: Vec<String>,
    pub observation_ids: Vec<ObservationId>,
    pub row_count: usize,
    pub feature_count: usize,
    pub value_count: usize,
    pub estimated_value_bytes: usize,
    pub buffer_fingerprint: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct NumericFeatureBufferBinding {
    pub feature_set_id: String,
    pub representation_id: RepresentationId,
    pub source_ids: Vec<SourceId>,
    pub row_count: usize,
    pub feature_count: usize,
    pub buffer_fingerprint: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct NumericFeatureBufferStore {
    buffers: BTreeMap<String, NumericFeatureBuffer>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct NumericFeatureBufferArena {
    store: NumericFeatureBufferStore,
    data_bindings: BTreeMap<u64, BTreeMap<String, NumericFeatureBufferBinding>>,
}

impl NumericFeatureBuffer {
    pub fn from_feature_table(table: CoordinatorFeatureTable) -> Result<Self> {
        table.validate()?;
        let row_count = table.rows.len();
        let mut observation_ids = Vec::with_capacity(row_count);
        let mut columns = (0..table.feature_names.len())
            .map(|_| Vec::with_capacity(row_count))
            .collect::<Vec<_>>();
        let mut row_index_by_observation = BTreeMap::new();

        for (row_idx, row) in table.rows.into_iter().enumerate() {
            if row_index_by_observation
                .insert(row.observation_id.clone(), row_idx)
                .is_some()
            {
                return Err(DataError::Validation(format!(
                    "feature table `{}` contains duplicate observation `{}`",
                    table.feature_set_id, row.observation_id
                )));
            }
            observation_ids.push(row.observation_id.clone());
            for (feature_idx, value) in row.values.into_iter().enumerate() {
                let feature_name = &table.feature_names[feature_idx];
                columns[feature_idx].push(numeric_feature_value(
                    &table.feature_set_id,
                    &row.observation_id,
                    feature_name,
                    value,
                )?);
            }
        }

        Ok(Self {
            feature_set_id: table.feature_set_id,
            representation_id: table.representation_id,
            feature_names: table.feature_names,
            observation_ids,
            columns,
            row_index_by_observation,
        })
    }

    pub fn row_count(&self) -> usize {
        self.observation_ids.len()
    }

    pub fn feature_count(&self) -> usize {
        self.feature_names.len()
    }

    pub fn value_count(&self) -> usize {
        self.row_count() * self.feature_count()
    }

    pub fn estimated_value_bytes(&self) -> usize {
        self.value_count() * std::mem::size_of::<f64>()
    }

    pub fn contains_observation(&self, observation_id: &ObservationId) -> bool {
        self.row_index_by_observation.contains_key(observation_id)
    }

    pub fn fingerprint(&self) -> Result<String> {
        #[derive(Serialize)]
        struct FingerprintPayload<'a> {
            feature_set_id: &'a str,
            representation_id: &'a RepresentationId,
            feature_names: &'a [String],
            observation_ids: &'a [ObservationId],
            columns: &'a [Vec<Option<f64>>],
        }

        let payload = FingerprintPayload {
            feature_set_id: &self.feature_set_id,
            representation_id: &self.representation_id,
            feature_names: &self.feature_names,
            observation_ids: &self.observation_ids,
            columns: &self.columns,
        };
        let json = serde_json::to_vec(&payload)?;
        let digest = Sha256::digest(json);
        Ok(to_hex(&digest))
    }

    pub fn manifest(&self) -> Result<NumericFeatureBufferManifest> {
        Ok(NumericFeatureBufferManifest {
            schema_version: NUMERIC_FEATURE_BUFFER_MANIFEST_SCHEMA_VERSION,
            feature_set_id: self.feature_set_id.clone(),
            representation_id: self.representation_id.clone(),
            feature_names: self.feature_names.clone(),
            observation_ids: self.observation_ids.clone(),
            row_count: self.row_count(),
            feature_count: self.feature_count(),
            value_count: self.value_count(),
            estimated_value_bytes: self.estimated_value_bytes(),
            buffer_fingerprint: self.fingerprint()?,
        })
    }

    pub fn selected_indices(&self, columns: Option<&[String]>) -> Result<Vec<usize>> {
        let index_by_name = self
            .feature_names
            .iter()
            .enumerate()
            .map(|(idx, name)| (name, idx))
            .collect::<BTreeMap<_, _>>();
        let indices = if let Some(columns) = columns {
            let mut seen = BTreeSet::new();
            columns
                .iter()
                .map(|column| {
                    if !seen.insert(column) {
                        return Err(DataError::Validation(format!(
                            "feature table `{}` selected duplicate feature column `{}`",
                            self.feature_set_id, column
                        )));
                    }
                    index_by_name.get(column).copied().ok_or_else(|| {
                        DataError::Validation(format!(
                            "feature table `{}` has no feature column `{}`",
                            self.feature_set_id, column
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            (0..self.feature_names.len()).collect()
        };
        if indices.is_empty() {
            return Err(DataError::Validation(format!(
                "feature table `{}` selected no feature columns",
                self.feature_set_id
            )));
        }
        Ok(indices)
    }

    pub fn project_relations(
        &self,
        relations: &CoordinatorRelationSet,
        source_id: Option<&SourceId>,
        columns: Option<&[String]>,
    ) -> Result<CoordinatorFeatureBlock> {
        relations.validate()?;
        let selected_indices = self.selected_indices(columns)?;
        let mut observation_ids = Vec::with_capacity(relations.records.len());
        let mut sample_ids = Vec::with_capacity(relations.records.len());
        let mut values = Vec::with_capacity(relations.records.len());

        for relation in relations.records.iter().filter(|relation| {
            source_id
                .map(|source_id| relation.source_id.as_ref() == Some(source_id))
                .unwrap_or(true)
        }) {
            let row_idx = self
                .row_index_by_observation
                .get(&relation.observation_id)
                .ok_or_else(|| {
                    DataError::Validation(format!(
                        "feature table `{}` has no row for observation `{}`",
                        self.feature_set_id, relation.observation_id
                    ))
                })?;
            observation_ids.push(relation.observation_id.clone());
            sample_ids.push(relation.sample_id.clone());
            values.push(
                selected_indices
                    .iter()
                    .map(|feature_idx| {
                        self.columns[*feature_idx][*row_idx]
                            .map_or(serde_json::Value::Null, serde_json::Value::from)
                    })
                    .collect(),
            );
        }

        Ok(CoordinatorFeatureBlock {
            feature_set_id: self.feature_set_id.clone(),
            representation_id: self.representation_id.clone(),
            feature_names: selected_indices
                .iter()
                .map(|idx| self.feature_names[*idx].clone())
                .collect(),
            observation_ids,
            sample_ids,
            values,
        })
    }

    fn binding_for_sources(
        &self,
        source_ids: Vec<SourceId>,
    ) -> Result<NumericFeatureBufferBinding> {
        Ok(NumericFeatureBufferBinding {
            feature_set_id: self.feature_set_id.clone(),
            representation_id: self.representation_id.clone(),
            source_ids,
            row_count: self.row_count(),
            feature_count: self.feature_count(),
            buffer_fingerprint: self.fingerprint()?,
        })
    }
}

impl NumericFeatureBufferStore {
    pub fn new(buffers: BTreeMap<String, NumericFeatureBuffer>) -> Result<Self> {
        for (feature_set_id, buffer) in &buffers {
            if feature_set_id != &buffer.feature_set_id {
                return Err(DataError::Validation(format!(
                    "feature buffer store key `{feature_set_id}` does not match buffer feature_set_id `{}`",
                    buffer.feature_set_id
                )));
            }
        }
        Ok(Self { buffers })
    }

    pub fn from_feature_tables(tables: Vec<CoordinatorFeatureTable>) -> Result<Self> {
        let mut buffers = BTreeMap::new();
        for table in tables {
            let feature_set_id = table.feature_set_id.clone();
            let buffer = NumericFeatureBuffer::from_feature_table(table)?;
            if buffers.insert(feature_set_id.clone(), buffer).is_some() {
                return Err(DataError::Validation(format!(
                    "duplicate feature table `{feature_set_id}`"
                )));
            }
        }
        Self::new(buffers)
    }

    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }

    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    pub fn get(&self, feature_set_id: &str) -> Option<&NumericFeatureBuffer> {
        self.buffers.get(feature_set_id)
    }

    pub fn manifests(&self) -> Result<Vec<NumericFeatureBufferManifest>> {
        self.buffers
            .values()
            .map(NumericFeatureBuffer::manifest)
            .collect()
    }

    pub fn bindings_for_relations(
        &self,
        relations: &CoordinatorRelationSet,
        representation_id: &RepresentationId,
    ) -> Result<Vec<NumericFeatureBufferBinding>> {
        relations.validate()?;
        let source_ids = relations
            .records
            .iter()
            .filter_map(|relation| relation.source_id.as_ref())
            .collect::<BTreeSet<_>>();

        let mut bindings = Vec::new();
        for buffer in self.buffers.values() {
            if &buffer.representation_id != representation_id {
                continue;
            }
            let mut covered_sources = Vec::new();
            if source_ids.is_empty() {
                if relations
                    .records
                    .iter()
                    .all(|relation| buffer.contains_observation(&relation.observation_id))
                {
                    bindings.push(buffer.binding_for_sources(Vec::new())?);
                }
                continue;
            }
            for source_id in &source_ids {
                let source_records = relations
                    .records
                    .iter()
                    .filter(|relation| relation.source_id.as_ref() == Some(*source_id));
                if source_records
                    .clone()
                    .all(|relation| buffer.contains_observation(&relation.observation_id))
                {
                    covered_sources.push((*source_id).clone());
                }
            }
            if !covered_sources.is_empty() {
                bindings.push(buffer.binding_for_sources(covered_sources)?);
            }
        }
        Ok(bindings)
    }

    pub fn project_relations(
        &self,
        feature_set_id: &str,
        relations: &CoordinatorRelationSet,
        source_id: Option<&SourceId>,
        columns: Option<&[String]>,
    ) -> Result<CoordinatorFeatureBlock> {
        let buffer = self.buffers.get(feature_set_id).ok_or_else(|| {
            DataError::Validation(format!("unknown feature buffer `{feature_set_id}`"))
        })?;
        buffer.project_relations(relations, source_id, columns)
    }
}

impl NumericFeatureBufferArena {
    pub fn new(store: NumericFeatureBufferStore) -> Self {
        Self {
            store,
            data_bindings: BTreeMap::new(),
        }
    }

    pub fn manifests(&self) -> Result<Vec<NumericFeatureBufferManifest>> {
        self.store.manifests()
    }

    pub fn bind_data_handle(
        &mut self,
        data_handle: u64,
        relations: &CoordinatorRelationSet,
        representation_id: &RepresentationId,
    ) -> Result<Vec<NumericFeatureBufferBinding>> {
        let bindings = self
            .store
            .bindings_for_relations(relations, representation_id)?;
        self.data_bindings.insert(
            data_handle,
            bindings
                .iter()
                .cloned()
                .map(|binding| (binding.feature_set_id.clone(), binding))
                .collect(),
        );
        Ok(bindings)
    }

    pub fn release_data_handle(&mut self, data_handle: u64) -> bool {
        self.data_bindings.remove(&data_handle).is_some()
    }

    pub fn bindings_for_data_handle(
        &self,
        data_handle: u64,
    ) -> Result<Vec<NumericFeatureBufferBinding>> {
        let bindings = self.data_bindings.get(&data_handle).ok_or_else(|| {
            DataError::Validation(format!(
                "data handle `{data_handle}` has no feature buffer bindings"
            ))
        })?;
        Ok(bindings.values().cloned().collect())
    }

    pub fn project_bound_relations(
        &self,
        data_handle: u64,
        feature_set_id: &str,
        relations: &CoordinatorRelationSet,
        source_id: Option<&SourceId>,
        columns: Option<&[String]>,
    ) -> Result<CoordinatorFeatureBlock> {
        self.validate_bound_sources(data_handle, feature_set_id, relations, source_id)?;
        self.store
            .project_relations(feature_set_id, relations, source_id, columns)
    }

    fn validate_bound_sources(
        &self,
        data_handle: u64,
        feature_set_id: &str,
        relations: &CoordinatorRelationSet,
        source_id: Option<&SourceId>,
    ) -> Result<()> {
        relations.validate()?;
        let binding = self
            .data_bindings
            .get(&data_handle)
            .and_then(|bindings| bindings.get(feature_set_id))
            .ok_or_else(|| {
                DataError::Validation(format!(
                    "feature buffer `{feature_set_id}` is not bound to data handle `{data_handle}`"
                ))
            })?;
        let relation_source_ids = relations
            .records
            .iter()
            .filter_map(|relation| relation.source_id.as_ref())
            .cloned()
            .collect::<BTreeSet<_>>();
        let required_source_ids = if let Some(source_id) = source_id {
            if relation_source_ids.is_empty() || !relation_source_ids.contains(source_id) {
                return Err(DataError::Validation(format!(
                    "feature buffer `{feature_set_id}` source `{source_id}` is not present in view for data handle `{data_handle}`"
                )));
            }
            vec![source_id.clone()]
        } else {
            relation_source_ids.into_iter().collect::<Vec<_>>()
        };
        for source_id in &required_source_ids {
            if !binding.source_ids.contains(source_id) {
                return Err(DataError::Validation(format!(
                    "feature buffer `{feature_set_id}` is not bound to source `{source_id}` for data handle `{data_handle}`"
                )));
            }
        }
        Ok(())
    }
}

fn numeric_feature_value(
    feature_set_id: &str,
    observation_id: &ObservationId,
    feature_name: &str,
    value: serde_json::Value,
) -> Result<Option<f64>> {
    match value {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::Number(number) => number.as_f64().map(Some).ok_or_else(|| {
            DataError::Validation(format!(
                "feature table `{feature_set_id}` row `{observation_id}` feature `{feature_name}` contains a non-f64 numeric value"
            ))
        }),
        _ => Err(DataError::Validation(format!(
            "feature table `{feature_set_id}` row `{observation_id}` feature `{feature_name}` must be numeric or null"
        ))),
    }
}

fn to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        write!(&mut out, "{byte:02x}").expect("writing to string cannot fail");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coordinator::CoordinatorRelation;
    use crate::handle::CoordinatorFeatureRow;
    use crate::ids::{SampleId, TargetId};

    fn oid(value: &str) -> ObservationId {
        ObservationId::new(value).unwrap()
    }

    fn sid(value: &str) -> SampleId {
        SampleId::new(value).unwrap()
    }

    fn source(value: &str) -> SourceId {
        SourceId::new(value).unwrap()
    }

    fn table() -> CoordinatorFeatureTable {
        CoordinatorFeatureTable {
            feature_set_id: "x".to_string(),
            representation_id: RepresentationId::new("tabular_numeric").unwrap(),
            feature_names: vec!["f0".to_string(), "f1".to_string()],
            rows: vec![
                CoordinatorFeatureRow {
                    observation_id: oid("obs.s1.nir"),
                    values: vec![serde_json::json!(1.0), serde_json::json!(10.0)],
                },
                CoordinatorFeatureRow {
                    observation_id: oid("obs.s1.chem"),
                    values: vec![serde_json::json!(2.0), serde_json::json!(20.0)],
                },
                CoordinatorFeatureRow {
                    observation_id: oid("obs.s2.nir"),
                    values: vec![serde_json::json!(3.0), serde_json::Value::Null],
                },
            ],
        }
    }

    fn relations() -> CoordinatorRelationSet {
        CoordinatorRelationSet {
            records: vec![
                relation("obs.s2.nir", "S2", "nir"),
                relation("obs.s1.nir", "S1", "nir"),
                relation("obs.s1.chem", "S1", "chem"),
            ],
        }
    }

    fn relation(observation_id: &str, sample_id: &str, source_id: &str) -> CoordinatorRelation {
        CoordinatorRelation {
            observation_id: oid(observation_id),
            sample_id: sid(sample_id),
            target_id: Some(TargetId::new("y").unwrap()),
            group_id: None,
            origin_sample_id: None,
            source_id: Some(source(source_id)),
            is_augmented: false,
        }
    }

    #[test]
    fn projects_view_relations_from_columnar_numeric_buffer() {
        let buffer = NumericFeatureBuffer::from_feature_table(table()).unwrap();
        assert_eq!(buffer.row_count(), 3);
        assert_eq!(buffer.feature_count(), 2);
        assert_eq!(buffer.value_count(), 6);
        let manifest = buffer.manifest().unwrap();
        assert_eq!(
            manifest.schema_version,
            NUMERIC_FEATURE_BUFFER_MANIFEST_SCHEMA_VERSION
        );
        assert_eq!(manifest.row_count, 3);
        assert_eq!(manifest.feature_count, 2);
        assert_eq!(manifest.value_count, 6);
        assert_eq!(manifest.estimated_value_bytes, 48);
        assert_eq!(manifest.buffer_fingerprint.len(), 64);

        let block = buffer
            .project_relations(
                &relations(),
                Some(&source("nir")),
                Some(&["f1".to_string()]),
            )
            .unwrap();

        assert_eq!(block.feature_set_id, "x");
        assert_eq!(block.feature_names, vec!["f1".to_string()]);
        assert_eq!(
            block.observation_ids,
            vec![oid("obs.s2.nir"), oid("obs.s1.nir")]
        );
        assert_eq!(block.sample_ids, vec![sid("S2"), sid("S1")]);
        assert_eq!(
            block.values,
            vec![vec![serde_json::Value::Null], vec![serde_json::json!(10.0)]]
        );
    }

    #[test]
    fn rejects_duplicate_selected_columns() {
        let buffer = NumericFeatureBuffer::from_feature_table(table()).unwrap();
        let error = buffer
            .selected_indices(Some(&["f0".to_string(), "f0".to_string()]))
            .unwrap_err();
        assert!(format!("{error}").contains("duplicate feature column"));
    }

    #[test]
    fn rejects_missing_observation_in_projection() {
        let buffer = NumericFeatureBuffer::from_feature_table(table()).unwrap();
        let missing = CoordinatorRelationSet {
            records: vec![relation("obs.missing", "S9", "nir")],
        };
        let error = buffer.project_relations(&missing, None, None).unwrap_err();
        assert!(format!("{error}").contains("has no row for observation"));
    }

    #[test]
    fn store_manifests_and_projects_by_feature_set_id() {
        let store = NumericFeatureBufferStore::from_feature_tables(vec![table()]).unwrap();
        assert_eq!(store.len(), 1);
        assert!(!store.is_empty());

        let manifests = store.manifests().unwrap();
        assert_eq!(manifests.len(), 1);
        assert_eq!(manifests[0].feature_set_id, "x");
        assert_eq!(manifests[0].feature_names, vec!["f0", "f1"]);

        let block = store
            .project_relations("x", &relations(), Some(&source("chem")), None)
            .unwrap();
        assert_eq!(block.observation_ids, vec![oid("obs.s1.chem")]);
        assert_eq!(
            block.values,
            vec![vec![serde_json::json!(2.0), serde_json::json!(20.0)]]
        );
    }

    #[test]
    fn store_derives_source_bindings_from_relation_coverage() {
        let store = NumericFeatureBufferStore::from_feature_tables(vec![table()]).unwrap();
        let bindings = store
            .bindings_for_relations(
                &relations(),
                &RepresentationId::new("tabular_numeric").unwrap(),
            )
            .unwrap();

        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].feature_set_id, "x");
        assert_eq!(bindings[0].source_ids, vec![source("chem"), source("nir")]);
        assert_eq!(bindings[0].row_count, 3);
        assert_eq!(bindings[0].feature_count, 2);
        assert_eq!(bindings[0].buffer_fingerprint.len(), 64);

        let wrong_representation = store
            .bindings_for_relations(
                &relations(),
                &RepresentationId::new("dense_signal").unwrap(),
            )
            .unwrap();
        assert!(wrong_representation.is_empty());
    }

    #[test]
    fn arena_binds_projects_and_releases_data_handle_buffers() {
        let store = NumericFeatureBufferStore::from_feature_tables(vec![table()]).unwrap();
        let mut arena = NumericFeatureBufferArena::new(store);
        let bindings = arena
            .bind_data_handle(
                7,
                &relations(),
                &RepresentationId::new("tabular_numeric").unwrap(),
            )
            .unwrap();

        assert_eq!(bindings.len(), 1);
        assert_eq!(arena.bindings_for_data_handle(7).unwrap(), bindings);

        let block = arena
            .project_bound_relations(7, "x", &relations(), Some(&source("nir")), None)
            .unwrap();
        assert_eq!(
            block.observation_ids,
            vec![oid("obs.s2.nir"), oid("obs.s1.nir")]
        );

        let error = arena
            .project_bound_relations(8, "x", &relations(), Some(&source("nir")), None)
            .unwrap_err();
        assert!(format!("{error}").contains("not bound to data handle"));

        assert!(arena.release_data_handle(7));
        let error = arena.bindings_for_data_handle(7).unwrap_err();
        assert!(format!("{error}").contains("no feature buffer bindings"));
    }

    #[test]
    fn store_refuses_duplicate_feature_sets() {
        let error =
            NumericFeatureBufferStore::from_feature_tables(vec![table(), table()]).unwrap_err();
        assert!(format!("{error}").contains("duplicate feature table"));
    }
}
