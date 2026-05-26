use std::collections::{BTreeMap, BTreeSet};

use crate::coordinator::CoordinatorRelationSet;
use crate::error::{DataError, Result};
use crate::handle::{CoordinatorFeatureBlock, CoordinatorFeatureTable};
use crate::ids::{ObservationId, RepresentationId, SourceId};

#[derive(Clone, Debug, PartialEq)]
pub struct NumericFeatureBuffer {
    pub feature_set_id: String,
    pub representation_id: RepresentationId,
    pub feature_names: Vec<String>,
    pub observation_ids: Vec<ObservationId>,
    columns: Vec<Vec<Option<f64>>>,
    row_index_by_observation: BTreeMap<ObservationId, usize>,
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
}
