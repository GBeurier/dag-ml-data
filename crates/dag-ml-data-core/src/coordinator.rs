use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{DataError, Result};
use crate::fingerprint::{data_plan_fingerprint, sample_relation_fingerprint, schema_fingerprint};
use crate::ids::{GroupId, ObservationId, SampleId, SourceId, TargetId};
use crate::model::DatasetSchema;
use crate::plan::DataPlan;
use crate::relation::SampleRelationTable;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct CoordinatorRelation {
    pub observation_id: ObservationId,
    pub sample_id: SampleId,
    #[serde(default)]
    pub target_id: Option<TargetId>,
    #[serde(default)]
    pub group_id: Option<GroupId>,
    #[serde(default)]
    pub origin_sample_id: Option<SampleId>,
    #[serde(default)]
    pub source_id: Option<SourceId>,
    #[serde(default)]
    pub is_augmented: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct CoordinatorRelationSet {
    #[serde(default)]
    pub records: Vec<CoordinatorRelation>,
}

impl CoordinatorRelationSet {
    pub fn validate(&self) -> Result<()> {
        if self.records.is_empty() {
            return Err(DataError::Validation(
                "coordinator relation set contains no records".to_string(),
            ));
        }
        let mut seen = std::collections::BTreeSet::new();
        for record in &self.records {
            if !seen.insert(&record.observation_id) {
                return Err(DataError::Validation(format!(
                    "duplicate coordinator observation `{}`",
                    record.observation_id
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CoordinatorDataPlanEnvelope {
    pub schema_fingerprint: String,
    pub plan_fingerprint: String,
    #[serde(default)]
    pub relation_fingerprint: Option<String>,
    pub plan: DataPlan,
    #[serde(default)]
    pub coordinator_relations: Option<CoordinatorRelationSet>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl CoordinatorDataPlanEnvelope {
    pub fn from_parts(
        schema: &DatasetSchema,
        plan: DataPlan,
        relations: Option<&SampleRelationTable>,
    ) -> Result<Self> {
        let schema_fingerprint = schema_fingerprint(schema)?;
        let plan_fingerprint = data_plan_fingerprint(&plan)?;
        let relation_fingerprint = relations.map(sample_relation_fingerprint).transpose()?;
        let coordinator_relations = relations
            .map(coordinator_relations_from_sample_table)
            .transpose()?;
        let envelope = Self {
            schema_fingerprint,
            plan_fingerprint,
            relation_fingerprint,
            plan,
            coordinator_relations,
            metadata: BTreeMap::new(),
        };
        envelope.validate()?;
        Ok(envelope)
    }

    pub fn validate(&self) -> Result<()> {
        validate_fingerprint("schema", &self.schema_fingerprint)?;
        validate_fingerprint("plan", &self.plan_fingerprint)?;
        self.plan.validate()?;
        let actual_plan = data_plan_fingerprint(&self.plan)?;
        if actual_plan != self.plan_fingerprint {
            return Err(DataError::Validation(format!(
                "data plan fingerprint mismatch: envelope has {}, actual is {}",
                self.plan_fingerprint, actual_plan
            )));
        }
        if let Some(relations) = &self.coordinator_relations {
            relations.validate()?;
        }
        if let Some(relation_fingerprint) = &self.relation_fingerprint {
            validate_fingerprint("relation", relation_fingerprint)?;
            if self.coordinator_relations.is_none() {
                return Err(DataError::Validation(
                    "relation_fingerprint requires coordinator_relations".to_string(),
                ));
            }
        }
        Ok(())
    }
}

pub fn coordinator_relations_from_sample_table(
    relations: &SampleRelationTable,
) -> Result<CoordinatorRelationSet> {
    relations.validate()?;
    let observation_to_sample = relations
        .rows
        .iter()
        .map(|row| (&row.observation_id, &row.sample_id))
        .collect::<BTreeMap<_, _>>();
    let mut records = relations
        .rows
        .iter()
        .map(|row| {
            let origin_sample_id = row
                .origin_id
                .as_ref()
                .map(|origin_id| {
                    observation_to_sample
                        .iter()
                        .find_map(|(observation_id, sample_id)| {
                            (observation_id.as_str() == origin_id.as_str())
                                .then_some((*sample_id).clone())
                        })
                        .ok_or_else(|| {
                            DataError::Validation(format!(
                                "origin `{origin_id}` is not present as an observation"
                            ))
                        })
                })
                .transpose()?;
            Ok(CoordinatorRelation {
                observation_id: row.observation_id.clone(),
                sample_id: row.sample_id.clone(),
                target_id: row.target_id.clone(),
                group_id: row.group_id.clone(),
                origin_sample_id,
                source_id: row.source_id.clone(),
                is_augmented: row.augmented,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    records.sort_by(|left, right| left.observation_id.cmp(&right.observation_id));
    let converted = CoordinatorRelationSet { records };
    converted.validate()?;
    Ok(converted)
}

pub(crate) fn validate_fingerprint(label: &str, value: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(DataError::Validation(format!(
            "{label} fingerprint must be a 64-character hex digest"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_schema() -> DatasetSchema {
        serde_json::from_str(include_str!(
            "../../../examples/fixtures/oof_campaign/schema_nir_6_samples.json"
        ))
        .unwrap()
    }

    fn load_plan() -> DataPlan {
        serde_json::from_str(include_str!(
            "../../../examples/fixtures/oof_campaign/expected_data_plan_nir_to_tabular.json"
        ))
        .unwrap()
    }

    fn load_relations() -> SampleRelationTable {
        serde_json::from_str(include_str!(
            "../../../examples/fixtures/oof_campaign/sample_relations_grouped_augmented.json"
        ))
        .unwrap()
    }

    #[test]
    fn converts_data_relations_to_coordinator_relations() {
        let converted = coordinator_relations_from_sample_table(&load_relations()).unwrap();

        let augmented = converted
            .records
            .iter()
            .find(|record| record.observation_id.as_str() == "obs.S001.aug0")
            .unwrap();
        assert_eq!(
            augmented.origin_sample_id.as_ref().map(ToString::to_string),
            Some("S001".to_string())
        );
        assert!(augmented.is_augmented);
    }

    #[test]
    fn envelope_validates_fingerprints_and_payloads() {
        let envelope = CoordinatorDataPlanEnvelope::from_parts(
            &load_schema(),
            load_plan(),
            Some(&load_relations()),
        )
        .unwrap();

        envelope.validate().unwrap();
        assert!(envelope.coordinator_relations.is_some());
    }

    #[test]
    fn envelope_refuses_plan_fingerprint_mismatch() {
        let mut envelope =
            CoordinatorDataPlanEnvelope::from_parts(&load_schema(), load_plan(), None).unwrap();
        envelope.plan_fingerprint = "0".repeat(64);

        assert!(envelope.validate().is_err());
    }

    #[test]
    fn fixture_envelope_validates() {
        let envelope: CoordinatorDataPlanEnvelope = serde_json::from_str(include_str!(
            "../../../examples/fixtures/oof_campaign/coordinator_data_plan_envelope_nir.json"
        ))
        .unwrap();

        envelope.validate().unwrap();
    }
}
