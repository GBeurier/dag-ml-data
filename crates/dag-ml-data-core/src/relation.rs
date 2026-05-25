use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{DataError, Result};
use crate::ids::{GroupId, ObservationId, OriginId, RepetitionId, SampleId, SourceId, TargetId};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct SampleRelation {
    pub observation_id: ObservationId,
    pub sample_id: SampleId,
    pub source_id: Option<SourceId>,
    pub target_id: Option<TargetId>,
    pub group_id: Option<GroupId>,
    pub origin_id: Option<OriginId>,
    pub repetition_id: Option<RepetitionId>,
    #[serde(default)]
    pub augmented: bool,
    #[serde(default)]
    pub excluded: bool,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct SampleRelationTable {
    pub rows: Vec<SampleRelation>,
}

impl SampleRelationTable {
    pub fn validate(&self) -> Result<()> {
        if self.rows.is_empty() {
            return Err(DataError::Validation(
                "sample relation table contains no rows".to_string(),
            ));
        }

        let mut observation_ids = BTreeSet::new();
        let mut origin_ids = BTreeSet::new();
        let mut sample_groups = BTreeMap::<&SampleId, &GroupId>::new();
        for row in &self.rows {
            if !observation_ids.insert(&row.observation_id) {
                return Err(DataError::Validation(format!(
                    "duplicate observation id `{}`",
                    row.observation_id
                )));
            }
            if let Some(group_id) = &row.group_id {
                if let Some(previous_group_id) = sample_groups.insert(&row.sample_id, group_id) {
                    if previous_group_id != group_id {
                        return Err(DataError::Validation(format!(
                            "sample `{}` appears with conflicting groups `{}` and `{}`",
                            row.sample_id, previous_group_id, group_id
                        )));
                    }
                }
            }
            if row.augmented && row.origin_id.is_none() {
                return Err(DataError::Validation(format!(
                    "augmented observation `{}` has no origin_id",
                    row.observation_id
                )));
            }
            if !row.augmented && row.origin_id.is_some() {
                return Err(DataError::Validation(format!(
                    "non-augmented observation `{}` declares origin_id",
                    row.observation_id
                )));
            }
            if let Some(origin_id) = &row.origin_id {
                if origin_id.as_str() == row.observation_id.as_str() {
                    return Err(DataError::Validation(format!(
                        "observation `{}` cannot be its own origin",
                        row.observation_id
                    )));
                }
                origin_ids.insert(origin_id);
            }
        }

        for origin_id in origin_ids {
            if !observation_ids
                .iter()
                .any(|observation_id| observation_id.as_str() == origin_id.as_str())
            {
                return Err(DataError::Validation(format!(
                    "origin `{origin_id}` is not present as an observation"
                )));
            }
        }

        Ok(())
    }

    pub fn sample_groups(&self) -> BTreeMap<SampleId, GroupId> {
        self.rows
            .iter()
            .filter_map(|row| {
                row.group_id
                    .as_ref()
                    .map(|group_id| (row.sample_id.clone(), group_id.clone()))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(value: &str) -> ObservationId {
        ObservationId::new(value).unwrap()
    }

    fn sample(value: &str) -> SampleId {
        SampleId::new(value).unwrap()
    }

    fn origin(value: &str) -> OriginId {
        OriginId::new(value).unwrap()
    }

    fn row(observation_id: &str, sample_id: &str) -> SampleRelation {
        SampleRelation {
            observation_id: obs(observation_id),
            sample_id: sample(sample_id),
            source_id: None,
            target_id: None,
            group_id: None,
            origin_id: None,
            repetition_id: None,
            augmented: false,
            excluded: false,
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn validates_group_and_origin_relations() {
        let mut base = row("obs1", "s1");
        base.group_id = Some(GroupId::new("g1").unwrap());
        let mut augmented = row("obs1_aug", "s1");
        augmented.group_id = Some(GroupId::new("g1").unwrap());
        augmented.origin_id = Some(origin("obs1"));
        augmented.augmented = true;

        let table = SampleRelationTable {
            rows: vec![base, augmented],
        };

        table.validate().unwrap();
        assert_eq!(
            table.sample_groups().get(&sample("s1")),
            Some(&GroupId::new("g1").unwrap())
        );
    }

    #[test]
    fn rejects_duplicate_observations() {
        let table = SampleRelationTable {
            rows: vec![row("obs1", "s1"), row("obs1", "s2")],
        };

        assert!(table.validate().is_err());
    }

    #[test]
    fn rejects_augmented_rows_without_known_origin() {
        let mut augmented = row("obs1_aug", "s1");
        augmented.origin_id = Some(origin("missing"));
        augmented.augmented = true;
        let table = SampleRelationTable {
            rows: vec![augmented],
        };

        assert!(table.validate().is_err());
    }

    #[test]
    fn grouped_augmented_fixture_validates() {
        let table: SampleRelationTable = serde_json::from_str(include_str!(
            "../../../examples/fixtures/oof_campaign/sample_relations_grouped_augmented.json"
        ))
        .unwrap();

        table.validate().unwrap();
        assert_eq!(table.sample_groups().len(), 2);
    }
}
