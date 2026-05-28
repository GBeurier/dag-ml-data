use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{DataError, Result};
use crate::ids::{RepresentationId, SampleId, SourceId, TargetId, TypeId};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AxisKind {
    Sample,
    Feature,
    Processing,
    Time,
    Height,
    Width,
    Channel,
    Node,
    Edge,
    Variant,
    Token,
    Target,
    Wavelength,
    Frequency,
    Depth,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AxisSpec {
    pub name: String,
    pub kind: AxisKind,
    pub unit: Option<String>,
    pub size: Option<usize>,
    #[serde(default)]
    pub variable: bool,
    #[serde(default)]
    pub coordinates: Option<Vec<serde_json::Value>>,
}

impl AxisSpec {
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(DataError::Validation("axis name is empty".to_string()));
        }
        if self.variable && self.size.is_some() {
            return Err(DataError::Validation(format!(
                "axis `{}` cannot be both variable and sized",
                self.name
            )));
        }
        if let (Some(size), Some(coordinates)) = (self.size, &self.coordinates) {
            if coordinates.len() != size {
                return Err(DataError::Validation(format!(
                    "axis `{}` has {} coordinates for size {}",
                    self.name,
                    coordinates.len(),
                    size
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepresentationSpec {
    pub id: RepresentationId,
    pub type_id: TypeId,
    pub rank: Option<usize>,
    pub axes: Vec<AxisSpec>,
    pub container: String,
    pub dtype: Option<String>,
    #[serde(default)]
    pub sparse: bool,
    #[serde(default)]
    pub ragged: bool,
}

impl RepresentationSpec {
    pub fn validate(&self) -> Result<()> {
        if self.container.trim().is_empty() {
            return Err(DataError::Validation(format!(
                "representation `{}` has an empty container",
                self.id
            )));
        }
        if self.rank.is_none() && !self.ragged {
            return Err(DataError::Validation(format!(
                "representation `{}` with no rank must be ragged",
                self.id
            )));
        }
        if let Some(rank) = self.rank {
            if self.axes.len() != rank {
                return Err(DataError::Validation(format!(
                    "representation `{}` has rank {} but {} axes",
                    self.id,
                    rank,
                    self.axes.len()
                )));
            }
        }
        for axis in &self.axes {
            axis.validate()?;
        }
        if self.container != "graph_batch"
            && !self.axes.iter().any(|axis| axis.kind == AxisKind::Sample)
        {
            return Err(DataError::Validation(format!(
                "representation `{}` has no sample axis",
                self.id
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceGranularity {
    PerSample,
    PerSampleRepeated,
    PerSampleSequence,
    PerSampleSet,
    PerGroup,
    PerTarget,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SourceDescriptor {
    pub id: SourceId,
    pub name: String,
    pub type_id: TypeId,
    pub modality: String,
    pub native_representation: RepresentationSpec,
    pub sample_key: String,
    pub granularity: SourceGranularity,
    #[serde(default)]
    pub schema: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub tags: BTreeMap<String, serde_json::Value>,
}

impl SourceDescriptor {
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(DataError::Validation(format!(
                "source `{}` has an empty name",
                self.id
            )));
        }
        if self.sample_key.trim().is_empty() {
            return Err(DataError::Validation(format!(
                "source `{}` has an empty sample key",
                self.id
            )));
        }
        self.native_representation.validate()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DatasetSchema {
    pub dataset_id: String,
    pub sample_ids: Vec<SampleId>,
    pub sources: Vec<SourceDescriptor>,
    #[serde(default)]
    pub targets: BTreeMap<TargetId, RepresentationSpec>,
    #[serde(default)]
    pub metadata: BTreeMap<String, RepresentationSpec>,
}

impl DatasetSchema {
    pub fn validate(&self) -> Result<()> {
        if self.dataset_id.trim().is_empty() {
            return Err(DataError::Validation(
                "dataset id must not be empty".to_string(),
            ));
        }
        if self.sample_ids.is_empty() {
            return Err(DataError::Validation(
                "dataset schema must contain at least one sample".to_string(),
            ));
        }
        let unique_samples = self.sample_ids.iter().collect::<BTreeSet<_>>();
        if unique_samples.len() != self.sample_ids.len() {
            return Err(DataError::Validation(
                "dataset schema contains duplicate sample ids".to_string(),
            ));
        }

        let mut source_ids = BTreeSet::new();
        for source in &self.sources {
            if !source_ids.insert(&source.id) {
                return Err(DataError::Validation(format!(
                    "duplicate source id `{}`",
                    source.id
                )));
            }
            source.validate()?;
        }
        for target in self.targets.values() {
            target.validate()?;
        }
        for representation in self.metadata.values() {
            representation.validate()?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct DataView {
    pub sample_ids: Option<Vec<SampleId>>,
    pub partition: Option<String>,
    pub fold_id: Option<String>,
    pub source_ids: Option<Vec<SourceId>>,
    pub columns: Option<Vec<String>>,
    #[serde(default = "default_true")]
    pub include_augmented: bool,
    #[serde(default)]
    pub include_excluded: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch_view: Option<crate::coordinator::CoordinatorBranchView>,
    #[serde(default)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PresenceMask {
    pub sample_ids: Vec<SampleId>,
    pub source_id: SourceId,
    pub present: Vec<bool>,
}

impl PresenceMask {
    pub fn validate(&self) -> Result<()> {
        if self.sample_ids.len() != self.present.len() {
            return Err(DataError::Validation(format!(
                "presence mask for `{}` has {} sample ids but {} flags",
                self.source_id,
                self.sample_ids.len(),
                self.present.len()
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_axis() -> AxisSpec {
        AxisSpec {
            name: "sample".to_string(),
            kind: AxisKind::Sample,
            unit: None,
            size: Some(2),
            variable: false,
            coordinates: None,
        }
    }

    #[test]
    fn rejects_representation_without_sample_axis() {
        let repr = RepresentationSpec {
            id: RepresentationId::new("tabular").unwrap(),
            type_id: TypeId::new("table").unwrap(),
            rank: Some(1),
            axes: vec![AxisSpec {
                name: "feature".to_string(),
                kind: AxisKind::Feature,
                unit: None,
                size: Some(3),
                variable: false,
                coordinates: None,
            }],
            container: "dataframe".to_string(),
            dtype: Some("float32".to_string()),
            sparse: false,
            ragged: false,
        };

        assert!(repr.validate().is_err());
    }

    #[test]
    fn accepts_sample_major_representation() {
        let repr = RepresentationSpec {
            id: RepresentationId::new("tabular").unwrap(),
            type_id: TypeId::new("table").unwrap(),
            rank: Some(1),
            axes: vec![sample_axis()],
            container: "dataframe".to_string(),
            dtype: Some("float32".to_string()),
            sparse: false,
            ragged: false,
        };

        assert!(repr.validate().is_ok());
    }
}
