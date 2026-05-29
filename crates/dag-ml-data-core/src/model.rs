use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{DataError, Result};
use crate::ids::{GroupId, RepresentationId, SampleId, SourceId, TargetId, TypeId};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
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
    Wavenumber,
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

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum SignalKind {
    Absorbance,
    Reflectance,
    Transmittance,
    LogReflectance,
    Preprocessed,
    Unknown,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct AxisSizeContract {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exact: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<usize>,
}

impl AxisSizeContract {
    pub fn validate(&self, axis_name: &str) -> Result<()> {
        if self.exact.is_none() && self.min.is_none() && self.max.is_none() {
            return Err(DataError::Validation(format!(
                "shape contract for axis `{axis_name}` does not constrain the size"
            )));
        }
        if let (Some(min), Some(max)) = (self.min, self.max) {
            if min > max {
                return Err(DataError::Validation(format!(
                    "shape contract for axis `{axis_name}` has min {min} greater than max {max}"
                )));
            }
        }
        if let Some(exact) = self.exact {
            if let Some(min) = self.min {
                if exact < min {
                    return Err(DataError::Validation(format!(
                        "shape contract for axis `{axis_name}` exact size {exact} is below min {min}"
                    )));
                }
            }
            if let Some(max) = self.max {
                if exact > max {
                    return Err(DataError::Validation(format!(
                        "shape contract for axis `{axis_name}` exact size {exact} is above max {max}"
                    )));
                }
            }
        }
        Ok(())
    }

    fn accepts(&self, size: usize) -> bool {
        if self.exact.is_some_and(|exact| size != exact) {
            return false;
        }
        if self.min.is_some_and(|min| size < min) {
            return false;
        }
        if self.max.is_some_and(|max| size > max) {
            return false;
        }
        true
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct ShapeContract {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank: Option<usize>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub axis_sizes: BTreeMap<String, AxisSizeContract>,
    #[serde(default)]
    pub allow_ragged: bool,
}

impl ShapeContract {
    pub fn validate(&self) -> Result<()> {
        if self.rank.is_none() && self.axis_sizes.is_empty() {
            return Err(DataError::Validation(
                "shape contract must constrain rank or at least one axis".to_string(),
            ));
        }
        for (axis_name, contract) in &self.axis_sizes {
            if axis_name.trim().is_empty() {
                return Err(DataError::Validation(
                    "shape contract contains an empty axis name".to_string(),
                ));
            }
            contract.validate(axis_name)?;
        }
        Ok(())
    }

    pub fn validate_representation(
        &self,
        source_id: &SourceId,
        representation: &RepresentationSpec,
    ) -> Result<()> {
        self.validate()?;
        if let Some(expected_rank) = self.rank {
            if representation.rank != Some(expected_rank) {
                return Err(DataError::Validation(format!(
                    "source `{source_id}` shape contract expects rank {expected_rank} but representation `{}` has {:?}",
                    representation.id, representation.rank
                )));
            }
        }
        if representation.ragged && !self.allow_ragged {
            return Err(DataError::Validation(format!(
                "source `{source_id}` shape contract does not allow ragged representation `{}`",
                representation.id
            )));
        }
        for (axis_name, contract) in &self.axis_sizes {
            let axis = representation
                .axes
                .iter()
                .find(|axis| axis.name == *axis_name)
                .ok_or_else(|| {
                    DataError::Validation(format!(
                        "source `{source_id}` shape contract references missing axis `{axis_name}`"
                    ))
                })?;
            if let Some(size) = axis.size {
                if !contract.accepts(size) {
                    return Err(DataError::Validation(format!(
                        "source `{source_id}` axis `{axis_name}` size {size} violates shape contract"
                    )));
                }
            } else if !axis.variable {
                return Err(DataError::Validation(format!(
                    "source `{source_id}` axis `{axis_name}` has no concrete size for shape contract"
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signal_type: Option<SignalKind>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shape_contract: Option<ShapeContract>,
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
        self.native_representation.validate()?;
        if let Some(shape_contract) = &self.shape_contract {
            shape_contract.validate_representation(&self.id, &self.native_representation)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetadataValueKind {
    String,
    Number,
    Integer,
    Boolean,
    Date,
    Datetime,
    Categorical,
    Json,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MetadataFieldSpec {
    pub kind: MetadataValueKind,
    #[serde(default)]
    pub required: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_values: Vec<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

impl MetadataFieldSpec {
    pub fn validate(&self, field_name: &str) -> Result<()> {
        if self.kind == MetadataValueKind::Categorical && self.allowed_values.is_empty() {
            return Err(DataError::Validation(format!(
                "metadata field `{field_name}` is categorical but declares no allowed_values"
            )));
        }
        if let Some(unit) = &self.unit {
            if unit.trim().is_empty() {
                return Err(DataError::Validation(format!(
                    "metadata field `{field_name}` has an empty unit"
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MetadataSchema {
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub fields: BTreeMap<String, MetadataFieldSpec>,
}

impl MetadataSchema {
    pub fn validate(&self) -> Result<()> {
        if self.fields.is_empty() {
            return Err(DataError::Validation(
                "metadata schema declares no fields".to_string(),
            ));
        }
        for (field_name, field) in &self.fields {
            if field_name.trim().is_empty() {
                return Err(DataError::Validation(
                    "metadata schema contains an empty field name".to_string(),
                ));
            }
            field.validate(field_name)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GroupKind {
    RepetitionGroup,
    Subject,
    Batch,
    Split,
    Custom,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GroupSpec {
    pub id: GroupId,
    pub kind: GroupKind,
    pub column: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_id: Option<SourceId>,
    #[serde(default)]
    pub strict: bool,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl GroupSpec {
    pub fn validate(&self) -> Result<()> {
        if self.column.trim().is_empty() {
            return Err(DataError::Validation(format!(
                "group `{}` has an empty column",
                self.id
            )));
        }
        for key in self.metadata.keys() {
            if key.trim().is_empty() {
                return Err(DataError::Validation(format!(
                    "group `{}` metadata contains an empty key",
                    self.id
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FoldSpec {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_id: Option<GroupId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split_column: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl FoldSpec {
    pub fn validate(&self) -> Result<()> {
        if self.id.trim().is_empty() {
            return Err(DataError::Validation("fold id is empty".to_string()));
        }
        if self.group_id.is_none() && self.split_column.is_none() {
            return Err(DataError::Validation(format!(
                "fold `{}` declares neither group_id nor split_column",
                self.id
            )));
        }
        if let Some(split_column) = &self.split_column {
            if split_column.trim().is_empty() {
                return Err(DataError::Validation(format!(
                    "fold `{}` has an empty split_column",
                    self.id
                )));
            }
        }
        for key in self.metadata.keys() {
            if key.trim().is_empty() {
                return Err(DataError::Validation(format!(
                    "fold `{}` metadata contains an empty key",
                    self.id
                )));
            }
        }
        Ok(())
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata_schema: Option<MetadataSchema>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<GroupSpec>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub folds: Vec<FoldSpec>,
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
        if let Some(metadata_schema) = &self.metadata_schema {
            metadata_schema.validate()?;
        }
        let mut group_ids = BTreeSet::new();
        for group in &self.groups {
            if !group_ids.insert(&group.id) {
                return Err(DataError::Validation(format!(
                    "duplicate group id `{}`",
                    group.id
                )));
            }
            if let Some(source_id) = &group.source_id {
                if !source_ids.contains(source_id) {
                    return Err(DataError::Validation(format!(
                        "group `{}` references unknown source `{source_id}`",
                        group.id
                    )));
                }
            }
            group.validate()?;
        }
        let mut fold_ids = BTreeSet::new();
        for fold in &self.folds {
            if !fold_ids.insert(&fold.id) {
                return Err(DataError::Validation(format!(
                    "duplicate fold id `{}`",
                    fold.id
                )));
            }
            if let Some(group_id) = &fold.group_id {
                if !group_ids.contains(group_id) {
                    return Err(DataError::Validation(format!(
                        "fold `{}` references unknown group `{group_id}`",
                        fold.id
                    )));
                }
            }
            fold.validate()?;
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
            signal_type: None,
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
            signal_type: None,
        };

        assert!(repr.validate().is_ok());
    }

    #[test]
    fn axis_kind_wavenumber_serializes_and_round_trips() {
        let value = AxisKind::Wavenumber;
        let json = serde_json::to_string(&value).unwrap();
        assert_eq!(json, "\"wavenumber\"");
        let decoded: AxisKind = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, value);
    }

    #[test]
    fn axis_kind_wavenumber_accepted_in_representation_axis() {
        let axes = vec![
            sample_axis(),
            AxisSpec {
                name: "wavenumber".to_string(),
                kind: AxisKind::Wavenumber,
                unit: Some("cm-1".to_string()),
                size: Some(1024),
                variable: false,
                coordinates: None,
            },
        ];
        let repr = RepresentationSpec {
            id: RepresentationId::new("ftir_spectrum").unwrap(),
            type_id: TypeId::new("dense_signal").unwrap(),
            rank: Some(2),
            axes,
            container: "ndarray".to_string(),
            dtype: Some("float64".to_string()),
            sparse: false,
            ragged: false,
            signal_type: Some(SignalKind::Absorbance),
        };
        repr.validate().unwrap();
    }

    #[test]
    fn dataset_schema_accepts_optional_nirs4all_integration_contracts() {
        let source_id = SourceId::new("nir").unwrap();
        let group_id = GroupId::new("rep.group").unwrap();
        let representation = RepresentationSpec {
            id: RepresentationId::new("nir.signal").unwrap(),
            type_id: TypeId::new("dense_signal").unwrap(),
            rank: Some(2),
            axes: vec![
                sample_axis(),
                AxisSpec {
                    name: "wavelength".to_string(),
                    kind: AxisKind::Wavelength,
                    unit: Some("nm".to_string()),
                    size: Some(3),
                    variable: false,
                    coordinates: None,
                },
            ],
            container: "ndarray".to_string(),
            dtype: Some("float32".to_string()),
            sparse: false,
            ragged: false,
            signal_type: Some(SignalKind::Reflectance),
        };
        let schema = DatasetSchema {
            dataset_id: "nirs4all-lite-smoke".to_string(),
            sample_ids: vec![SampleId::new("s1").unwrap(), SampleId::new("s2").unwrap()],
            sources: vec![SourceDescriptor {
                id: source_id.clone(),
                name: "NIR spectra".to_string(),
                type_id: TypeId::new("dense_signal").unwrap(),
                modality: "nir".to_string(),
                native_representation: representation,
                sample_key: "sample_id".to_string(),
                granularity: SourceGranularity::PerSampleRepeated,
                schema: BTreeMap::new(),
                tags: BTreeMap::new(),
                shape_contract: Some(ShapeContract {
                    rank: Some(2),
                    axis_sizes: BTreeMap::from([(
                        "wavelength".to_string(),
                        AxisSizeContract {
                            exact: Some(3),
                            min: None,
                            max: None,
                        },
                    )]),
                    allow_ragged: false,
                }),
            }],
            targets: BTreeMap::new(),
            metadata: BTreeMap::new(),
            metadata_schema: Some(MetadataSchema {
                fields: BTreeMap::from([(
                    "cultivar".to_string(),
                    MetadataFieldSpec {
                        kind: MetadataValueKind::Categorical,
                        required: true,
                        unit: None,
                        allowed_values: vec![serde_json::Value::String("a".to_string())],
                        description: None,
                    },
                )]),
            }),
            groups: vec![GroupSpec {
                id: group_id.clone(),
                kind: GroupKind::RepetitionGroup,
                column: "sample_id".to_string(),
                source_id: Some(source_id),
                strict: true,
                metadata: BTreeMap::new(),
            }],
            folds: vec![FoldSpec {
                id: "cv.repetition.safe".to_string(),
                group_id: Some(group_id),
                split_column: Some("fold_id".to_string()),
                metadata: BTreeMap::new(),
            }],
        };

        schema.validate().unwrap();
        let json = serde_json::to_value(&schema).unwrap();
        assert_eq!(
            json["sources"][0]["native_representation"]["signal_type"],
            "reflectance"
        );
        assert_eq!(json["groups"][0]["kind"], "repetition_group");
    }

    #[test]
    fn dataset_schema_refuses_shape_contract_mismatch() {
        let representation = RepresentationSpec {
            id: RepresentationId::new("nir.signal").unwrap(),
            type_id: TypeId::new("dense_signal").unwrap(),
            rank: Some(2),
            axes: vec![
                sample_axis(),
                AxisSpec {
                    name: "wavelength".to_string(),
                    kind: AxisKind::Wavelength,
                    unit: Some("nm".to_string()),
                    size: Some(3),
                    variable: false,
                    coordinates: None,
                },
            ],
            container: "ndarray".to_string(),
            dtype: Some("float32".to_string()),
            sparse: false,
            ragged: false,
            signal_type: Some(SignalKind::Absorbance),
        };
        let source = SourceDescriptor {
            id: SourceId::new("nir").unwrap(),
            name: "NIR spectra".to_string(),
            type_id: TypeId::new("dense_signal").unwrap(),
            modality: "nir".to_string(),
            native_representation: representation,
            sample_key: "sample_id".to_string(),
            granularity: SourceGranularity::PerSample,
            schema: BTreeMap::new(),
            tags: BTreeMap::new(),
            shape_contract: Some(ShapeContract {
                rank: Some(2),
                axis_sizes: BTreeMap::from([(
                    "wavelength".to_string(),
                    AxisSizeContract {
                        exact: Some(4),
                        min: None,
                        max: None,
                    },
                )]),
                allow_ragged: false,
            }),
        };

        assert!(source.validate().is_err());
    }

    #[test]
    fn dataset_schema_refuses_empty_shape_contract() {
        let representation = RepresentationSpec {
            id: RepresentationId::new("nir.signal").unwrap(),
            type_id: TypeId::new("dense_signal").unwrap(),
            rank: Some(2),
            axes: vec![
                sample_axis(),
                AxisSpec {
                    name: "wavelength".to_string(),
                    kind: AxisKind::Wavelength,
                    unit: Some("nm".to_string()),
                    size: Some(3),
                    variable: false,
                    coordinates: None,
                },
            ],
            container: "ndarray".to_string(),
            dtype: Some("float32".to_string()),
            sparse: false,
            ragged: false,
            signal_type: None,
        };
        let source = SourceDescriptor {
            id: SourceId::new("nir").unwrap(),
            name: "NIR spectra".to_string(),
            type_id: TypeId::new("dense_signal").unwrap(),
            modality: "nir".to_string(),
            native_representation: representation,
            sample_key: "sample_id".to_string(),
            granularity: SourceGranularity::PerSample,
            schema: BTreeMap::new(),
            tags: BTreeMap::new(),
            shape_contract: Some(ShapeContract::default()),
        };

        assert!(source.validate().is_err());
    }

    #[test]
    fn dataset_schema_refuses_unknown_fold_group() {
        let schema = DatasetSchema {
            dataset_id: "folds".to_string(),
            sample_ids: vec![SampleId::new("s1").unwrap()],
            sources: Vec::new(),
            targets: BTreeMap::new(),
            metadata: BTreeMap::new(),
            metadata_schema: None,
            groups: Vec::new(),
            folds: vec![FoldSpec {
                id: "fold.cv".to_string(),
                group_id: Some(GroupId::new("missing").unwrap()),
                split_column: None,
                metadata: BTreeMap::new(),
            }],
        };

        assert!(schema.validate().is_err());
    }

    #[test]
    fn dataset_schema_refuses_empty_fold_declaration() {
        let schema = DatasetSchema {
            dataset_id: "folds".to_string(),
            sample_ids: vec![SampleId::new("s1").unwrap()],
            sources: Vec::new(),
            targets: BTreeMap::new(),
            metadata: BTreeMap::new(),
            metadata_schema: None,
            groups: Vec::new(),
            folds: vec![FoldSpec {
                id: "fold.cv".to_string(),
                group_id: None,
                split_column: None,
                metadata: BTreeMap::new(),
            }],
        };

        let error = schema.validate().unwrap_err();
        assert!(error
            .to_string()
            .contains("neither group_id nor split_column"));
    }
}
