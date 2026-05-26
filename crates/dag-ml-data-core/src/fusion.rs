use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::alignment::{SampleAlignmentPlan, SourceSampleSet};
use crate::error::{DataError, Result};
use crate::handle::CoordinatorFeatureBlock;
use crate::ids::{ObservationId, SampleId, SourceId};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FeatureFusionPolicy {
    #[serde(default = "default_true")]
    pub namespace_columns: bool,
}

impl Default for FeatureFusionPolicy {
    fn default() -> Self {
        Self {
            namespace_columns: true,
        }
    }
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SourceFeatureBlock {
    pub source_id: SourceId,
    pub block: CoordinatorFeatureBlock,
}

pub fn source_sample_set_from_feature_block(block: &SourceFeatureBlock) -> Result<SourceSampleSet> {
    validate_feature_block(&block.block)?;
    let mut seen = BTreeSet::new();
    let mut sample_ids = Vec::new();
    for sample_id in &block.block.sample_ids {
        if seen.insert(sample_id) {
            sample_ids.push(sample_id.clone());
        }
    }
    Ok(SourceSampleSet {
        source_id: block.source_id.clone(),
        sample_ids,
    })
}

pub fn fuse_feature_blocks(
    feature_set_id: impl Into<String>,
    blocks: &[SourceFeatureBlock],
    alignment: &SampleAlignmentPlan,
    policy: &FeatureFusionPolicy,
) -> Result<CoordinatorFeatureBlock> {
    let feature_set_id = feature_set_id.into();
    if feature_set_id.trim().is_empty() {
        return Err(DataError::Validation(
            "fused feature set id is empty".to_string(),
        ));
    }
    if blocks.is_empty() {
        return Err(DataError::Validation(
            "feature fusion requires at least one source block".to_string(),
        ));
    }
    alignment.validate()?;

    let mut source_ids = BTreeSet::new();
    for block in blocks {
        if !source_ids.insert(&block.source_id) {
            return Err(DataError::Validation(format!(
                "feature fusion contains duplicate source `{}`",
                block.source_id
            )));
        }
        validate_feature_block(&block.block)?;
    }
    for mask in &alignment.masks {
        if !source_ids.contains(&mask.source_id) {
            return Err(DataError::Validation(format!(
                "alignment mask references source `{}` absent from feature fusion",
                mask.source_id
            )));
        }
    }
    if alignment.masks.len() != blocks.len() {
        return Err(DataError::Validation(
            "feature fusion sources and alignment masks differ".to_string(),
        ));
    }

    let representation_id = blocks[0].block.representation_id.clone();
    for block in blocks.iter().skip(1) {
        if block.block.representation_id != representation_id {
            return Err(DataError::Validation(format!(
                "feature fusion source `{}` representation `{}` does not match `{}`",
                block.source_id, block.block.representation_id, representation_id
            )));
        }
    }

    let mut feature_names = Vec::new();
    for block in blocks {
        for feature_name in &block.block.feature_names {
            let output_name = if policy.namespace_columns {
                format!("{}.{}", block.source_id, feature_name)
            } else {
                feature_name.clone()
            };
            feature_names.push(output_name);
        }
    }
    let mut seen_features = BTreeSet::new();
    for feature_name in &feature_names {
        if !seen_features.insert(feature_name) {
            return Err(DataError::Validation(format!(
                "feature fusion produced duplicate feature `{feature_name}`"
            )));
        }
    }

    let row_maps = blocks
        .iter()
        .map(|block| (&block.source_id, rows_by_sample(&block.block)))
        .collect::<BTreeMap<_, _>>();
    validate_alignment_presence(blocks, alignment, &row_maps)?;
    let reference = &blocks[0];
    let reference_rows = row_maps
        .get(&reference.source_id)
        .expect("reference source map was created");

    let mut observation_ids = Vec::new();
    let mut sample_ids = Vec::new();
    let mut values = Vec::new();

    for sample_id in &alignment.sample_ids {
        let output_rows = reference_rows
            .get(sample_id)
            .map(|indices| {
                indices
                    .iter()
                    .map(|idx| OutputRow::Reference(*idx))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_else(|| vec![OutputRow::Synthetic]);

        for output_row in output_rows {
            let mut row_values = Vec::new();
            match output_row {
                OutputRow::Reference(idx) => {
                    observation_ids.push(reference.block.observation_ids[idx].clone());
                    sample_ids.push(sample_id.clone());
                }
                OutputRow::Synthetic => {
                    observation_ids.push(synthetic_observation_id(sample_id)?);
                    sample_ids.push(sample_id.clone());
                }
            }
            for block in blocks {
                let source_rows = row_maps
                    .get(&block.source_id)
                    .expect("source row map was created");
                if block.source_id == reference.source_id {
                    match output_row {
                        OutputRow::Reference(idx) => {
                            row_values.extend(block.block.values[idx].iter().cloned());
                        }
                        OutputRow::Synthetic => {
                            row_values.extend(std::iter::repeat_n(
                                serde_json::Value::Null,
                                block.block.feature_names.len(),
                            ));
                        }
                    }
                    continue;
                }

                match source_rows.get(sample_id).map(Vec::as_slice) {
                    Some([idx]) => row_values.extend(block.block.values[*idx].iter().cloned()),
                    Some(indices) => {
                        return Err(DataError::Validation(format!(
                            "feature fusion cannot broadcast {} repeated rows from non-reference source `{}` for sample `{sample_id}`",
                            indices.len(),
                            block.source_id
                        )));
                    }
                    None => row_values.extend(std::iter::repeat_n(
                        serde_json::Value::Null,
                        block.block.feature_names.len(),
                    )),
                }
            }
            values.push(row_values);
        }
    }

    let fused = CoordinatorFeatureBlock {
        feature_set_id,
        representation_id,
        feature_names,
        observation_ids,
        sample_ids,
        values,
    };
    validate_feature_block(&fused)?;
    Ok(fused)
}

#[derive(Clone, Copy)]
enum OutputRow {
    Reference(usize),
    Synthetic,
}

fn validate_feature_block(block: &CoordinatorFeatureBlock) -> Result<()> {
    if block.feature_set_id.trim().is_empty() {
        return Err(DataError::Validation(
            "feature block feature_set_id is empty".to_string(),
        ));
    }
    if block.feature_names.is_empty() {
        return Err(DataError::Validation(format!(
            "feature block `{}` contains no features",
            block.feature_set_id
        )));
    }
    if block.observation_ids.len() != block.sample_ids.len()
        || block.sample_ids.len() != block.values.len()
    {
        return Err(DataError::Validation(format!(
            "feature block `{}` row identity/value lengths differ",
            block.feature_set_id
        )));
    }
    let mut observations = BTreeSet::new();
    for (idx, values) in block.values.iter().enumerate() {
        if !observations.insert(&block.observation_ids[idx]) {
            return Err(DataError::Validation(format!(
                "feature block `{}` contains duplicate observation `{}`",
                block.feature_set_id, block.observation_ids[idx]
            )));
        }
        if values.len() != block.feature_names.len() {
            return Err(DataError::Validation(format!(
                "feature block `{}` row `{}` has {} values for {} features",
                block.feature_set_id,
                block.observation_ids[idx],
                values.len(),
                block.feature_names.len()
            )));
        }
    }
    Ok(())
}

fn rows_by_sample(block: &CoordinatorFeatureBlock) -> BTreeMap<&SampleId, Vec<usize>> {
    let mut rows = BTreeMap::<&SampleId, Vec<usize>>::new();
    for (idx, sample_id) in block.sample_ids.iter().enumerate() {
        rows.entry(sample_id).or_default().push(idx);
    }
    rows
}

fn validate_alignment_presence<'a>(
    blocks: &'a [SourceFeatureBlock],
    alignment: &SampleAlignmentPlan,
    row_maps: &BTreeMap<&'a SourceId, BTreeMap<&'a SampleId, Vec<usize>>>,
) -> Result<()> {
    for (idx, sample_id) in alignment.sample_ids.iter().enumerate() {
        if !alignment.masks.iter().any(|mask| mask.present[idx]) {
            return Err(DataError::Validation(format!(
                "alignment sample `{sample_id}` is absent from every fused source"
            )));
        }
    }
    for block in blocks {
        let mask = alignment
            .masks
            .iter()
            .find(|mask| mask.source_id == block.source_id)
            .expect("alignment sources were checked before presence validation");
        let rows = row_maps
            .get(&block.source_id)
            .expect("source row map was created");
        for (sample_id, present) in alignment.sample_ids.iter().zip(mask.present.iter()) {
            let has_rows = rows.contains_key(sample_id);
            if *present != has_rows {
                return Err(DataError::Validation(format!(
                    "alignment presence for source `{}` sample `{sample_id}` is {present} but feature block rows are {}",
                    block.source_id,
                    if has_rows { "present" } else { "absent" }
                )));
            }
        }
    }
    Ok(())
}

fn synthetic_observation_id(sample_id: &SampleId) -> Result<ObservationId> {
    ObservationId::new(format!("fused.{}", sample_id.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alignment::{build_sample_alignment_plan, AlignmentMode, AlignmentPolicy};
    use crate::ids::RepresentationId;
    use serde_json::json;

    fn block(
        source_id: &str,
        feature_names: &[&str],
        rows: &[(&str, &str, Vec<serde_json::Value>)],
    ) -> SourceFeatureBlock {
        SourceFeatureBlock {
            source_id: SourceId::new(source_id).unwrap(),
            block: CoordinatorFeatureBlock {
                feature_set_id: source_id.to_string(),
                representation_id: RepresentationId::new("tabular_numeric").unwrap(),
                feature_names: feature_names.iter().map(ToString::to_string).collect(),
                observation_ids: rows
                    .iter()
                    .map(|(observation_id, _, _)| ObservationId::new(*observation_id).unwrap())
                    .collect(),
                sample_ids: rows
                    .iter()
                    .map(|(_, sample_id, _)| SampleId::new(*sample_id).unwrap())
                    .collect(),
                values: rows.iter().map(|(_, _, values)| values.clone()).collect(),
            },
        }
    }

    fn alignment(blocks: &[SourceFeatureBlock], mode: AlignmentMode) -> SampleAlignmentPlan {
        let source_sets = blocks
            .iter()
            .map(source_sample_set_from_feature_block)
            .collect::<Result<Vec<_>>>()
            .unwrap();
        build_sample_alignment_plan(&source_sets, &AlignmentPolicy { mode }).unwrap()
    }

    #[test]
    fn fusion_broadcasts_singleton_source_to_reference_repetitions() {
        let blocks = vec![
            block(
                "nir",
                &["n0"],
                &[
                    ("obs.S001.r1", "S001", vec![json!(1.0)]),
                    ("obs.S001.r2", "S001", vec![json!(2.0)]),
                    ("obs.S002.r1", "S002", vec![json!(3.0)]),
                ],
            ),
            block(
                "chem",
                &["c0"],
                &[
                    ("chem.S001", "S001", vec![json!(10.0)]),
                    ("chem.S002", "S002", vec![json!(20.0)]),
                ],
            ),
        ];
        let fused = fuse_feature_blocks(
            "fused",
            &blocks,
            &alignment(&blocks, AlignmentMode::Inner),
            &FeatureFusionPolicy::default(),
        )
        .unwrap();

        assert_eq!(fused.feature_names, vec!["nir.n0", "chem.c0"]);
        assert_eq!(fused.sample_ids, blocks[0].block.sample_ids);
        assert_eq!(
            fused.values,
            vec![
                vec![json!(1.0), json!(10.0)],
                vec![json!(2.0), json!(10.0)],
                vec![json!(3.0), json!(20.0)]
            ]
        );
    }

    #[test]
    fn outer_fusion_creates_synthetic_rows_for_samples_missing_in_reference() {
        let blocks = vec![
            block("nir", &["n0"], &[("obs.S001.r1", "S001", vec![json!(1.0)])]),
            block(
                "chem",
                &["c0"],
                &[
                    ("chem.S001", "S001", vec![json!(10.0)]),
                    ("chem.S002", "S002", vec![json!(20.0)]),
                ],
            ),
        ];
        let fused = fuse_feature_blocks(
            "fused",
            &blocks,
            &alignment(&blocks, AlignmentMode::Outer),
            &FeatureFusionPolicy::default(),
        )
        .unwrap();

        assert_eq!(
            fused
                .observation_ids
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>(),
            vec!["obs.S001.r1", "fused.S002"]
        );
        assert_eq!(
            fused.values,
            vec![
                vec![json!(1.0), json!(10.0)],
                vec![serde_json::Value::Null, json!(20.0)]
            ]
        );
    }

    #[test]
    fn fusion_refuses_ambiguous_non_reference_repetitions() {
        let blocks = vec![
            block("chem", &["c0"], &[("chem.S001", "S001", vec![json!(10.0)])]),
            block(
                "nir",
                &["n0"],
                &[
                    ("obs.S001.r1", "S001", vec![json!(1.0)]),
                    ("obs.S001.r2", "S001", vec![json!(2.0)]),
                ],
            ),
        ];
        let err = fuse_feature_blocks(
            "fused",
            &blocks,
            &alignment(&blocks, AlignmentMode::Inner),
            &FeatureFusionPolicy::default(),
        )
        .unwrap_err();

        assert!(err.to_string().contains("cannot broadcast"));
    }

    #[test]
    fn fusion_refuses_duplicate_unnamespaced_feature_names() {
        let blocks = vec![
            block("nir", &["x"], &[("nir.S001", "S001", vec![json!(1.0)])]),
            block("chem", &["x"], &[("chem.S001", "S001", vec![json!(10.0)])]),
        ];
        let err = fuse_feature_blocks(
            "fused",
            &blocks,
            &alignment(&blocks, AlignmentMode::Inner),
            &FeatureFusionPolicy {
                namespace_columns: false,
            },
        )
        .unwrap_err();

        assert!(err.to_string().contains("duplicate feature"));
    }

    #[test]
    fn fusion_refuses_alignment_presence_that_does_not_match_rows() {
        let blocks = vec![
            block("nir", &["n0"], &[("nir.S001", "S001", vec![json!(1.0)])]),
            block("chem", &["c0"], &[("chem.S001", "S001", vec![json!(10.0)])]),
        ];
        let mut bad_alignment = alignment(&blocks, AlignmentMode::Inner);
        bad_alignment.masks[1].present[0] = false;
        let err = fuse_feature_blocks(
            "fused",
            &blocks,
            &bad_alignment,
            &FeatureFusionPolicy::default(),
        )
        .unwrap_err();

        assert!(err.to_string().contains("alignment presence"));
    }

    #[test]
    fn fusion_refuses_alignment_sample_absent_from_all_sources() {
        let blocks = vec![
            block("nir", &["n0"], &[("nir.S001", "S001", vec![json!(1.0)])]),
            block("chem", &["c0"], &[("chem.S001", "S001", vec![json!(10.0)])]),
        ];
        let mut bad_alignment = alignment(&blocks, AlignmentMode::Inner);
        bad_alignment.masks[0].present[0] = false;
        bad_alignment.masks[1].present[0] = false;
        let err = fuse_feature_blocks(
            "fused",
            &blocks,
            &bad_alignment,
            &FeatureFusionPolicy::default(),
        )
        .unwrap_err();

        assert!(err.to_string().contains("absent from every fused source"));
    }
}
