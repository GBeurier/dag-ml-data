use sha2::{Digest, Sha256};

use crate::error::Result;
use crate::model::DatasetSchema;

pub fn schema_fingerprint(schema: &DatasetSchema) -> Result<String> {
    let mut canonical = schema.clone();
    canonical.validate()?;
    canonical.sample_ids.sort();
    canonical
        .sources
        .sort_by(|left, right| left.id.cmp(&right.id));

    let json = serde_json::to_vec(&canonical)?;
    let digest = Sha256::digest(json);
    Ok(to_hex(&digest))
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
    use std::collections::BTreeMap;

    use crate::ids::{RepresentationId, SampleId, SourceId, TypeId};
    use crate::model::{
        AxisKind, AxisSpec, DatasetSchema, RepresentationSpec, SourceDescriptor, SourceGranularity,
    };

    use super::schema_fingerprint;

    fn representation(id: &str) -> RepresentationSpec {
        RepresentationSpec {
            id: RepresentationId::new(id).unwrap(),
            type_id: TypeId::new("table").unwrap(),
            rank: Some(2),
            axes: vec![
                AxisSpec {
                    name: "sample".to_string(),
                    kind: AxisKind::Sample,
                    unit: None,
                    size: Some(2),
                    variable: false,
                    coordinates: None,
                },
                AxisSpec {
                    name: "feature".to_string(),
                    kind: AxisKind::Feature,
                    unit: None,
                    size: Some(1),
                    variable: false,
                    coordinates: None,
                },
            ],
            container: "dataframe".to_string(),
            dtype: Some("float32".to_string()),
            sparse: false,
            ragged: false,
        }
    }

    fn source(id: &str) -> SourceDescriptor {
        SourceDescriptor {
            id: SourceId::new(id).unwrap(),
            name: id.to_string(),
            type_id: TypeId::new("table").unwrap(),
            modality: "metadata".to_string(),
            native_representation: representation("tabular"),
            sample_key: "sample_id".to_string(),
            granularity: SourceGranularity::PerSample,
            schema: BTreeMap::new(),
            tags: BTreeMap::new(),
        }
    }

    #[test]
    fn fingerprint_is_independent_of_source_order() {
        let mut left = DatasetSchema {
            dataset_id: "d".to_string(),
            sample_ids: vec![SampleId::new("s2").unwrap(), SampleId::new("s1").unwrap()],
            sources: vec![source("b"), source("a")],
            targets: BTreeMap::new(),
            metadata: BTreeMap::new(),
        };
        let mut right = left.clone();
        right.sources.reverse();
        right.sample_ids.reverse();

        assert_eq!(
            schema_fingerprint(&left).unwrap(),
            schema_fingerprint(&right).unwrap()
        );

        left.dataset_id = "different".to_string();
        assert_ne!(
            schema_fingerprint(&left).unwrap(),
            schema_fingerprint(&right).unwrap()
        );
    }
}
