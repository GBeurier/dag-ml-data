//! Frozen, published catalogue of the built-in representation IDs (`B-014` /
//! `DMD-001`).
//!
//! `B-014` is a freeze/publish problem, not an invention problem: the stable
//! representation-ID vocabulary already lives in [`crate::builtin_models`] as
//! `pub const` strings, a [`BuiltinDataModel`] enum and its constructors. This
//! module does not add a new vocabulary; it *publishes* the existing one as a
//! single, deterministic, machine-readable registry so cross-repo consumers
//! (notably `ControllerManifest.data_requirements` ports on `L16` and the
//! `nirs4all-io` emit on `L7`) can reference the frozen strings, and so a drift
//! test can freeze them against [`crate::builtin_models`].
//!
//! The companion artifact `docs/contracts/representation_registry.v1.json` is
//! generated from [`representation_registry`] (via the
//! `dag-ml-data-cli representation-registry` command) and pinned by the
//! `published_registry_matches_builtin_models` drift test, which fails the
//! moment a built-in representation changes without the published manifest
//! being regenerated.

use serde::{Deserialize, Serialize};

use crate::builtin_models::{
    BuiltinDataModel, BUILTIN_DATA_MODELS, REPRESENTATION_FEATURE_BLOCK_SET,
    REPRESENTATION_GRAY_IMAGE, REPRESENTATION_MC_IMAGE, REPRESENTATION_MULTISPECTRAL_IMAGE,
    REPRESENTATION_RGB_IMAGE, REPRESENTATION_SAMPLE_METADATA, REPRESENTATION_SIGNAL_1D,
    REPRESENTATION_SIGNAL_WITH_PROCESSINGS, REPRESENTATION_TARGET_CATEGORICAL,
    REPRESENTATION_TARGET_CATEGORICAL_MATRIX, REPRESENTATION_TARGET_NUMERIC,
    REPRESENTATION_TARGET_NUMERIC_MATRIX,
};
use crate::model::RepresentationSpec;

/// Wire-shape version of the published representation registry manifest.
pub const REPRESENTATION_REGISTRY_SCHEMA_VERSION: u32 = 1;

/// Stable identifier of the representation registry contract.
pub const REPRESENTATION_REGISTRY_ID: &str = "dag-ml-data.representation_registry.v1";

/// Repo-relative provenance of the frozen representation IDs.
pub const REPRESENTATION_REGISTRY_SOURCE: &str = "crates/dag-ml-data-core/src/builtin_models.rs";

/// Name of the spectra+image MVP profile defined by `B-014` / `IO_spec.md` §5.
pub const MVP_PROFILE_SPECTRA_IMAGE: &str = "spectra_image";

/// The 8 spectra/target/metadata representation IDs that the `nirs4all-io`
/// bridge already emits for the spectra+image MVP (`IO_spec.md` §5).
const MVP_SPECTRA_IMAGE_EMITTED: &[&str] = &[
    REPRESENTATION_SIGNAL_1D,
    REPRESENTATION_SIGNAL_WITH_PROCESSINGS,
    REPRESENTATION_FEATURE_BLOCK_SET,
    REPRESENTATION_TARGET_NUMERIC,
    REPRESENTATION_TARGET_CATEGORICAL,
    REPRESENTATION_TARGET_NUMERIC_MATRIX,
    REPRESENTATION_TARGET_CATEGORICAL_MATRIX,
    REPRESENTATION_SAMPLE_METADATA,
];

/// The 4 image representation IDs that are landed in this crate but not yet
/// emitted by `nirs4all-io` (`IO-010`).
const MVP_SPECTRA_IMAGE_PENDING: &[&str] = &[
    REPRESENTATION_GRAY_IMAGE,
    REPRESENTATION_RGB_IMAGE,
    REPRESENTATION_MC_IMAGE,
    REPRESENTATION_MULTISPECTRAL_IMAGE,
];

/// Cross-repo publication status of a representation ID inside an MVP profile.
///
/// This annotates *downstream emission readiness only*; every representation in
/// the registry is fully landed in this crate (constructor, axes and dtype all
/// present). The status is sourced from `IO_spec.md` §5, not verified at runtime
/// by `dag-ml-data`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MvpEmissionStatus {
    /// Landed here and already emitted by the `nirs4all-io` → `dag-ml-data`
    /// bridge (the 8 spectra/target/metadata IDs of `IO_spec.md` §5).
    Emitted,
    /// Landed here (representation, axes, dtype and constructor present) but not
    /// yet emitted by `nirs4all-io`; emission is tracked by `IO-010`.
    LandedPendingEmit,
}

/// MVP profile membership for a published representation ID.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct MvpStatus {
    /// Named MVP profile this ID belongs to (currently only
    /// [`MVP_PROFILE_SPECTRA_IMAGE`]).
    pub profile: String,
    /// Cross-repo emission status of the ID within that profile.
    pub emission: MvpEmissionStatus,
}

/// One frozen, published entry of the built-in representation registry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegisteredRepresentation {
    /// Stable representation-ID string (e.g. `signal_1d`) — the value that
    /// `ControllerManifest.data_requirements` ports reference by string.
    pub representation_id: String,
    /// [`BuiltinDataModel`] catalogue key (e.g. `nirs.signal_1d`).
    pub builtin_key: String,
    /// Generic modality label (e.g. `nirs`, `image`, `target`).
    pub modality: String,
    /// Optional spectra+image MVP annotation; `None` for post-MVP IDs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mvp: Option<MvpStatus>,
    /// The complete frozen representation spec (axes, rank, dtype, container).
    pub representation: RepresentationSpec,
}

/// The frozen, published catalogue of built-in representation IDs.
///
/// Build it with [`representation_registry`]. The serialized form is the
/// `docs/contracts/representation_registry.v1.json` contract artifact.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepresentationRegistry {
    /// Wire-shape version of this manifest
    /// ([`REPRESENTATION_REGISTRY_SCHEMA_VERSION`]).
    pub schema_version: u32,
    /// Stable identifier of this registry contract
    /// ([`REPRESENTATION_REGISTRY_ID`]).
    pub registry_id: String,
    /// Repo-relative provenance of the frozen IDs
    /// ([`REPRESENTATION_REGISTRY_SOURCE`]).
    pub source: String,
    /// Every built-in representation, in [`BUILTIN_DATA_MODELS`] order.
    pub representations: Vec<RegisteredRepresentation>,
}

/// Build the frozen, published catalogue of built-in representation IDs from
/// [`BUILTIN_DATA_MODELS`]. The result is deterministic and source-order stable.
pub fn representation_registry() -> RepresentationRegistry {
    RepresentationRegistry {
        schema_version: REPRESENTATION_REGISTRY_SCHEMA_VERSION,
        registry_id: REPRESENTATION_REGISTRY_ID.to_string(),
        source: REPRESENTATION_REGISTRY_SOURCE.to_string(),
        representations: BUILTIN_DATA_MODELS
            .iter()
            .copied()
            .map(registered_representation)
            .collect(),
    }
}

fn registered_representation(model: BuiltinDataModel) -> RegisteredRepresentation {
    let representation = model.representation();
    let representation_id = representation.id.as_str().to_string();
    let mvp = mvp_status(&representation_id);
    RegisteredRepresentation {
        representation_id,
        builtin_key: model.key().to_string(),
        modality: model.modality().to_string(),
        mvp,
        representation,
    }
}

fn mvp_status(representation_id: &str) -> Option<MvpStatus> {
    if MVP_SPECTRA_IMAGE_EMITTED.contains(&representation_id) {
        Some(MvpStatus {
            profile: MVP_PROFILE_SPECTRA_IMAGE.to_string(),
            emission: MvpEmissionStatus::Emitted,
        })
    } else if MVP_SPECTRA_IMAGE_PENDING.contains(&representation_id) {
        Some(MvpStatus {
            profile: MVP_PROFILE_SPECTRA_IMAGE.to_string(),
            emission: MvpEmissionStatus::LandedPendingEmit,
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    const PUBLISHED: &str = include_str!("../../../docs/contracts/representation_registry.v1.json");

    #[test]
    fn published_registry_matches_builtin_models() {
        let generated = serde_json::to_value(representation_registry()).unwrap();
        let published: serde_json::Value = serde_json::from_str(PUBLISHED).unwrap();
        assert_eq!(
            published, generated,
            "docs/contracts/representation_registry.v1.json drifted from builtin_models.rs; \
             regenerate with `cargo run -p dag-ml-data-cli -- representation-registry`"
        );
    }

    #[test]
    fn registry_header_is_stable() {
        let registry = representation_registry();
        assert_eq!(
            registry.schema_version,
            REPRESENTATION_REGISTRY_SCHEMA_VERSION
        );
        assert_eq!(registry.registry_id, REPRESENTATION_REGISTRY_ID);
        assert_eq!(registry.source, REPRESENTATION_REGISTRY_SOURCE);
    }

    #[test]
    fn registry_covers_every_builtin_with_unique_ids() {
        let registry = representation_registry();
        assert_eq!(registry.representations.len(), BUILTIN_DATA_MODELS.len());

        let mut ids = BTreeSet::new();
        let mut keys = BTreeSet::new();
        for entry in &registry.representations {
            assert!(
                ids.insert(entry.representation_id.clone()),
                "duplicate representation id {}",
                entry.representation_id
            );
            assert!(
                keys.insert(entry.builtin_key.clone()),
                "duplicate builtin key {}",
                entry.builtin_key
            );
            // The denormalized id must equal the embedded representation spec id.
            assert_eq!(entry.representation_id, entry.representation.id.as_str());
            // The embedded spec must validate (it is the frozen contract object).
            entry.representation.validate().unwrap();
        }
    }

    #[test]
    fn spectra_image_mvp_profile_is_consistent() {
        let registry = representation_registry();
        let frozen: BTreeSet<&str> = registry
            .representations
            .iter()
            .map(|entry| entry.representation_id.as_str())
            .collect();

        let emitted: BTreeSet<&str> = registry
            .representations
            .iter()
            .filter(|entry| {
                matches!(&entry.mvp, Some(status) if status.emission == MvpEmissionStatus::Emitted)
            })
            .map(|entry| entry.representation_id.as_str())
            .collect();
        let pending: BTreeSet<&str> = registry
            .representations
            .iter()
            .filter(|entry| {
                matches!(
                    &entry.mvp,
                    Some(status) if status.emission == MvpEmissionStatus::LandedPendingEmit
                )
            })
            .map(|entry| entry.representation_id.as_str())
            .collect();

        // B-014 spectra+image MVP = 12 IDs (8 emitted + 4 image landed-pending-emit).
        assert_eq!(emitted.len(), 8, "spectra MVP must publish 8 emitted IDs");
        assert_eq!(pending.len(), 4, "image MVP must publish 4 pending IDs");
        assert!(emitted.is_disjoint(&pending), "MVP groups must be disjoint");
        for id in emitted.iter().chain(pending.iter()) {
            assert!(
                frozen.contains(id),
                "MVP id {id} is not a frozen representation"
            );
        }

        // The four pending IDs are exactly the image set.
        assert_eq!(
            pending,
            BTreeSet::from([
                REPRESENTATION_GRAY_IMAGE,
                REPRESENTATION_RGB_IMAGE,
                REPRESENTATION_MC_IMAGE,
                REPRESENTATION_MULTISPECTRAL_IMAGE,
            ])
        );

        // Every MVP annotation uses the published profile name.
        for entry in &registry.representations {
            if let Some(status) = &entry.mvp {
                assert_eq!(status.profile, MVP_PROFILE_SPECTRA_IMAGE);
            }
        }
    }

    #[test]
    fn registry_round_trips_through_json() {
        let registry = representation_registry();
        let json = serde_json::to_string(&registry).unwrap();
        let decoded: RepresentationRegistry = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, registry);
    }
}
