use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::error::{DataError, Result};

pub const FITTED_ADAPTER_REF_SCHEMA_VERSION: u32 = 1;
pub const FITTED_ADAPTER_MANIFEST_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FittedAdapterBackend {
    Joblib,
    Pickle,
    Json,
    Numpy,
    Onnx,
    Raw,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FittedAdapterRef {
    #[serde(default = "default_fitted_adapter_ref_schema_version")]
    pub schema_version: u32,
    pub adapter_id: String,
    pub adapter_version: String,
    pub params_fingerprint: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend: Option<FittedAdapterBackend>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_fingerprint: Option<String>,
    // Matches `dag-ml` `ArtifactRef.size_bytes` wire shape: when `None` this
    // field serializes as `"size_bytes": null`, not omitted, so the two refs
    // round-trip identically over JSON.
    #[serde(default)]
    pub size_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plugin: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plugin_version: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub metadata: BTreeMap<String, serde_json::Value>,
}

fn default_fitted_adapter_ref_schema_version() -> u32 {
    FITTED_ADAPTER_REF_SCHEMA_VERSION
}

fn default_fitted_adapter_manifest_schema_version() -> u32 {
    FITTED_ADAPTER_MANIFEST_SCHEMA_VERSION
}

impl FittedAdapterRef {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != FITTED_ADAPTER_REF_SCHEMA_VERSION {
            return Err(DataError::Validation(format!(
                "fitted adapter `{}` uses unsupported schema_version {}, expected {}",
                self.adapter_id, self.schema_version, FITTED_ADAPTER_REF_SCHEMA_VERSION
            )));
        }
        if self.adapter_id.trim().is_empty() {
            return Err(DataError::Validation(
                "fitted adapter ref has empty adapter_id".to_string(),
            ));
        }
        if self.adapter_version.trim().is_empty() {
            return Err(DataError::Validation(format!(
                "fitted adapter `{}` has empty adapter_version",
                self.adapter_id
            )));
        }
        validate_fingerprint(
            "fitted adapter params",
            &self.params_fingerprint,
            &self.adapter_id,
        )?;
        validate_optional_text("uri", &self.uri, &self.adapter_id)?;
        validate_optional_text("plugin", &self.plugin, &self.adapter_id)?;
        validate_optional_text("plugin_version", &self.plugin_version, &self.adapter_id)?;
        if self.plugin_version.is_some() && self.plugin.is_none() {
            return Err(DataError::Validation(format!(
                "fitted adapter `{}` has plugin_version without plugin",
                self.adapter_id
            )));
        }
        if let Some(content_fingerprint) = &self.content_fingerprint {
            validate_fingerprint(
                "fitted adapter content",
                content_fingerprint,
                &self.adapter_id,
            )?;
        }
        if self.uri.is_some() && self.backend.is_none() {
            return Err(DataError::Validation(format!(
                "fitted adapter `{}` has uri without backend",
                self.adapter_id
            )));
        }
        if self.uri.is_some() && self.content_fingerprint.is_none() {
            return Err(DataError::Validation(format!(
                "fitted adapter `{}` has uri without content_fingerprint",
                self.adapter_id
            )));
        }
        for key in self.metadata.keys() {
            if key.trim().is_empty() {
                return Err(DataError::Validation(format!(
                    "fitted adapter `{}` metadata contains an empty key",
                    self.adapter_id
                )));
            }
        }
        Ok(())
    }

    /// Validate that the fitted adapter carries portable persistence metadata:
    /// a backend, a safe relative URI and a content fingerprint. Legacy refs
    /// that only carry inline state stay readable through [`Self::validate`]
    /// but are refused here so persisted manifests can be moved with their
    /// payloads.
    pub fn validate_portable(&self) -> Result<()> {
        self.validate()?;
        let Some(uri) = self.uri.as_deref() else {
            return Err(DataError::Validation(format!(
                "fitted adapter `{}` is not portable: requires backend, uri and content_fingerprint",
                self.adapter_id
            )));
        };
        validate_relative_uri(&self.adapter_id, uri)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FittedAdapterManifestEntry {
    pub adapter_id: String,
    pub fitted_adapter: FittedAdapterRef,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FittedAdapterManifest {
    #[serde(default = "default_fitted_adapter_manifest_schema_version")]
    pub schema_version: u32,
    pub entries: Vec<FittedAdapterManifestEntry>,
}

impl FittedAdapterManifest {
    pub fn new(entries: Vec<FittedAdapterManifestEntry>) -> Self {
        Self {
            schema_version: FITTED_ADAPTER_MANIFEST_SCHEMA_VERSION,
            entries,
        }
    }

    pub fn validate(&self) -> Result<()> {
        self.validate_inner(false)
    }

    /// Validate the manifest and require every entry to carry portable
    /// persistence metadata. Use this when serializing manifests for replay
    /// across machines or repositories.
    pub fn validate_portable(&self) -> Result<()> {
        self.validate_inner(true)
    }

    fn validate_inner(&self, require_portable: bool) -> Result<()> {
        if self.schema_version != FITTED_ADAPTER_MANIFEST_SCHEMA_VERSION {
            return Err(DataError::Validation(format!(
                "fitted adapter manifest uses unsupported schema_version {}, expected {}",
                self.schema_version, FITTED_ADAPTER_MANIFEST_SCHEMA_VERSION
            )));
        }
        let mut seen = BTreeSet::new();
        for entry in &self.entries {
            if entry.adapter_id != entry.fitted_adapter.adapter_id {
                return Err(DataError::Validation(format!(
                    "fitted adapter manifest entry key `{}` does not match ref adapter_id `{}`",
                    entry.adapter_id, entry.fitted_adapter.adapter_id
                )));
            }
            if !seen.insert(&entry.adapter_id) {
                return Err(DataError::Validation(format!(
                    "fitted adapter manifest contains duplicate adapter_id `{}`",
                    entry.adapter_id
                )));
            }
            if require_portable {
                entry.fitted_adapter.validate_portable()?;
            } else {
                entry.fitted_adapter.validate()?;
            }
        }
        Ok(())
    }
}

fn validate_optional_text(label: &str, value: &Option<String>, adapter_id: &str) -> Result<()> {
    if let Some(text) = value {
        if text.trim().is_empty() {
            return Err(DataError::Validation(format!(
                "fitted adapter `{adapter_id}` has empty {label}"
            )));
        }
        if text.chars().any(char::is_control) {
            return Err(DataError::Validation(format!(
                "fitted adapter `{adapter_id}` has control characters in {label}"
            )));
        }
    }
    Ok(())
}

fn validate_fingerprint(label: &str, value: &str, adapter_id: &str) -> Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(DataError::Validation(format!(
            "fitted adapter `{adapter_id}` {label} fingerprint must be a 64-character hex digest"
        )));
    }
    Ok(())
}

/// Deterministic path safety for relative fitted-adapter URIs. Rejects empty
/// values, control characters, absolute paths (POSIX root, Windows root or
/// drive prefix), URI schemes such as `http://`, `s3://` or `file://` (any
/// colon in the leading path segment) and any `..` traversal component.
/// Parsing is platform-independent so portable manifests validate identically
/// everywhere; it adds no dependency. Kept byte-for-byte equivalent with
/// `dag-ml`'s `validate_relative_artifact_uri` in
/// `crates/dag-ml-core/src/runtime.rs` so portable refs accepted by one repo
/// are accepted by the other.
fn validate_relative_uri(adapter_id: &str, uri: &str) -> Result<()> {
    if uri.is_empty() {
        return Err(DataError::Validation(format!(
            "fitted adapter `{adapter_id}` has empty uri"
        )));
    }
    if uri.chars().any(char::is_control) {
        return Err(DataError::Validation(format!(
            "fitted adapter `{adapter_id}` uri has control characters"
        )));
    }
    if uri.starts_with('/') || uri.starts_with('\\') {
        return Err(DataError::Validation(format!(
            "fitted adapter `{adapter_id}` uri `{uri}` must be a relative path"
        )));
    }
    let mut prefix = uri.chars();
    if let (Some(drive), Some(':')) = (prefix.next(), prefix.next()) {
        if drive.is_ascii_alphabetic() {
            return Err(DataError::Validation(format!(
                "fitted adapter `{adapter_id}` uri `{uri}` must be a relative path"
            )));
        }
    }
    let first_segment = uri.split(['/', '\\']).next().unwrap_or(uri);
    if first_segment.contains(':') {
        return Err(DataError::Validation(format!(
            "fitted adapter `{adapter_id}` uri `{uri}` must not include a scheme or colon in its first path segment"
        )));
    }
    for segment in uri.split(['/', '\\']) {
        if segment == ".." {
            return Err(DataError::Validation(format!(
                "fitted adapter `{adapter_id}` uri `{uri}` must not contain `..` components"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fingerprint(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn portable_ref() -> FittedAdapterRef {
        FittedAdapterRef {
            schema_version: FITTED_ADAPTER_REF_SCHEMA_VERSION,
            adapter_id: "snv".to_string(),
            adapter_version: "1.0.0".to_string(),
            params_fingerprint: fingerprint(0x12),
            backend: Some(FittedAdapterBackend::Joblib),
            uri: Some("fitted/snv.joblib".to_string()),
            content_fingerprint: Some(fingerprint(0x34)),
            size_bytes: Some(2048),
            plugin: Some("sklearn".to_string()),
            plugin_version: Some("1.4.0".to_string()),
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn portable_ref_validates() {
        let value = portable_ref();
        value.validate().unwrap();
        value.validate_portable().unwrap();
    }

    #[test]
    fn inline_ref_validates_but_is_not_portable() {
        let value = FittedAdapterRef {
            uri: None,
            backend: None,
            content_fingerprint: None,
            size_bytes: None,
            plugin: None,
            plugin_version: None,
            ..portable_ref()
        };
        value.validate().unwrap();
        let error = value.validate_portable().unwrap_err();
        assert!(format!("{error}").contains("is not portable"));
    }

    #[test]
    fn rejects_unsupported_schema_version() {
        let mut value = portable_ref();
        value.schema_version = FITTED_ADAPTER_REF_SCHEMA_VERSION + 1;
        let error = value.validate().unwrap_err();
        assert!(format!("{error}").contains("unsupported schema_version"));
    }

    #[test]
    fn rejects_uri_without_backend_or_fingerprint() {
        let mut value = portable_ref();
        value.backend = None;
        let error = value.validate().unwrap_err();
        assert!(format!("{error}").contains("has uri without backend"));

        let mut value = portable_ref();
        value.content_fingerprint = None;
        let error = value.validate().unwrap_err();
        assert!(format!("{error}").contains("has uri without content_fingerprint"));
    }

    #[test]
    fn rejects_plugin_version_without_plugin() {
        let mut value = portable_ref();
        value.plugin = None;
        let error = value.validate().unwrap_err();
        assert!(format!("{error}").contains("has plugin_version without plugin"));
    }

    #[test]
    fn rejects_non_portable_uris() {
        for uri in [
            "/abs/path.joblib",
            "C:\\fitted\\snv.joblib",
            "http://example.com/fitted/snv.joblib",
            "s3://bucket/fitted/snv.joblib",
            "file:///fitted/snv.joblib",
            "../escape/snv.joblib",
            "fitted/../escape/snv.joblib",
        ] {
            let value = FittedAdapterRef {
                uri: Some(uri.to_string()),
                ..portable_ref()
            };
            let error = value.validate_portable().unwrap_err();
            assert!(
                format!("{error}").contains("must") || format!("{error}").contains("relative path"),
                "URI `{uri}` should be rejected, got: {error}"
            );
        }
    }

    #[test]
    fn rejects_uri_with_control_chars_or_empty() {
        let mut value = portable_ref();
        value.uri = Some("".to_string());
        let error = value.validate_portable().unwrap_err();
        assert!(format!("{error}").contains("empty uri"));

        let mut value = portable_ref();
        value.uri = Some("fitted/\u{0001}".to_string());
        let error = value.validate_portable().unwrap_err();
        assert!(format!("{error}").contains("control characters"));
    }

    #[test]
    fn rejects_whitespace_only_optional_text_fields() {
        for (label, value) in [
            (
                "uri",
                FittedAdapterRef {
                    uri: Some("   ".to_string()),
                    ..portable_ref()
                },
            ),
            (
                "plugin",
                FittedAdapterRef {
                    plugin: Some("\t".to_string()),
                    ..portable_ref()
                },
            ),
            (
                "plugin_version",
                FittedAdapterRef {
                    plugin_version: Some("   ".to_string()),
                    ..portable_ref()
                },
            ),
        ] {
            let error = value.validate().unwrap_err();
            let message = format!("{error}");
            assert!(
                message.contains(&format!("has empty {label}")),
                "expected empty {label} error, got: {message}"
            );
        }
    }

    #[test]
    fn rejects_metadata_with_empty_keys() {
        let mut value = portable_ref();
        value
            .metadata
            .insert("".to_string(), serde_json::Value::Null);
        let error = value.validate().unwrap_err();
        assert!(format!("{error}").contains("metadata contains an empty key"));
    }

    #[test]
    fn rejects_invalid_fingerprints() {
        let mut value = portable_ref();
        value.params_fingerprint = "not-a-fingerprint".to_string();
        let error = value.validate().unwrap_err();
        assert!(format!("{error}").contains("params fingerprint"));

        let mut value = portable_ref();
        value.content_fingerprint = Some("not-a-fingerprint".to_string());
        let error = value.validate().unwrap_err();
        assert!(format!("{error}").contains("content fingerprint"));
    }

    #[test]
    fn manifest_validates_and_requires_unique_adapter_ids() {
        let entry = FittedAdapterManifestEntry {
            adapter_id: "snv".to_string(),
            fitted_adapter: portable_ref(),
        };
        let manifest = FittedAdapterManifest::new(vec![entry.clone()]);
        manifest.validate().unwrap();
        manifest.validate_portable().unwrap();

        let manifest = FittedAdapterManifest::new(vec![entry.clone(), entry]);
        let error = manifest.validate().unwrap_err();
        assert!(format!("{error}").contains("duplicate adapter_id"));
    }

    #[test]
    fn manifest_refuses_key_mismatch_and_bad_schema_version() {
        let entry = FittedAdapterManifestEntry {
            adapter_id: "snv".to_string(),
            fitted_adapter: FittedAdapterRef {
                adapter_id: "msc".to_string(),
                ..portable_ref()
            },
        };
        let manifest = FittedAdapterManifest::new(vec![entry]);
        let error = manifest.validate().unwrap_err();
        assert!(format!("{error}").contains("does not match ref adapter_id"));

        let mut manifest = FittedAdapterManifest::new(vec![]);
        manifest.schema_version = FITTED_ADAPTER_MANIFEST_SCHEMA_VERSION + 1;
        let error = manifest.validate().unwrap_err();
        assert!(format!("{error}").contains("unsupported schema_version"));
    }

    #[test]
    fn manifest_portable_check_rejects_non_portable_entries() {
        let entry = FittedAdapterManifestEntry {
            adapter_id: "snv".to_string(),
            fitted_adapter: FittedAdapterRef {
                uri: None,
                backend: None,
                content_fingerprint: None,
                size_bytes: None,
                plugin: None,
                plugin_version: None,
                ..portable_ref()
            },
        };
        let manifest = FittedAdapterManifest::new(vec![entry]);
        manifest.validate().unwrap();
        let error = manifest.validate_portable().unwrap_err();
        assert!(format!("{error}").contains("is not portable"));
    }
}
