use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;

/// A stable ADR-11 error payload that can be serialized across bindings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataErrorDescriptor {
    /// ADR-11 category, for example `validation`, `data` or `compatibility`.
    pub category: String,
    /// Stable machine-readable code inside the category.
    pub code: String,
    /// Error severity. Current failing variants use `error`.
    pub severity: String,
    /// Human-readable error message.
    pub message: String,
    /// One-sentence remediation hint suitable for user-facing diagnostics.
    pub remediation_hint: String,
    /// Structured debug fields that remain stable enough for logs and tests.
    pub context: BTreeMap<String, Value>,
}

#[derive(Debug, Error)]
pub enum DataError {
    #[error("invalid identifier `{value}`: {reason}")]
    InvalidIdentifier { value: String, reason: &'static str },

    #[error("data contract validation failed: {0}")]
    Validation(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

impl DataError {
    /// Return the stable ADR-11 category for this error.
    pub fn category(&self) -> &'static str {
        self.taxonomy_parts().0
    }

    /// Return the stable ADR-11 code for this error.
    pub fn code(&self) -> &'static str {
        self.taxonomy_parts().1
    }

    /// Return the ADR-11 severity for this error.
    pub fn severity(&self) -> &'static str {
        self.taxonomy_parts().2
    }

    /// Return the remediation hint associated with this error.
    pub fn remediation_hint(&self) -> &'static str {
        match self {
            Self::InvalidIdentifier { .. } => {
                "Use a non-empty stable identifier that matches the dag-ml-data identifier grammar."
            }
            Self::Validation(_) => {
                "Fix the data contract inputs, schema, handles or coordinator request before retrying."
            }
            Self::Serialization(_) => {
                "Check that the JSON payload matches the supported dag-ml-data contract version."
            }
        }
    }

    /// Return structured context fields for logs, bindings and tests.
    pub fn context(&self) -> BTreeMap<String, Value> {
        let mut context = BTreeMap::new();
        match self {
            Self::InvalidIdentifier { value, reason } => {
                context.insert("value".to_string(), json!(value));
                context.insert("reason".to_string(), json!(reason));
            }
            Self::Validation(detail) => {
                context.insert("detail".to_string(), json!(detail));
            }
            Self::Serialization(error) => {
                context.insert("detail".to_string(), json!(error.to_string()));
            }
        }
        context
    }

    /// Build the serializable ADR-11 descriptor for this error.
    pub fn descriptor(&self) -> DataErrorDescriptor {
        DataErrorDescriptor {
            category: self.category().to_string(),
            code: self.code().to_string(),
            severity: self.severity().to_string(),
            message: self.to_string(),
            remediation_hint: self.remediation_hint().to_string(),
            context: self.context(),
        }
    }

    /// Serialize the ADR-11 descriptor as compact JSON.
    pub fn descriptor_json(&self) -> std::result::Result<String, serde_json::Error> {
        serde_json::to_string(&self.descriptor())
    }

    /// Stable ADR-11 numeric error code for FFI consumers: the high 16 bits are
    /// the taxonomy category id and the low 16 bits are the per-category code id,
    /// mirroring the `(category << 16) | code` convention from ADR-11. The
    /// category ids are shared with `dag-ml` so a given category maps to the same
    /// number in both libraries.
    pub fn error_code(&self) -> u32 {
        let (category_id, code_id) = self.numeric_taxonomy();
        (u32::from(category_id) << 16) | u32::from(code_id)
    }

    fn taxonomy_parts(&self) -> (&'static str, &'static str, &'static str) {
        match self {
            Self::InvalidIdentifier { .. } => ("validation", "invalid_identifier", "error"),
            Self::Validation(_) => ("data", "data_contract_validation", "error"),
            Self::Serialization(_) => ("compatibility", "serialization_error", "error"),
        }
    }

    /// Stable `(category_id, code_id)` pair backing [`error_code`](Self::error_code).
    ///
    /// Category ids follow ADR-11 and match `dag-ml`: validation=0, runtime=1,
    /// data=2, controller=3, bundle=4, lineage=5, replay=6, security=7,
    /// compatibility=8, internal=9. Code ids are **1-based** so a packed
    /// `error_code()` is never `0` — `0` is the "no error" sentinel returned by
    /// `dagmldata_last_error_code()`. Code ids are stable within their category;
    /// never renumber a shipped pair.
    fn numeric_taxonomy(&self) -> (u16, u16) {
        match self {
            Self::InvalidIdentifier { .. } => (0, 1),
            Self::Validation(_) => (2, 1),
            Self::Serialization(_) => (8, 1),
        }
    }
}

pub type Result<T> = std::result::Result<T, DataError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_identifier_descriptor_carries_taxonomy_and_context() {
        let error = DataError::InvalidIdentifier {
            value: "".to_string(),
            reason: "empty",
        };

        let descriptor = error.descriptor();

        assert_eq!(descriptor.category, "validation");
        assert_eq!(descriptor.code, "invalid_identifier");
        assert_eq!(descriptor.severity, "error");
        assert_eq!(descriptor.context["value"], json!(""));
        assert_eq!(descriptor.context["reason"], json!("empty"));
        assert!(descriptor.remediation_hint.contains("identifier"));
    }

    #[test]
    fn error_code_packs_category_and_code() {
        // Code ids are 1-based so no real error packs to the 0 "no error" sentinel.
        assert_eq!(
            DataError::InvalidIdentifier {
                value: "x".to_string(),
                reason: "bad",
            }
            .error_code(),
            0x0000_0001
        );
        // data (2) / data_contract_validation (1) -> 0x0002_0001
        assert_eq!(
            DataError::Validation("x".to_string()).error_code(),
            0x0002_0001
        );
        // compatibility (8) / serialization_error (1) -> 0x0008_0001
        let serde_error = serde_json::from_str::<Value>("{").expect_err("invalid JSON");
        assert_eq!(
            DataError::Serialization(serde_error).error_code(),
            0x0008_0001
        );
    }

    #[test]
    fn validation_descriptor_uses_data_category() {
        let error = DataError::Validation("shape contract mismatch".to_string());

        let descriptor = error.descriptor();

        assert_eq!(descriptor.category, "data");
        assert_eq!(descriptor.code, "data_contract_validation");
        assert_eq!(
            descriptor.context["detail"],
            json!("shape contract mismatch")
        );
        assert!(descriptor
            .message
            .contains("data contract validation failed"));
    }

    #[test]
    fn descriptor_json_is_stable_json_payload() {
        let serde_error = serde_json::from_str::<Value>("{").expect_err("invalid JSON");
        let error = DataError::Serialization(serde_error);

        let payload = error.descriptor_json().expect("descriptor JSON");
        let parsed = serde_json::from_str::<Value>(&payload).expect("parse descriptor");

        assert_eq!(parsed["category"], json!("compatibility"));
        assert_eq!(parsed["code"], json!("serialization_error"));
        assert!(parsed["message"]
            .as_str()
            .expect("message")
            .contains("serialization error"));
        assert!(parsed["remediation_hint"]
            .as_str()
            .expect("hint")
            .contains("contract version"));
    }
}
