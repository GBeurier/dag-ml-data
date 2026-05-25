//! Core contracts for `dag-ml-data`.
//!
//! This crate contains serializable descriptors and validation logic only. It
//! does not execute ML phases and does not own heavy host buffers.

pub mod adapter;
pub mod error;
pub mod fingerprint;
pub mod ids;
pub mod model;
pub mod plan;
pub mod planner;
pub mod relation;

pub use adapter::*;
pub use error::{DataError, Result};
pub use fingerprint::*;
pub use ids::*;
pub use model::*;
pub use plan::*;
pub use planner::*;
pub use relation::*;
