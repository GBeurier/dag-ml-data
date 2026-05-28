//! Core contracts for `dag-ml-data`.
//!
//! This crate contains serializable descriptors and validation logic only. It
//! does not execute ML phases and does not own heavy host buffers.

pub mod adapter;
pub mod alignment;
pub mod buffer;
pub mod buffer_file_store;
pub mod collation;
pub mod coordinator;
pub mod error;
pub mod fingerprint;
pub mod fitted_adapter;
pub mod fusion;
pub mod handle;
pub mod ids;
pub mod model;
pub mod plan;
pub mod planner;
pub mod relation;

pub use adapter::*;
pub use alignment::*;
pub use buffer::*;
pub use buffer_file_store::*;
pub use collation::*;
pub use coordinator::*;
pub use error::{DataError, Result};
pub use fingerprint::*;
pub use fitted_adapter::*;
pub use fusion::*;
pub use handle::*;
pub use ids::*;
pub use model::*;
pub use plan::*;
pub use planner::*;
pub use relation::*;
