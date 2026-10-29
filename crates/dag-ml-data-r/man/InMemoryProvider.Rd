\name{InMemoryProvider}
\alias{InMemoryProvider}
\alias{$.InMemoryProvider}
\alias{[[.InMemoryProvider}
\title{In-Memory Data Provider}
\description{
An in-memory \code{dag-ml-data} provider with JSON input and output and opaque
decimal-string handles. The Rust provider owns the copied data and validates
the shared data contracts. It does not choose folds or fit models.
}
\usage{
InMemoryProvider
}
\format{
An environment containing the constructor and provider methods. Construct an
instance with \code{InMemoryProvider$new(envelope_json, target_tables_json,
feature_tables_json, f64_feature_matrices_json)}. All four arguments are JSON
strings; use \code{""} for an absent optional table. At most one of
\code{feature_tables_json} and \code{f64_feature_matrices_json} may contain a
non-empty array.
}
\details{
Call methods on the constructed instance with \code{provider$method(...)}:
\describe{
  \item{\code{materialize(request_json)}}{Materializes data and returns a data handle.}
  \item{\code{make_view(data_handle, view_json)}}{Creates a view and returns its handle.}
  \item{\code{view_identity(view_handle)}}{Returns view identity rows as JSON.}
  \item{\code{target_block(view_handle, target_id)}}{Returns a target block as JSON.}
  \item{\code{feature_block(view_handle, feature_set_id)}}{Returns a feature block as JSON.}
  \item{\code{feature_collation(view_handle, selector_json)}}{Returns a row-major numeric tensor block as JSON.}
  \item{\code{feature_buffer_manifests()}}{Returns provider-wide feature-buffer manifests as JSON.}
  \item{\code{data_feature_buffer_bindings(data_handle)}}{Returns feature-buffer bindings scoped to a data handle as JSON.}
  \item{\code{release(handle)}}{Releases a data or view handle and returns whether it was released.}
}
Handles must be passed back as strings, without conversion to R numeric values.
Release data and view handles when they are no longer needed. Invalid inputs
raise R errors containing the Rust provider's structured error descriptor.
}
\keyword{data}
