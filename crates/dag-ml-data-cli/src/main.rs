use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use dag_ml_data_core::{
    plan_model_input, schema_fingerprint, AdapterRegistry, AdapterRegistrySpec, DataPlanRequest,
    DatasetSchema, ModelInputSpec, SourceId,
};

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    FingerprintSchema {
        path: PathBuf,
    },
    PlanModelInput {
        #[arg(long)]
        schema: PathBuf,
        #[arg(long)]
        model_input: PathBuf,
        #[arg(long)]
        adapters: PathBuf,
        #[arg(long)]
        id: String,
        #[arg(long = "source")]
        sources: Vec<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::FingerprintSchema { path } => {
            let data = std::fs::read(&path)
                .with_context(|| format!("failed to read schema JSON at {}", path.display()))?;
            let schema: DatasetSchema = serde_json::from_slice(&data)
                .with_context(|| format!("failed to parse schema JSON at {}", path.display()))?;
            let fingerprint = schema_fingerprint(&schema)
                .with_context(|| format!("invalid schema at {}", path.display()))?;
            println!("{fingerprint}");
        }
        Command::PlanModelInput {
            schema,
            model_input,
            adapters,
            id,
            sources,
        } => {
            let schema: DatasetSchema = read_json(&schema, "schema")?;
            let model_input: ModelInputSpec = read_json(&model_input, "model input")?;
            let registry_spec: AdapterRegistrySpec = read_json(&adapters, "adapter registry")?;
            let registry = AdapterRegistry::from_spec(registry_spec)
                .with_context(|| format!("invalid adapter registry at {}", adapters.display()))?;
            let plan = plan_model_input(
                &schema,
                &model_input,
                &registry,
                &DataPlanRequest {
                    id,
                    source_ids: (!sources.is_empty())
                        .then(|| {
                            sources
                                .into_iter()
                                .map(SourceId::new)
                                .collect::<dag_ml_data_core::Result<Vec<_>>>()
                        })
                        .transpose()?,
                    planning_policy: Default::default(),
                },
            )
            .context("failed to plan model input")?;
            println!("{}", serde_json::to_string_pretty(&plan)?);
        }
    }

    Ok(())
}

fn read_json<T: serde::de::DeserializeOwned>(path: &PathBuf, label: &str) -> Result<T> {
    let data = std::fs::read(path)
        .with_context(|| format!("failed to read {label} JSON at {}", path.display()))?;
    serde_json::from_slice(&data)
        .with_context(|| format!("failed to parse {label} JSON at {}", path.display()))
}
