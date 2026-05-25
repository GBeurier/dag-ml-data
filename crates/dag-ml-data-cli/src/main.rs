use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use dag_ml_data_core::{schema_fingerprint, DatasetSchema};

#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    FingerprintSchema { path: PathBuf },
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
    }

    Ok(())
}
