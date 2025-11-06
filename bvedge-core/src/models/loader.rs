//! Model loading from HuggingFace Hub
//!
//! This module provides traits and utilities for loading models from HuggingFace.
//! Each model type implements the `FromPretrained` trait to define its own loading logic.

use candle_core::{Device, Result};
use hf_hub::{Repo, RepoType, api::sync::Api};
use std::path::PathBuf;

/// Trait for models that can be loaded from HuggingFace Hub
pub trait FromPretrained: Sized {
    /// Load model from HuggingFace repository
    ///
    /// # Arguments
    /// * `repo_id` - HuggingFace repository ID (e.g., "databio/atacformer-base-hg38")
    /// * `device` - Device to load the model on (CPU, CUDA, Metal)
    ///
    /// # Returns
    /// * `Result<Self>` - Loaded model instance
    fn from_pretrained(repo_id: &str, device: &Device) -> Result<Self>;
}

/// Helper struct for downloading files from HuggingFace Hub
pub struct HubLoader {
    api: Api,
}

impl HubLoader {
    /// Create a new HubLoader with default cache directory
    pub fn new() -> Result<Self> {
        let api = Api::new().map_err(|e| {
            candle_core::Error::Msg(format!("Failed to initialize HF Hub API: {}", e))
        })?;
        Ok(Self { api })
    }

    /// Create a new HubLoader with custom cache directory
    pub fn with_cache_dir(_cache_dir: PathBuf) -> Result<Self> {
        let api = Api::new().map_err(|e| {
            candle_core::Error::Msg(format!("Failed to initialize HF Hub API: {}", e))
        })?;
        // Note: hf-hub 0.3 doesn't expose cache_dir in constructor,
        // it uses env var HF_HOME. You can set this before calling.
        Ok(Self { api })
    }

    /// Download a file from a HuggingFace repository
    ///
    /// # Arguments
    /// * `repo_id` - Repository ID (e.g., "databio/atacformer-base-hg38")
    /// * `filename` - File to download (e.g., "model.safetensors", "config.json")
    ///
    /// # Returns
    /// * `Result<PathBuf>` - Path to downloaded file in cache
    pub fn download_file(&self, repo_id: &str, filename: &str) -> Result<PathBuf> {
        let repo = Repo::new(repo_id.to_string(), RepoType::Model);

        self.api.repo(repo).get(filename).map_err(|e| {
            candle_core::Error::Msg(format!(
                "Failed to download {} from {}: {}",
                filename, repo_id, e
            ))
        })
    }

    /// Download multiple files from a HuggingFace repository
    pub fn download_files(&self, repo_id: &str, filenames: &[&str]) -> Result<Vec<PathBuf>> {
        filenames
            .iter()
            .map(|filename| self.download_file(repo_id, filename))
            .collect()
    }
}

impl Default for HubLoader {
    fn default() -> Self {
        Self::new().expect("Failed to create default HubLoader")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires network access
    fn test_download_file() {
        let loader = HubLoader::new().unwrap();
        let path = loader
            .download_file("databio/atacformer-base-hg38", "config.json")
            .unwrap();
        assert!(path.exists());
    }
}
