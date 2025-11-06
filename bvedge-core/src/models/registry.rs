//! Support model registry
//!
//! This is the centralized location that keeps track of models that we support
//! from our huggingface organization
use crate::models::atacformer::AtacformerForCellClustering;
use crate::models::embedding::EncodeTokenizedRegions;
use crate::models::loader::FromPretrained;
use candle_core::{Device, Result};

/// Enum of models supported by bvedge
#[derive(Debug, Clone, Copy)]
pub enum SupportedModel {
    AtacformerBaseHg38,
    AtacformerCtftHg38,
}

impl SupportedModel {
    /// Get the HuggingFace repository ID for this model
    pub fn repo_id(&self) -> &str {
        match self {
            Self::AtacformerBaseHg38 => "databio/atacformer-base-hg38",
            Self::AtacformerCtftHg38 => "databio/atacformer-ctft-hg38",
        }
    }

    /// Load this model from HuggingFace Hub
    ///
    /// This method downloads the model from HF and instantiates it.
    /// Currently all supported models are Atacformer variants.
    pub fn load(&self, device: &Device) -> Result<Box<dyn EncodeTokenizedRegions>> {
        let repo_id = self.repo_id();

        match self {
            Self::AtacformerBaseHg38 | Self::AtacformerCtftHg38 => {
                let model = AtacformerForCellClustering::from_pretrained(repo_id, device)?;
                Ok(Box::new(model))
            }
        }
    }
}
