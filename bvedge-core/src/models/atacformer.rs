use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, VarBuilder, embedding, layer_norm, linear};
use serde::Deserialize;

use crate::models::embedding::EncodeTokenizedRegions;
use crate::models::loader::{FromPretrained, HubLoader};

#[derive(Debug, Clone, Deserialize)]
pub struct AtacformerConfig {
    #[serde(default = "default_use_pos_embeddings")]
    pub use_pos_embeddings: bool,
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_pad_token_id")]
    pub pad_token_id: u32,
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: u32,
    #[serde(default = "default_bos_token_id")]
    pub bos_token_id: u32,
    #[serde(default = "default_cls_token_id")]
    pub cls_token_id: u32,
    #[serde(default = "default_sep_token_id")]
    pub sep_token_id: u32,
    #[serde(default = "default_sparse_prediction")]
    pub sparse_prediction: bool,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    #[serde(default = "default_embedding_dropout")]
    pub embedding_dropout: f64,
    #[serde(default = "default_initializer_range")]
    pub initializer_range: f64,
    #[serde(default = "default_initializer_cutoff_factor")]
    pub initializer_cutoff_factor: f64,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    pub num_batches: Option<usize>,
    #[serde(default = "default_lambda_adv")]
    pub lambda_adv: f64,
    #[serde(default = "default_grl_alpha")]
    pub grl_alpha: f64,
    #[serde(default = "default_bc_unfreeze_last_n_layers")]
    pub bc_unfreeze_last_n_layers: usize,
}

fn default_use_pos_embeddings() -> bool {
    true
}
fn default_vocab_size() -> usize {
    890711
}
fn default_max_position_embeddings() -> usize {
    8192
}
fn default_hidden_size() -> usize {
    384
}
fn default_intermediate_size() -> usize {
    1536
}
fn default_num_hidden_layers() -> usize {
    6
}
fn default_num_attention_heads() -> usize {
    8
}
fn default_pad_token_id() -> u32 {
    890705
}
fn default_eos_token_id() -> u32 {
    890708
}
fn default_bos_token_id() -> u32 {
    890709
}
fn default_cls_token_id() -> u32 {
    890707
}
fn default_sep_token_id() -> u32 {
    890710
}
fn default_sparse_prediction() -> bool {
    true
}
fn default_norm_eps() -> f64 {
    1e-5
}
fn default_embedding_dropout() -> f64 {
    0.0
}
fn default_initializer_range() -> f64 {
    0.02
}
fn default_initializer_cutoff_factor() -> f64 {
    2.0
}
fn default_tie_word_embeddings() -> bool {
    true
}
fn default_lambda_adv() -> f64 {
    1.0
}
fn default_grl_alpha() -> f64 {
    1.0
}
fn default_bc_unfreeze_last_n_layers() -> usize {
    2
}

pub struct AtacformerEmbeddings {
    token_embeddings: Embedding,
    position_embeddings: Embedding,
    use_pos_embeddings: bool,
}

impl AtacformerEmbeddings {
    pub fn load(vb: VarBuilder, config: &AtacformerConfig) -> Result<Self> {
        let token_embeddings = embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("token_embeddings"),
        )?;

        let position_embeddings = embedding(
            config.max_position_embeddings,
            config.hidden_size,
            vb.pp("position_embeddings"),
        )?;

        Ok(Self {
            token_embeddings,
            position_embeddings,
            use_pos_embeddings: config.use_pos_embeddings,
        })
    }

    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_bsz, seq_len) = input_ids.dims2()?; // (batch size, num_regions)

        // create position IDs [0, 1, 2, ..., seq_len-1]
        let position_ids = Tensor::arange(0u32, seq_len as u32, input_ids.device())?
            .unsqueeze(0)?
            .broadcast_as(input_ids.shape())?;

        let token_emb = self.token_embeddings.forward(input_ids)?;

        if self.use_pos_embeddings {
            let pos_emb = self.position_embeddings.forward(&position_ids)?;
            token_emb.add(&pos_emb)
        } else {
            Ok(token_emb)
        }
    }
}

pub struct TransformerEncoderLayer {
    self_attn: MultiHeadAttention,
    linear1: Linear,
    linear2: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl TransformerEncoderLayer {
    pub fn load(
        vb: VarBuilder,
        hidden_size: usize,
        num_heads: usize,
        intermediate_size: usize,
        norm_eps: f64,
    ) -> Result<Self> {
        let self_attn = MultiHeadAttention::load(vb.pp("self_attn"), hidden_size, num_heads)?;

        let linear1 = linear(hidden_size, intermediate_size, vb.pp("linear1"))?;
        let linear2 = linear(intermediate_size, hidden_size, vb.pp("linear2"))?;
        let norm1 = layer_norm(hidden_size, norm_eps, vb.pp("norm1"))?;
        let norm2 = layer_norm(hidden_size, norm_eps, vb.pp("norm2"))?;

        Ok(Self {
            self_attn,
            linear1,
            linear2,
            norm1,
            norm2,
        })
    }

    pub fn forward(&self, x: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // self-attention with residual
        let attn_output = self.self_attn.forward(x, x, x, attention_mask)?;
        let x = (x + attn_output)?;
        let x = self.norm1.forward(&x)?;

        // ffn with residual
        let ffn_output = self.linear1.forward(&x)?.relu()?;
        let ffn_output = self.linear2.forward(&ffn_output)?;
        let x = (x + ffn_output)?;
        self.norm2.forward(&x)
    }
}

pub struct MultiHeadAttention {
    in_proj_weight: Tensor,
    in_proj_bias: Option<Tensor>,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl MultiHeadAttention {
    pub fn load(vb: VarBuilder, hidden_size: usize, num_heads: usize) -> Result<Self> {
        let head_dim = hidden_size / num_heads;

        // PyTorch's MultiheadAttention uses in_proj_weight (3 * hidden_size, hidden_size)
        // PyTorch Linear stores weights as (out_features, in_features)
        // So shape is (3 * hidden_size, hidden_size) = (576, 192) for hidden_size=192
        let in_proj_weight = vb.get((3 * hidden_size, hidden_size), "in_proj_weight")?;
        let in_proj_bias = vb.get(3 * hidden_size, "in_proj_bias").ok();

        Ok(Self {
            in_proj_weight,
            in_proj_bias,
            out_proj: linear(hidden_size, hidden_size, vb.pp("out_proj"))?,
            num_heads,
            head_dim,
            hidden_size,
        })
    }

    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = query.dims3()?;

        // For self-attention, query/key/value should be the same tensor
        // Apply combined QKV projection: (batch, seq, hidden) @ (3*hidden, hidden)^T
        // Result: (batch, seq, 3*hidden)
        let qkv = query.matmul(&self.in_proj_weight.t()?)?;

        // Add bias if present
        let qkv = if let Some(bias) = &self.in_proj_bias {
            qkv.broadcast_add(bias)?
        } else {
            qkv
        };

        // Split into Q, K, V: each is (batch, seq, hidden)
        let q = qkv.narrow(2, 0, self.hidden_size)?;
        let k = qkv.narrow(2, self.hidden_size, self.hidden_size)?;
        let v = qkv.narrow(2, 2 * self.hidden_size, self.hidden_size)?;

        // Reshape to (batch, num_heads, seq_len, head_dim)
        let q = q
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Attention scores
        let scale = (self.head_dim as f64).sqrt();
        let mut attn_weights = (q.matmul(&k.transpose(2, 3)?)? / scale)?;

        // Apply mask if provided
        if let Some(mask) = attention_mask {
            // mask is (batch, seq_len), need (batch, 1, seq_len, seq_len)
            let mask = mask
                .unsqueeze(1)?
                .unsqueeze(2)?
                .broadcast_as(attn_weights.shape())?;

            // Where mask is 0, set to large negative value
            let mask_value = Tensor::new(f32::NEG_INFINITY, attn_weights.device())?;
            attn_weights = mask.where_cond(&attn_weights, &mask_value)?;
        }

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // Apply attention to values
        let output = attn_weights.matmul(&v)?.transpose(1, 2)?.reshape((
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        self.out_proj.forward(&output)
    }
}

pub struct AtacformerModel {
    embeddings: AtacformerEmbeddings,
    encoder_layers: Vec<TransformerEncoderLayer>,
}

impl AtacformerModel {
    pub fn load(vb: VarBuilder, config: &AtacformerConfig) -> Result<Self> {
        let embeddings = AtacformerEmbeddings::load(vb.pp("embeddings"), config)?;

        let mut encoder_layers = Vec::new();
        let vb_encoder = vb.pp("encoder").pp("layers");

        for i in 0..config.num_hidden_layers {
            let layer = TransformerEncoderLayer::load(
                vb_encoder.pp(i),
                config.hidden_size,
                config.num_attention_heads,
                config.intermediate_size,
                config.norm_eps,
            )?;
            encoder_layers.push(layer);
        }

        Ok(Self {
            embeddings,
            encoder_layers,
        })
    }

    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.embeddings.forward(input_ids)?;

        for layer in &self.encoder_layers {
            hidden_states = layer.forward(&hidden_states, Some(attention_mask))?;
        }

        Ok(hidden_states)
    }
}

pub struct AtacformerForCellClustering {
    atacformer: AtacformerModel,
    config: AtacformerConfig,
}

impl AtacformerForCellClustering {
    pub fn load(vb: VarBuilder, config: &AtacformerConfig) -> Result<Self> {
        let atacformer = AtacformerModel::load(vb.pp("atacformer"), config)?;

        Ok(Self {
            atacformer,
            config: config.clone(),
        })
    }

    /// mean pooling over sequence dimension, respecting attention mask
    fn mean_pooling(&self, embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // embeddings: (batch, seq_len, hidden_size)
        // attention_mask: (batch, seq_len)

        let mask_expanded = attention_mask
            .unsqueeze(2)?
            .broadcast_as(embeddings.shape())?
            .to_dtype(embeddings.dtype())?;

        // sum embeddings where mask is 1
        let sum_embeddings = (embeddings * &mask_expanded)?.sum(1)?;

        // count non-masked tokens
        let sum_mask = attention_mask
            .to_dtype(DType::F32)?
            .sum(1)?
            .unsqueeze(1)?
            .clamp(1e-9, f32::MAX)?;

        // average
        sum_embeddings.broadcast_div(&sum_mask)
    }

    /// forward pass for single input (inference mode)
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        let hidden_states = self.atacformer.forward(input_ids, attention_mask)?;
        self.mean_pooling(&hidden_states, attention_mask)
    }

    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }
}

impl EncodeTokenizedRegions for AtacformerForCellClustering {
    /// encode a batch of tokenized sequences
    fn encode_batch(
        &self,
        input_ids: &[Vec<u32>],
        max_length: Option<usize>,
    ) -> std::result::Result<Tensor, candle_core::Error> {
        let device = self
            .atacformer
            .embeddings
            .token_embeddings
            .embeddings()
            .device();
        let max_len = max_length.unwrap_or(self.config.max_position_embeddings);
        let pad_id = self.config.pad_token_id;

        // Truncate/pad sequences
        let batch_size = input_ids.len();
        let mut padded_batch = vec![vec![pad_id; max_len]; batch_size];
        let mut mask_batch = vec![vec![0u8; max_len]; batch_size];

        for (i, seq) in input_ids.iter().enumerate() {
            let seq_len = seq.len().min(max_len);
            padded_batch[i][..seq_len].copy_from_slice(&seq[..seq_len]);
            mask_batch[i][..seq_len].fill(1);
        }

        // Convert to tensors
        let input_ids_flat: Vec<u32> = padded_batch.into_iter().flatten().collect();
        let mask_flat: Vec<u8> = mask_batch.into_iter().flatten().collect();

        let input_ids_tensor = Tensor::from_vec(input_ids_flat, (batch_size, max_len), device)?;

        let attention_mask = Tensor::from_vec(mask_flat, (batch_size, max_len), device)?;

        self.forward(&input_ids_tensor, &attention_mask)
    }
}

impl FromPretrained for AtacformerForCellClustering {
    fn from_pretrained(repo_id: &str, device: &Device) -> Result<Self> {
        let loader = HubLoader::new()?;

        // Download required files from HuggingFace
        let config_path = loader.download_file(repo_id, "config.json")?;
        let weights_path = loader.download_file(repo_id, "model.safetensors")?;

        // Load config
        let config_json = std::fs::read_to_string(config_path)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to read config: {}", e)))?;
        let config: AtacformerConfig = serde_json::from_str(&config_json)
            .map_err(|e| candle_core::Error::Msg(format!("Failed to parse config: {}", e)))?;

        // Load weights using VarBuilder from safetensors
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)? };

        // Load model with weights
        Self::load(vb, &config)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    fn create_test_config() -> AtacformerConfig {
        AtacformerConfig {
            use_pos_embeddings: true,
            vocab_size: 100,
            max_position_embeddings: 512,
            hidden_size: 64,
            intermediate_size: 256,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            pad_token_id: 0,
            eos_token_id: 1,
            bos_token_id: 2,
            cls_token_id: 3,
            sep_token_id: 4,
            sparse_prediction: false,
            norm_eps: 1e-5,
            embedding_dropout: 0.0,
            initializer_range: 0.02,
            initializer_cutoff_factor: 2.0,
            tie_word_embeddings: false,
            num_batches: None,
            lambda_adv: 1.0,
            grl_alpha: 1.0,
            bc_unfreeze_last_n_layers: 2,
        }
    }

    #[rstest]
    fn test_config_defaults() {
        let config = AtacformerConfig {
            use_pos_embeddings: default_use_pos_embeddings(),
            vocab_size: default_vocab_size(),
            max_position_embeddings: default_max_position_embeddings(),
            hidden_size: default_hidden_size(),
            intermediate_size: default_intermediate_size(),
            num_hidden_layers: default_num_hidden_layers(),
            num_attention_heads: default_num_attention_heads(),
            pad_token_id: default_pad_token_id(),
            eos_token_id: default_eos_token_id(),
            bos_token_id: default_bos_token_id(),
            cls_token_id: default_cls_token_id(),
            sep_token_id: default_sep_token_id(),
            sparse_prediction: default_sparse_prediction(),
            norm_eps: default_norm_eps(),
            embedding_dropout: default_embedding_dropout(),
            initializer_range: default_initializer_range(),
            initializer_cutoff_factor: default_initializer_cutoff_factor(),
            tie_word_embeddings: default_tie_word_embeddings(),
            num_batches: None,
            lambda_adv: default_lambda_adv(),
            grl_alpha: default_grl_alpha(),
            bc_unfreeze_last_n_layers: default_bc_unfreeze_last_n_layers(),
        };

        assert_eq!(config.vocab_size, 890711);
        assert_eq!(config.hidden_size, 384);
        assert_eq!(config.num_attention_heads, 8);
    }

    #[rstest]
    #[case("databio/atacformer-base-hg38", 192)]
    #[case("databio/atacformer-craft100k-hg38", 192)]
    fn test_create_model_from_hf(#[case] repo_id: &str, #[case] hidden_size: usize) -> Result<()> {
        let device = Device::Cpu;

        let model = AtacformerForCellClustering::from_pretrained(repo_id, &device)?;
        assert_eq!(model.config.vocab_size, default_vocab_size());
        assert_eq!(model.config.hidden_size, hidden_size);

        Ok(())
    }

    #[rstest]
    fn test_embeddings_forward() -> Result<()> {
        let device = Device::Cpu;
        let config = create_test_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let embeddings = AtacformerEmbeddings::load(vb, &config)?;

        let batch_size = 2;
        let seq_len = 10;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device)?;

        let output = embeddings.forward(&input_ids)?;
        assert_eq!(output.dims(), &[batch_size, seq_len, config.hidden_size]);

        Ok(())
    }

    #[rstest]
    fn test_atacformer_for_cell_clustering_forward() -> Result<()> {
        let device = Device::Cpu;
        let config = create_test_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = AtacformerForCellClustering::load(vb, &config)?;

        let batch_size = 1;
        let seq_len = 10;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device)?;
        let attention_mask = Tensor::ones((batch_size, seq_len), DType::U8, &device)?;

        let output = model.forward(&input_ids, &attention_mask)?;

        // Output should be mean-pooled, so (batch_size, hidden_size)
        assert_eq!(output.dims(), &[batch_size, config.hidden_size]);

        Ok(())
    }

    #[rstest]
    fn test_atacformer_forward_with_padding() -> Result<()> {
        let device = Device::Cpu;
        let config = create_test_config();
        let vb = VarBuilder::zeros(DType::F32, &device);

        let model = AtacformerForCellClustering::load(vb, &config)?;

        let batch_size = 2;
        let seq_len = 10;
        let input_ids = Tensor::zeros((batch_size, seq_len), DType::U32, &device)?;

        // Create attention mask with padding (second half is padding)
        let mask_data = vec![1u8; batch_size * seq_len / 2]
            .into_iter()
            .chain(vec![0u8; batch_size * seq_len / 2])
            .collect::<Vec<_>>();
        let attention_mask = Tensor::from_vec(mask_data, (batch_size, seq_len), &device)?;

        let output = model.forward(&input_ids, &attention_mask)?;

        assert_eq!(output.dims(), &[batch_size, config.hidden_size]);

        Ok(())
    }

    #[rstest]
    #[case("databio/atacformer-base-hg38")]
    #[case("databio/atacformer-craft100k-hg38")]
    fn test_embed_bed_file(#[case] repo_id: &str) -> Result<()> {
        let device = Device::Cpu;

        let model = AtacformerForCellClustering::from_pretrained(repo_id, &device)?;

        let input_ids = vec![vec![101u32, 202, 345]];
        let res = model.encode_batch(&input_ids, Some(3))?;

        assert_eq!(res.dims(), &[1, 3, model.hidden_size()]);

        Ok(())
    }
}
