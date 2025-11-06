use candle_core::{Error, Tensor};

pub trait EncodeTokenizedRegions {
    /// Encode a batch of tokenized genomic regions
    fn encode_batch(
        &self,
        input_ids: &[Vec<u32>],
        max_length: Option<usize>,
    ) -> Result<Tensor, Error>;
}
