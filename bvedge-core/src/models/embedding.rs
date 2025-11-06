use candle_core::{Tensor, Error};

pub trait EncodeTokenizedRegions {
    /// Encode a set of genomic regions
    fn encode(&self, input_ids: Vec<u32>, max_length: Option<usize>) -> Result<Tensor, Error>;
}