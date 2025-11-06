// use bvedge_core::models::embedding::EncodeTokenizedRegions;
use candle_core::Device;

use bvedge_core::models::AtacformerForCellClustering;
use bvedge_core::models::loader::FromPretrained;

fn main() {
    let repo_id = "databio/atacformer-base-hg38";
    let model = AtacformerForCellClustering::from_pretrained(repo_id, &Device::Cpu).unwrap();
    // let result = model.encode(vec![101, 202, 303], Some(10usize)).unwrap();
    // println!("{:?}", result);
}
