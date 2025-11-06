use candle_core::Tensor;

pub fn convert_pytorch_transformer_weights(
    pytorch_state_dict: &std::collections::HashMap<String, Tensor>,
) -> Result<std::collections::HashMap<String, Tensor>, Box<dyn std::error::Error>> {
    let mut candle_weights = std::collections::HashMap::new();

    for (key, tensor) in pytorch_state_dict {
        // PyTorch uses: encoder.layers.{i}.self_attn.in_proj_weight
        // We need to split into q, k, v projections

        if key.contains("in_proj_weight") {
            // Split concatenated QKV weights
            let chunks = tensor.chunk(3, 0)?;
            let base_key = key.replace("in_proj_weight", "");
            candle_weights.insert(format!("{}q_proj.weight", base_key), chunks[0].clone());
            candle_weights.insert(format!("{}k_proj.weight", base_key), chunks[1].clone());
            candle_weights.insert(format!("{}v_proj.weight", base_key), chunks[2].clone());
        } else if key.contains("in_proj_bias") {
            let chunks = tensor.chunk(3, 0)?;
            let base_key = key.replace("in_proj_bias", "");
            candle_weights.insert(format!("{}q_proj.bias", base_key), chunks[0].clone());
            candle_weights.insert(format!("{}k_proj.bias", base_key), chunks[1].clone());
            candle_weights.insert(format!("{}v_proj.bias", base_key), chunks[2].clone());
        } else {
            candle_weights.insert(key.clone(), tensor.clone());
        }
    }

    Ok(candle_weights)
}
