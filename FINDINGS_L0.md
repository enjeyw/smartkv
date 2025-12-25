# SmartKV: Findings on Layer 0 & Thresholding

## 1. Threshold Sensitivity
- **Experiment**: Lowered attention threshold from `0.11` to `0.05`.
- **Result**: Significant increase in predictability (R²) for Layer 0 heads.
    - `l0_h6`: R² improved from <0.4 to **0.71**.
    - `l0_h2`, `l0_h7`, `l0_h4`: All showed R² > 0.65.
- **Conclusion**: The previous threshold of 0.11 was too strict for early layers, discarding valid local dependency signals as noise. 0.05 is a better baseline for these layers.

## 2. Layer 0 Characterization
- **Short-Sightedness**: Despite high predictability (R²), the predicted *lifespan* scores for Layer 0 heads remain low (typically < 2.0 in log4 space, i.e., < 16 tokens).
- **The `l0_h3` Case**:
    - **Observation**: Predicted lifespan is capped around $2^{2.8} \approx 7$ tokens.
    - **Analysis**: This head likely specializes in immediate local syntax (e.g., attending to the previous word). Even if ground truth shows occasional long-range attention (attention sinks or noise), the semantic content of the key only justifies a short lifespan.
- **Implication**: Layer 0 keys can be evicted very aggressively (e.g., sliding window of 16-32 tokens) without learning a complex gate, or the gate can confirm they are "short-term only".

## 3. Recommendations for Deployment
1.  **Hybrid Policy**:
    - **Layer 0-2**: Use a fixed short sliding window (e.g., 32 tokens) or the trained gates with a low cutoff. They are local processors.
    - **Middle Layers (6-14)**: These showed the highest "Needle" retrieval scores (~1.8+). These are the critical "Long Term Memory" layers. Use the Gates here to selectively keep important keys.
    - **Deep Layers (20+)**: These had near-zero scores for the needle. They likely process the immediate next-token prediction state. Aggressive eviction or strict "recent-only" policy is safe.
2.  **Training**: Continue using `THRESHOLD=0.05` for future training runs.

## 4. Next Steps
- [ ] Train all layers (1-28) with `THRESHOLD=0.05`.
- [ ] Implement the **Hybrid Policy** in the inference engine.
- [ ] Evaluate Perplexity on `PG-19` to ensure no degradation.

