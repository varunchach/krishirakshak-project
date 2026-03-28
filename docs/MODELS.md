# Model Selection — KrishiRakshak

## Why Florence-2 and Not Others

### Models Considered

| Model | Params | Fine-tunable on T4? | Image→Text? | License | Verdict |
|-------|--------|---------------------|-------------|---------|---------|
| **Florence-2-base** | **232M** | **Yes (<1hr)** | **Yes (seq2seq)** | **MIT** | **SELECTED** |
| Florence-2-large | 770M | Yes (2-3hr) | Yes | MIT | Backup if budget allows |
| PaliGemma-3B | 3B | Tight on T4 | Yes | Gemma license | Too large for $250 |
| LLaVA 1.6 7B | 7B | No (needs A100) | Yes | Llama license | Way over budget |
| Qwen2-VL 2B | 2B | Tight on T4 | Yes | Apache 2.0 | Possible but less tested |
| EfficientNet-B0 | 5.3M | Yes | No (classifier only) | Apache 2.0 | Students already did this |
| ResNet-50 | 25M | Yes | No (classifier only) | BSD | Students already did this |

### Florence-2 Architecture (for student explanation)

```
Input Image ──► DaViT Vision Encoder ──► Visual Embeddings ──┐
                                                               ├──► Transformer Encoder-Decoder ──► Text Output
Text Prompt ──► BERT Tokenizer ──────► Text Embeddings ───────┘
```

- **DaViT:** Dual Attention Vision Transformer — captures both local and global features
- **Encoder-Decoder:** Standard seq2seq — like T5 but for vision+language
- **Output:** Free-form text generated token by token

### Fine-tuning Strategy

#### LoRA Configuration
```yaml
lora_config:
  r: 8                    # Rank — low for small model
  lora_alpha: 16           # Scaling factor
  lora_dropout: 0.05
  target_modules:          # Apply LoRA to attention layers only
    - "q_proj"
    - "v_proj"
  bias: "none"
  task_type: "CAUSAL_LM"
```

#### Training Hyperparameters
```yaml
training:
  epochs: 10
  batch_size: 8
  learning_rate: 1e-4
  weight_decay: 0.01
  warmup_ratio: 0.1
  lr_scheduler: "cosine"
  max_length: 512
  fp16: true
  gradient_accumulation_steps: 4
  freeze_vision_encoder: true   # CRITICAL — saves memory
```

#### Dataset Format for Florence-2
Florence-2 uses task-specific prompt tokens. Custom task: `<CROP_DISEASE>`

```json
{
  "image": "path/to/tomato_early_blight_001.jpg",
  "prefix": "<CROP_DISEASE>",
  "suffix": "Tomato Early Blight. Confidence: High. Treatment: Remove infected leaves immediately. Apply Mancozeb 75WP at 2.5g per litre of water. Spray at 7-day intervals. Ensure proper plant spacing for air circulation. Rotate crops next season."
}
```

#### What the Model Learns
- Given a leaf image + `<CROP_DISEASE>` prompt → generate disease name + structured treatment
- Fine-tuning teaches it agricultural vocabulary, Indian pesticide names, dosage formats
- After fine-tuning, it should output structured text parseable by the pipeline

### Evaluation Metrics

| Metric | Base Florence-2 | Fine-tuned (target) |
|--------|----------------|-------------------|
| Disease classification accuracy | ~0% (not trained for this) | >90% |
| Treatment relevance (LLM-judged) | N/A | >80% |
| Inference latency (T4) | <1s | <1.5s |
| Model size (adapter) | N/A | ~10MB |

### Bedrock Fallback Logic
```python
def should_use_bedrock(florence_output):
    """Use Bedrock when Florence-2 output is insufficient"""
    if florence_output.confidence < 0.4:
        return True  # Low confidence — need better reasoning
    if len(florence_output.treatment) < 50:
        return True  # Treatment too short — augment
    if "unknown" in florence_output.disease.lower():
        return True  # Model uncertain
    return False
```

### Sarvam Model Details

#### Mayura v1 (Translation)
- Supports: en-IN → hi-IN, mr-IN, ta-IN, te-IN, kn-IN, bn-IN, gu-IN, ml-IN, od-IN, pa-IN
- Modes: formal, colloquial
- Pricing: ₹2.3 per 10K characters (covered by free ₹1,000 credits)

#### Bulbul v1 (Text-to-Speech)
- Voices: meera (female), arvind (male), and 4 others
- Languages: hi-IN, ta-IN, te-IN, kn-IN, bn-IN, ml-IN
- Output: Base64 encoded WAV audio
- Pricing: ₹3 per 10K characters (covered by free credits)

### Model Versioning Strategy
```
v1.0.0 — Base Florence-2-base-ft (zero-shot baseline)
v1.1.0 — LoRA fine-tuned on PlantVillage (38 classes)
v1.2.0 — Retrained with farmer feedback data (future)
```

Each version registered in SageMaker Model Registry with:
- Accuracy on test set
- Inference latency benchmark
- Training data hash
- Approval status (PendingApproval → Approved → Deployed)
