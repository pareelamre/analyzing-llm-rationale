# Local Inference Setup for Qwen2.5-7B-Instruct

This guide shows how to run the Qwen model locally instead of using the HuggingFace API.

## Prerequisites

Install required packages:
```bash
pip install transformers torch huggingface_hub accelerate
```

For GPU acceleration (recommended):
```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Step 1: Download the Model

Run the download script:
```bash
python download_qwen_model.py
```

This will download ~15GB of model weights to `~/.cache/huggingface/hub/`.

**Note**: The download may take 30-60 minutes depending on your internet speed.

## Step 2: Run Local Inference

Use the modular runner instead of the API-based version:

```bash
# Run variant 3 with local inference
python scripts/run_variant.py --variant variant3_reasoning_type

# Process only 10 records (for testing)
MAX_RECORDS=10 python scripts/run_variant.py --variant variant3_reasoning_type
```

## Performance Comparison

| Method | Cost | Speed | Setup |
|--------|------|-------|-------|
| HuggingFace API | Pay per token | Fast | Easy |
| Local Inference (GPU) | Free | Medium | Requires GPU |
| Local Inference (CPU) | Free | Slow | Works anywhere |

### GPU Requirements

- **Minimum**: 8GB VRAM (can run with float16)
- **Recommended**: 16GB+ VRAM (for better performance)
- **CPU only**: Possible but ~10-20x slower

## Running Other Variants

Pick a different configured variant name:

```bash
python scripts/run_variant.py --variant variant5_key_conditions
python scripts/run_variant.py --variant variant6_step_by_step_reasoning
```

## Troubleshooting

### Out of Memory Error

If you get CUDA out of memory errors:
1. Reduce batch size (already set to 1)
2. Use CPU instead: edit `local_inference.py` and change `device="cpu"`
3. Close other GPU-using programs

### Model Not Found

If the model isn't found after downloading:
- Check `~/.cache/huggingface/hub/` has the model files
- Re-run `download_qwen_model.py`

### Slow Performance on CPU

CPU inference is much slower (~2-3 minutes per question vs. ~5-10 seconds on GPU).
Consider:
- Running overnight for large batches
- Using a machine with GPU access
- Using HuggingFace API with credits

## Files Created

- `download_qwen_model.py`: Downloads the Qwen model
- `local_inference.py`: Local inference wrapper
- `scripts/run_variant.py`: Config-driven runner for local or router inference
- This README

## Memory Usage

- **Model loading**: ~14GB (GPU) or ~28GB (CPU)
- **Inference**: +2-4GB per query
- **Total**: ~16-18GB GPU or ~30-32GB RAM (CPU)
