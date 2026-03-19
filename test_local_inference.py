#!/usr/bin/env python3
"""
Quick test script to verify local inference is working.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from local_inference import chat_completion_local

    print("Testing local Qwen inference...")
    print("="*60)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2? Answer in one sentence."}
    ]

    print("\nSending test query: 'What is 2+2?'")
    print("This will load the model (may take 30-60 seconds)...\n")

    response = chat_completion_local(messages, temperature=0.0)
    answer = response["choices"][0]["message"]["content"]

    print("Model response:")
    print("-" * 60)
    print(answer)
    print("-" * 60)
    print("\n✓ Local inference is working!")
    print("\nYou can now run: python scripts/run_variant.py --variant variant3_reasoning_type")

except ImportError as e:
    print(f"✗ Missing dependencies: {e}")
    print("\nInstall required packages:")
    print("  pip install transformers torch huggingface_hub accelerate")
    sys.exit(1)

except FileNotFoundError as e:
    print(f"✗ Model not found: {e}")
    print("\nDownload the model first:")
    print("  python download_qwen_model.py")
    sys.exit(1)

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
