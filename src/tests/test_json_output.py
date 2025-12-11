"""
Quick test for JSON output from VLM
"""
import os
import sys
sys.path.append("/app/src")
from run_vlm_pipeline import HESTVLMAnalyzer
from PIL import Image
import numpy as np

print("Initializing VLM...")
vlm = HESTVLMAnalyzer()

print("\n--- Testing Patch JSON ---")
# Create dummy image
img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
result = vlm.analyze_patch(img)
print(f"Result type: {type(result)}")
print(f"Result: {result}")

if isinstance(result, dict) and "cell_morphology" in result:
    print("✅ Patch JSON valid")
else:
    print("❌ Patch JSON invalid")
