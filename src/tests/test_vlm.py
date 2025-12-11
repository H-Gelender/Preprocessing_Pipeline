"""
Hulu-Med VLM with 8-bit quantization to reduce memory
"""
import os
os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/huggingface"

print("Step 1: Importing libraries...")
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

model_path = "ZJU-AI4H/Hulu-Med-4B"

print("Step 2: Loading processor...")
processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True
)
print("Processor loaded!")

print("Step 3: Loading model with 8-bit quantization...")
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    quantization_config=quantization_config,
    device_map="auto",
    low_cpu_mem_usage=True
)
print("Model loaded!")

print("Step 4: Testing simple inference...")
from PIL import Image
import numpy as np

# Create a simple test image
test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
test_img.save("/tmp/test.png")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": {"image_path": "/tmp/test.png"}},
            {"type": "text", "text": "What do you see?"}
        ]
    }
]

print("Processing inputs...")
inputs = processor(
    conversation=conversation,
    add_system_prompt=True,
    add_generation_prompt=True,
    return_tensors="pt"
)

# Move to device and cast dtype
inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

print("Generating response...")
output_ids = model.generate(**inputs, max_new_tokens=50)
response = processor.batch_decode(output_ids, skip_special_tokens=True, use_think=False)[0]
print(f"Response: {response}")
print("SUCCESS!")
