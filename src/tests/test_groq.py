"""Quick test of Groq API for VLM"""
import os
from dotenv import load_dotenv
load_dotenv('/app/.env')

from groq import Groq
import base64
from PIL import Image
import numpy as np
import io

# Create test image
test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
buffer = io.BytesIO()
test_img.save(buffer, format='PNG')
base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

print("Calling Groq API...")
completion = client.chat.completions.create(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Return a simple JSON: {\"test\": \"success\"}"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
        ]
    }],
    temperature=0.3,
    max_completion_tokens=100,
    response_format={"type": "json_object"}
)

print(f"Response: {completion.choices[0].message.content}")
print("✅ Groq API working!")
