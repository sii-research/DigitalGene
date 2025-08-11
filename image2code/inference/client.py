import requests
import base64
from PIL import Image
import io
import json

def encode_image_to_base64(image_path: str) -> str:
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_inference_service(image_path: str, category: str, service_url: str = "http://localhost:8000"):
    # Encode image
    base64_image = encode_image_to_base64(image_path)
    # Prepare request
    request_data = {
        "image": base64_image,
        "category": category
    }
    # Call service
    response = requests.post(f"{service_url}/inference", json=request_data)
    if response.status_code == 200:
        result = response.json()
        if result["success"]:
            print("Output:", result["output_text"])
        else:
            print("Error:", result["error"])
    else:
        print(f"Request failed: {response.status_code}")

# 使用示例
if __name__ == "__main__":
    image_path: str = ""
    category: str = ""  # ["Bottle", "Box", "Bucket", ...]
    call_inference_service(
        image_path=image_path,
        category=category
    )
