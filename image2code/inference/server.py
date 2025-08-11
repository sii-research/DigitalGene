from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
import base64
import io
from PIL import Image
import tempfile
import os

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from param_dims import param_dims

class InferenceRequest(BaseModel):
    image: str  # base64 encoded image
    category: str

class InferenceResponse(BaseModel):
    output_text: str
    success: bool
    error: Optional[str] = None

class VisionLanguageService:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.detailed_prompt = """
You are given a task that involves both language reasoning and image understanding. Based on the provided textual and visual inputs, estimate the underlying structure and parameters of the described object. Your goal is to generate a structured representation of the object as JSON code.

Use both linguistic reasoning and visual cues to infer the object's geometry, configuration, and relevant parameters.

All numerical values in the code should be linearly mapped and discretized into integers within the range 2048 to 3072.

The final output must be a JSON code block enclosed within <code> and </code> tags. Only include the code inside these tags -- no explanations, descriptions, or formatting outside of them.

Ensure your output is accurate, complete, and strictly adheres to this format.
"""
    
    def load_model(self):
        """Load model and processor once during service startup"""
        print("Loading model...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype="auto", device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        print("Model loaded successfully!")
    
    def process_image(self, base64_image: str) -> str:
        """Convert base64 image to temporary file path"""
        try:
            # Decode base64 image
            image_data = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_data))
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            image.save(temp_file.name)
            temp_file.close()
            return temp_file.name
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")
    
    def inference(self, image_path: str, category: str) -> str:
        return self.inference_code(image_path, category)
    
    def inference_code(self, image_path: str, category: str) -> str:
        """Run inference on image and category"""
        try:
            # Create instruction
            if category in param_dims:
                instruction = f"Inspect the image, classify the '{category}' object, and return its size, shape, and position in a structured JSON format."
                instruction += f"\nValid Concepts of {category}: " + ", ".join(param_dims[category].keys())
            else:
                instruction = "Examine the objects in this image and provide their type and physical characteristics in JSON format."
            # Prepare messages
            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.detailed_prompt}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"file://{image_path}",
                        },
                        {"type": "text", "text": instruction}
                    ],
                }
            ]
            
            # Process inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            # Generate output
            generated_ids = self.model.generate(**inputs, max_new_tokens=8192)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0] if output_text else ""
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(image_path):
                os.unlink(image_path)

# Initialize service
MODEL_PATH = ""  # set your model path
service = VisionLanguageService(MODEL_PATH)

# Create FastAPI app
app = FastAPI(title="Vision Language Service", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """Load model when service starts"""
    service.load_model()

@app.post("/inference", response_model=InferenceResponse)
async def inference_endpoint(request: InferenceRequest):
    """Inference endpoint"""
    try:
        # Process image
        image_path = service.process_image(request.image)
        # Run inference
        output_text = service.inference(image_path, request.category)
        return InferenceResponse(
            output_text=output_text,
            success=True
        )
    except HTTPException:
        raise
    except Exception as e:
        return InferenceResponse(
            output_text="",
            success=False,
            error=str(e)
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": service.model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
