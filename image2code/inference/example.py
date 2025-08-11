from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from param_dims import param_dims

# model
model_path: str = ""
# image
image_path: str = ""
# category
cat: str = ""   # ["Bottle", "Box", "Bucket", ...]

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="auto"
)
# default processor
processor = AutoProcessor.from_pretrained(model_path)

DETAILED_PROMPT = """
You are given a task that involves both language reasoning and image understanding. Based on the provided textual and visual inputs, estimate the underlying structure and parameters of the described object. Your goal is to generate a structured representation of the object as JSON code.

Use both linguistic reasoning and visual cues to infer the object's geometry, configuration, and relevant parameters.

All numerical values in the code should be linearly mapped and discretized into integers within the range 2048 to 3072.

The final output must be a JSON code block enclosed within <code> and </code> tags. Only include the code inside these tags — no explanations, descriptions, or formatting outside of them.

Ensure your output is accurate, complete, and strictly adheres to this format.
"""

instruction: str = "Inspect the image, classify the '{}' object, and return its size, shape, and position in a structured JSON format.".format(cat)
instruction += f"\nValid Concepts of {cat}: " + ", ".join(param_dims[cat].keys())

# messages
messages = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": DETAILED_PROMPT}
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

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
