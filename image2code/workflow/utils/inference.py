"""
Inference utilities for image2code model.

This module provides inference functions for generating structured JSON representations
of objects from images using the image2code model.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from .param_dims import param_dims
from .prompt import baseline_prompt
from .qwen_hf import QwenVLGenerationModel

# Configuration constants
DETAILED_PROMPT = """
You are given a task that involves both language reasoning and image understanding. Based on the provided textual and visual inputs, estimate the underlying structure and parameters of the described object. Your goal is to generate a structured representation of the object as JSON code.

Use both linguistic reasoning and visual cues to infer the object's geometry, configuration, and relevant parameters.

All numerical values in the code should be linearly mapped and discretized into integers within the range 2048 to 3072.

The final output must be a JSON code block enclosed within <code> and </code> tags. Only include the code inside these tags -- no explanations, descriptions, or formatting outside of them.

Ensure your output is accurate, complete, and strictly adheres to this format.
"""

GENERIC_INSTRUCTION = "Examine the objects in this image and provide their type and physical characteristics in JSON format."


def post_process_text(text: str) -> str:
    """
    Clean and post-process generated text.
    
    Args:
        text: Raw generated text
    
    Returns:
        Cleaned text
    """
    # Fix parameter format
    text = text.replace('"parameters": [', '"parameters": ')
    # Replace Unicode characters with default token
    pattern = r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]'
    return re.sub(pattern, '2048', text)


def parse_generated_text(text: str) -> Dict[str, Any]:
    """
    Parse JSON from generated text.
    
    Args:
        text: Generated text containing JSON
    
    Returns:
        Parsed JSON object, empty dict if parsing fails
    """
    # Extract content from code tags
    if "<code>" in text and "</code>" in text:
        start = text.rfind("<code>") + len("<code>")
        end = text.rfind("</code>")
        text = text[start:end]
    try:
        processed_text = post_process_text(text)
        return json.loads(processed_text)
    except Exception as e:
        print(f"JSON parsing failed: {e}")
        return {}

def create_instruction(category: str = "") -> str:
    """
    Create instruction text based on category.
    
    Args:
        category: Object category
    
    Returns:
        Formatted instruction string
    """
    if category and category in param_dims:
        instruction = (
            f"Inspect the image, classify the '{category}' object, and return its "
            "size, shape, and position in a structured JSON format."
        )
        valid_concepts = ", ".join(param_dims[category].keys())
        instruction += f"\nValid Concepts of {category}: {valid_concepts}"
        return instruction
    return GENERIC_INSTRUCTION


def _inference_single(
    model: QwenVLGenerationModel,
    image_path: Union[str, Path],
    category: str = "",
    return_raw_output: bool = True
) -> Union[str, Tuple[str, str]]:
    """
    Perform inference on a single image.
    
    Args:
        model: image2code model instance
        image_path: Path to input image
        category: Object category for specialized processing
        return_raw_output: Whether to return raw model output
        
    Returns:
        JSON string or (json_string, raw_output) tuple
    """
    instruction = create_instruction(category)
    try:
        generated_text_list = model.generate(
            question=instruction,
            image_path=str(image_path),
            system_prompt=DETAILED_PROMPT
        )
        generated_text = generated_text_list[0] if generated_text_list else ""
        parsed_data = parse_generated_text(generated_text)
        json_result = json.dumps(parsed_data)
        if return_raw_output:
            return json_result, generated_text
        return json_result
    except Exception as e:
        print(f"Error during inference: {e}")
        empty_result = json.dumps({})
        if return_raw_output:
            return empty_result, str(e)
        return empty_result


def _inference_batch(
    model: QwenVLGenerationModel,
    image_paths: List[Union[str, Path]],
    category: str = "",
    return_raw_output: bool = True
) -> List[Union[str, Tuple[str, str]]]:
    """
    Perform batch inference on multiple images.
    
    Args:
        model: QwenVL model instance
        image_paths: List of paths to input images
        category: Object category for specialized processing
        return_raw_output: Whether to return raw model outputs
        
    Returns:
        List of results
    """
    if not image_paths:
        return []
    instruction = create_instruction(category)
    # Prepare batch items
    batch_items = [
        (instruction, str(image_path), DETAILED_PROMPT)
        for image_path in image_paths
    ]
    try:
        generated_text_list = model.generate_batch(batch_items)
        results = []
        for generated_text in generated_text_list:
            parsed_data = parse_generated_text(generated_text)
            json_result = json.dumps(parsed_data)
            if return_raw_output:
                results.append((json_result, generated_text))
            else:
                results.append(json_result)
        return results
    except Exception as e:
        print(f"Error during batch inference: {e}")
        empty_result = json.dumps({})
        error_result = (empty_result, str(e)) if return_raw_output else empty_result
        return [error_result] * len(image_paths)

def _inference_batch_baseline(
    model: QwenVLGenerationModel,
    image_paths: List[Union[str, Path]],
    category: str = "",
    return_raw_output: bool = True
) -> List[Union[str, Tuple[str, str]]]:
    """
    Perform batch inference using baseline prompts.
    
    Args:
        model: QwenVL model instance
        image_paths: List of paths to input images
        category: Object category for specialized processing
        return_raw_output: Whether to return raw model outputs
        
    Returns:
        List of results
    """
    if not image_paths:
        return []
    instruction = create_instruction(category)
    # Use baseline prompt for each item
    batch_items = [
        (instruction, str(image_path), baseline_prompt(category))
        for image_path in image_paths
    ]
    try:
        generated_text_list = model.generate_batch(batch_items)
        results = []
        for generated_text in generated_text_list:
            parsed_data = parse_generated_text(generated_text)
            json_result = json.dumps(parsed_data)
            if return_raw_output:
                results.append((json_result, generated_text))
            else:
                results.append(json_result)
        return results
    except Exception as e:
        print(f"Error during baseline batch inference: {e}")
        empty_result = json.dumps({})
        error_result = (empty_result, str(e)) if return_raw_output else empty_result
        return [error_result] * len(image_paths)

def inference(model: QwenVLGenerationModel, image_path: str, **kwargs) -> Union[str, Tuple[str, str]]:
    """Legacy single inference function."""
    return _inference_single(model, image_path, **kwargs)

def inference_batch(model: QwenVLGenerationModel, image_paths: List[str], **kwargs) -> List[Union[str, Tuple[str, str]]]:
    """Legacy batch inference function."""
    return _inference_batch(model, image_paths, **kwargs)
