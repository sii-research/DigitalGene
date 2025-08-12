import os
import sys
import math
import io
import re
import base64
import random
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Blender / rendering
import bpy
import mathutils

# Image encoding
from PIL import Image

# OpenAI-compatible client (DashScope base_url)
from openai import OpenAI

# ---------------------------
# Client Initialization
# ---------------------------
def get_client(api_key: str = None, base_url: str = None) -> OpenAI:
    """
    Initialize the OpenAI-compatible client. Uses DashScope's compatible mode by default.
    """
    key = api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("Missing API key. Set DASHSCOPE_API_KEY or pass --api-key.")
    url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    return OpenAI(api_key=key, base_url=url)

# ---------------------------
# Image utilities
# ---------------------------
def encode_image_to_base64(image_path: Path) -> str:
    """
    Load an image and return base64 PNG (RGB) string.
    """
    with Image.open(image_path) as img:
        if img.mode == "RGBA":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

# ---------------------------
# Geometry and normalization
# ---------------------------
def get_object_max_dimension_and_center(obj):
    """Compute max dimension and bounding-box center in world space."""
    if not obj:
        return 0.0, mathutils.Vector((0.0, 0.0, 0.0))

    world_matrix = obj.matrix_world
    if obj.type == 'MESH' and obj.data and obj.data.vertices:
        local_bbox_corners = [mathutils.Vector(c) for c in obj.bound_box]
        if not local_bbox_corners:
            dims = getattr(obj, "dimensions", mathutils.Vector((1.0, 1.0, 1.0)))
            return max(dims) if dims else 1.0, obj.location.copy()

        world_bbox = [world_matrix @ v for v in local_bbox_corners]
        min_x = min(v.x for v in world_bbox); max_x = max(v.x for v in world_bbox)
        min_y = min(v.y for v in world_bbox); max_y = max(v.y for v in world_bbox)
        min_z = min(v.z for v in world_bbox); max_z = max(v.z for v in world_bbox)
        center = mathutils.Vector(((min_x + max_x)/2, (min_y + max_y)/2, (min_z + max_z)/2))
        dims = mathutils.Vector((max_x - min_x, max_y - min_y, max_z - min_z))
        return max(dims), center
    else:
        return max(getattr(obj, "dimensions", mathutils.Vector((1.0, 1.0, 1.0)))), obj.location.copy()

def normalize_object_scale(obj, target_max_dimension=2.0, center_at_origin=True):
    """
    Scale object so its largest dimension equals target_max_dimension; optionally recenter to (0,0,0).
    """
    if not obj:
        print("normalize_object_scale: obj is None.")
        return

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    # Apply existing rotation/scale
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

    current_max, center = get_object_max_dimension_and_center(obj)
    if current_max <= 0:
        print(f"normalize_object_scale: zero dimension for {obj.name}")
        obj.select_set(False)
        return

    scale_factor = target_max_dimension / current_max
    obj.scale = (scale_factor, scale_factor, scale_factor)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    if center_at_origin:
        # Move origin to geometry bounds center, then move to origin
        bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
        obj.location = (0.0, 0.0, 0.0)
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

    obj.select_set(False)

# ---------------------------
# Scene helpers
# ---------------------------
def clear_scene():
    # Remove all objects
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    # Optionally purge orphans (may require user input in UI)
    # bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

def import_obj(filepath: Path):
    """
    Import an OBJ using Blender 4.x operator; fallback to older operator if needed.
    Returns the imported object (best-effort).
    """
    filepath = str(filepath)
    # Try new operator (Blender 4.x)
    try:
        bpy.ops.wm.obj_import(filepath=filepath)
        imported = bpy.context.selected_objects[-1] if bpy.context.selected_objects else None
        if imported:
            return imported
    except Exception:
        pass

    # Fallback (Blender 3.x)
    try:
        bpy.ops.import_scene.obj(filepath=filepath)
        imported = bpy.context.selected_objects[-1] if bpy.context.selected_objects else None
        if imported:
            return imported
    except Exception as e:
        print(f"OBJ import failed: {e}")

    # As a last resort, guess the last mesh added
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    return meshes[-1] if meshes else None

def apply_simple_material(obj):
    """
    Assign a simple Principled BSDF material with a fixed pastel base color (texture ignored).
    """
    if not obj or not obj.data:
        print("apply_simple_material: no mesh data")
        return
    mat = bpy.data.materials.new(name=f"Mat_{random.randint(0,9999)}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        # Fixed soft color to reduce any accidental bias
        bsdf.inputs["Base Color"].default_value = (0.50, 0.85, 0.95, 1.0)
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

def add_camera_and_light():
    # Camera
    bpy.ops.object.camera_add(location=(0.0, 0.0, 0.0))
    cam = bpy.context.object
    cam.data.lens_unit = 'FOV'
    cam.data.angle = math.radians(50.0)

    # Sun light
    light_data = bpy.data.lights.new(name="TopSunData", type='SUN')
    light_data.energy = 2.0
    light_obj = bpy.data.objects.new(name="TopSun", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.location = (0.0, 0.0, 10.0)
    light_obj.rotation_euler = (0.0, 0.0, 0.0)
    return cam, light_obj

def configure_render(engine: str, width: int, height: int):
    scene = bpy.context.scene
    engine = engine.upper()
    if engine not in {"CYCLES", "BLENDER_EEVEE", "BLENDER_WORKBENCH"}:
        engine = "CYCLES"
    scene.render.engine = engine
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.image_settings.file_format = 'PNG'
    scene.render.film_transparent = True

# ---------------------------
# Rendering from viewpoints
# ---------------------------
def render_object_views(obj_path: Path, out_dir: Path, image_size=(512, 512), engine="CYCLES", distance=5.0, z_offset=2.5, num_views=8):
    """
    Render OBJ from multiple viewpoints around the origin. Returns list of PNG paths.
    """
    clear_scene()
    imported = import_obj(obj_path)
    if not imported:
        raise RuntimeError(f"Failed to import OBJ: {obj_path}")

    normalize_object_scale(imported, target_max_dimension=2.0, center_at_origin=True)
    apply_simple_material(imported)
    cam, light = add_camera_and_light()
    configure_render(engine, image_size[0], image_size[1])

    # Define candidate viewpoints (on unit circle in XY plane); take up to num_views
    candidates = [
        ( 1,  0, 0), (-1,  0, 0), ( 0,  1, 0), ( 0, -1, 0),
        ( 1,  1, 0), (-1,  1, 0), ( 1, -1, 0), (-1, -1, 0),
        ( 0.5, 0.5, 0), (-0.5, 0.5, 0), (0.5, -0.5, 0), (-0.5, -0.5, 0),
    ]
    viewpoints = candidates[:max(1, int(num_views))]

    # Prepare output folder: out_dir/<obj_stem>/
    obj_stem = obj_path.stem
    obj_render_dir = out_dir / obj_stem
    obj_render_dir.mkdir(parents=True, exist_ok=True)

    scene = bpy.context.scene
    rendered = []
    render_idx = 0

    for i, (x, y, z) in enumerate(viewpoints):
        # Normalize direction
        length = math.sqrt(x*x + y*y + z*z)
        if length == 0:
            continue
        x, y, z = x/length, y/length, z/length

        cam_x = x * distance
        cam_y = y * distance
        cam_z = (z * distance) + (z_offset if z == 0 else z_offset)
        cam.location = (cam_x, cam_y, cam_z)
        bpy.context.view_layer.update()

        # Point camera at origin
        direction = mathutils.Vector((0.0, 0.0, 0.0)) - cam.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam.rotation_euler = rot_quat.to_euler()

        scene.camera = cam
        out_png = obj_render_dir / f"render_{render_idx:02d}.png"
        scene.render.filepath = str(out_png)
        try:
            bpy.ops.render.render(write_still=True)
            rendered.append(str(out_png))
            render_idx += 1
        except Exception as e:
            print(f"Render error at viewpoint {i}: {e}")

    # Clean up objects we created (keep data minimal between runs)
    try:
        if imported and imported.name in bpy.data.objects:
            bpy.data.objects.remove(imported, do_unlink=True)
        if cam and cam.name in bpy.data.objects:
            bpy.data.objects.remove(cam, do_unlink=True)
        # Remove all lights created
        for l in list(bpy.data.lights):
            bpy.data.lights.remove(l, do_unlink=True)
    except Exception as e:
        print(f"Cleanup warning: {e}")

    return rendered

# ---------------------------
# VLM request (single)
# ---------------------------
VLM_PROMPT = """
You are a 3D geometry comparison expert. I will provide you with a reference image and multiple rendered images. These rendered images are from different viewpoints of a single 3D model. Please compare the geometric shape of the object in the reference image with the 3D model depicted in the rendered images. Providing multiple rendered images is to help you understand the 3D information of the model from different viewpoints.
*Do not consider texture or color differences.* Focus *exclusively* on the 3D shape, proportions, and the presence, absence, and relative positioning of components.
First, describe your reasoning step-by-step. Analyze the similarities and differences you observe between the reference image and the rendered images of the 3D model. Consider:
* Overall shape and silhouette.
* Presence and relative position of major components.
* Proportions and sizes of components.
* Any noticeable distortions, exaggerations, or omissions.
* Specific features and details.
* The object in the reference image may rotate, but the object in the rendered images are assumed placed on the ground.
# If the reference image contains an obscured or incomplete object, or the object cannot be seen, simply return 1.0.
After your detailed reasoning, provide a single numerical score between 0.0 and 1.0, representing the geometric similarity. Use the following scale as a guide:
* **1.0:** Perfect geometric match.
* **0.9 - 0.99:** Near-perfect match with minor differences.
* **0.8 - 0.89:** Very good match; small noticeable differences.
* **0.7 - 0.79:** Good match; clear differences in several sub-components.
* **0.6 - 0.69:** Moderate match with significant differences.
* **0.5 - 0.59:** Fair match with major structural differences.
* **0.4 - 0.49:** Poor match.
* **0.3 - 0.39:** Very poor match.
* **0.2 - 0.29:** Extremely poor match.
* **0.0 - 0.19:** No discernible geometric similarity.
The first image is the reference image. The following images are the rendered views of the 3D model.
Your final answer MUST end with a line in the following format:
`FINAL SCORE: X.X`
"""

def vlm_single_request(client: OpenAI, reference_image: Path, render_paths, model_name: str, sample_k: int = 4):
    """
    One VLM call with the reference image + up to sample_k renders.
    Returns (score_float_or_None, raw_text_response).
    """
    if not render_paths:
        return None, "No rendered images."
    k = min(sample_k, len(render_paths))
    batch = random.sample(render_paths, k)

    msgs_content = [
        {"type": "text", "text": VLM_PROMPT},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{encode_image_to_base64(reference_image)}", "detail": "high"},
        },
    ]
    for p in batch:
        b64 = encode_image_to_base64(Path(p))
        msgs_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}})

    messages = [{"role": "user", "content": msgs_content}]

    resp = client.chat.completions.create(model=model_name, messages=messages, stream=False)
    text = resp.choices[0].message.content
    # Parse FINAL SCORE:
    m = re.search(r"FINAL SCORE:\s*([0-1](?:\.\d+)?)", text, flags=re.IGNORECASE)
    if m:
        try:
            return float(m.group(1)), text
        except ValueError:
            pass
    return None, text

# ---------------------------
# Parallel averaging
# ---------------------------
def average_vlm_score_parallel(client: OpenAI, reference_image: Path, render_paths, model_name: str, n_requests: int = 4):
    scores = []
    responses = []
    with ThreadPoolExecutor(max_workers=n_requests) as ex:
        futures = [ex.submit(vlm_single_request, client, reference_image, render_paths, model_name) for _ in range(n_requests)]
        for f in as_completed(futures):
            try:
                s, txt = f.result()
                responses.append(txt if txt else "No response")
                if s is not None:
                    scores.append(s)
            except Exception as e:
                responses.append(f"Exception: {e}")
    avg = sum(scores)/len(scores) if scores else None
    return avg, responses

# ---------------------------
# Evaluation entry point
# ---------------------------
def evaluate_single(image_path: Path,
                    obj_path: Path,
                    out_dir: Path,
                    model_name: str,
                    engine: str = "CYCLES",
                    width: int = 512,
                    height: int = 512,
                    n_views: int = 8,
                    n_requests: int = 4,
                    save_files: bool = True):
    """
    Renders views, calls VLM in parallel, returns (avg_score, responses, render_paths).
    Also saves CSV and JSON to out_dir if save_files=True.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Render object
    render_paths = render_object_views(obj_path, out_dir, image_size=(width, height), engine=engine, num_views=n_views)
    if not render_paths:
        print("Rendering failed or produced no images.")
        if save_files:
            # Write a short CSV noting failure
            csv_p = out_dir / f"similarity_score_{model_name}.csv"
            with csv_p.open("w", encoding="utf-8") as f:
                f.write("obj_name,vlm_model,similarity_score,status\n")
                f.write(f"{obj_path.stem},{model_name},,\n")
        return None, [], []

    # 2) VLM averaging
    client = get_client()
    avg_score, all_responses = average_vlm_score_parallel(client, image_path, render_paths, model_name, n_requests=n_requests)

    # 3) Save outputs
    if save_files:
        # CSV
        csv_p = out_dir / f"similarity_score_{model_name}.csv"
        with csv_p.open("w", encoding="utf-8") as f:
            f.write("obj_name,vlm_model,similarity_score,status\n")
            status = "Success" if avg_score is not None else "VLMFailed"
            f.write(f"{obj_path.stem},{model_name},{avg_score if avg_score is not None else ''},{status}\n")

        # JSON
        json_p = out_dir / f"vlm_response_{model_name}.json"
        with json_p.open("w", encoding="utf-8") as jf:
            json.dump({
                "obj_name": obj_path.stem,
                "vlm_model": model_name,
                "final_score": avg_score,
                "vlm_responses": all_responses,
                "renders": render_paths
            }, jf, indent=4, ensure_ascii=False)

    return avg_score, all_responses, render_paths

# ---------------------------
# CLI
# ---------------------------
def parse_args(argv):
    parser = argparse.ArgumentParser(description="Evaluate 3D similarity between an image and an OBJ using a VLM.")
    parser.add_argument("--image", required=True, type=str, help="Path to reference image (e.g., frame.png).")
    parser.add_argument("--obj", required=True, type=str, help="Path to predict OBJ file (e.g., Scan.obj).")
    parser.add_argument("--out", required=True, type=str, help="Output directory for renders and metrics.")
    parser.add_argument("--model", default="qwen-vl-max-2025-04-08", type=str, help="VLM model name.")
    parser.add_argument("--engine", default="CYCLES", type=str, choices=["CYCLES", "BLENDER_EEVEE", "BLENDER_WORKBENCH"], help="Blender render engine.")
    parser.add_argument("--size", nargs=2, default=[512, 512], type=int, help="Render size: width height")
    parser.add_argument("--views", default=8, type=int, help="Number of viewpoints to render (1..12).")
    parser.add_argument("--requests", default=4, type=int, help="Parallel VLM requests to average.")
    parser.add_argument("--api-key", default=None, type=str, help="API key (otherwise uses DASHSCOPE_API_KEY).")
    parser.add_argument("--base-url", default=None, type=str, help="Base URL for OpenAI-compatible endpoint.")
    return parser.parse_args(argv)

def main():
    # Check if running in Blender or standalone Python
    argv = sys.argv
    if "--" in argv:
        # Blender mode: read only args after "--"
        idx = argv.index("--")
        argv = argv[idx + 1:]
    else:
        # Standalone Python mode: use all args except script name
        argv = argv[1:] if len(argv) > 1 else []

    args = parse_args(argv)
    image_path = Path(args.image)
    obj_path = Path(args.obj)
    out_dir = Path(args.out)

    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        sys.exit(2)
    if not obj_path.exists():
        print(f"[ERROR] OBJ not found: {obj_path}")
        sys.exit(2)

    # Optionally override client globals (not strictly required here since we call get_client inside evaluate_single)
    if args.api_key:
        os.environ["DASHSCOPE_API_KEY"] = args.api_key
    if args.base_url:
        os.environ["DASHSCOPE_BASE_URL"] = args.base_url  # not used directly, but you can wire it if desired

    print("== 3D Similarity Evaluation ==")
    print(f"Image : {image_path}")
    print(f"OBJ   : {obj_path}")
    print(f"Out   : {out_dir}")
    print(f"Model : {args.model}")
    print(f"Engine: {args.engine}")
    print(f"Size  : {args.size[0]}x{args.size[1]}")
    print(f"Views : {args.views}")
    print(f"Requests: {args.requests}")

    score, responses, renders = evaluate_single(
        image_path=image_path,
        obj_path=obj_path,
        out_dir=out_dir,
        model_name=args.model,
        engine=args.engine,
        width=int(args.size[0]),
        height=int(args.size[1]),
        n_views=int(args.views),
        n_requests=int(args.requests),
        save_files=True
    )

    if score is not None:
        print(f"\nFINAL AVERAGE SCORE: {score:.4f}")
    else:
        print("\nFINAL AVERAGE SCORE: None (VLM failed or no parses)")
    print(f"Rendered views: {len(renders)}")
    print("Done.")

if __name__ == "__main__":
    main()
