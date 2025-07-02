import gzip
import io
import json

import os
import pickle
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import trimesh
from google import genai
from google.genai import types
from PIL import Image


def jpg_encode(image, do_gray):
    image = Image.fromarray(image)
    if do_gray:
        image = image.convert("L")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def visualize_point_cloud(points, vis_path, colors=None) -> None:
    if colors is None:
        Path(vis_path).write_text(
            "\n".join(f"v {p[0]} {p[1]} {p[2]} 127 127 127" for p in points)
        )
    else:
        Path(vis_path).write_text(
            "\n".join(
                f"v {p[0]} {p[1]} {p[2]} {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}"
                for i, p in enumerate(points)
            )
        )


def project_point_to_image(points_world, camera_intrinsics, cam2world, W, H):
    K = camera_intrinsics[:3, :3]
    world2cam = np.linalg.inv(cam2world)
    points_camera = (world2cam[:3, :3] @ points_world.T).T + world2cam[:3, 3]
    points_image = (K @ points_camera.T).T
    points_image = points_image / points_image[:, 2:3]
    return points_image


def plot_dots(uv, W, H):
    # Initialize a blank image
    img = np.zeros((H, W), dtype=np.float32)

    # Create a 2D histogram of the points
    x = np.clip(uv[:, 0], 0, W - 1).astype(int)
    y = np.clip(uv[:, 1], 0, H - 1).astype(int)
    np.add.at(img, (y, x), 1)

    return (img * 255).astype(np.uint8)


def get_prompt():
    supported_cats = [
        "COUCH",
        "CHAIR",
        "TABLE",
        "SCREEN",
        "BED",
        "LAMP",
        "PLANT",
        "STORAGE",
        "MEDIA CONSOLE",
        "OTTOMAN",
        "REFRIGERATOR",
        "WHITEBOARD",
        "WALL ART",
        "ISLAND",
        "MIRROR",
        "LAPTOP",
        "CHAIR_TABLE",
        "SINK",
        "TOILET",
        "WASHERDRYER",
        "OTHER",
    ]
    supported_cats = [cat.lower() for cat in supported_cats]
    supported_cats_text = "{" + ", ".join(supported_cats) + "}"
    prompt = f"""You are an AI assistant that accurately describes objects within images. You will be given an image, where the image shows the object. Your task is to provide a single description of the object, using the following format:\n<category_limited>, <category_free>, <shape_description>\n\n with \n<category_limited>: Choose ONE of the following categories that best describes the object: {supported_cats_text}.\n<category_free>: Provide a short, descriptive label for the object using a maximum of TWO words. You are NOT limited to a predefined set of words for this category.\n<shape_description>: Describe the object's shape in one or two concise sentences. Focus ONLY on the shape, and do NOT include any information about the object's appearance (e.g., color, material, texture, or details).\nImportant: Your response must only contain the elements described above, separated by commas. Do NOT include any introductory phrases, unnecessary words, or filler text such as "The object within the red box is...", "I see...", or anything similar. Just the three comma-separated elements.\n\nExample Responses:\nchair, armchair, It is a four-legged structure with a flat seat and a curved back.\nNow, analyze the image provided and give your description in the specified format."""

    return prompt


def caption_using_gemini(image_bytes):
    model = "gemini-2.0-flash"
    client = genai.Client(
        api_key=os.environ.get("VLLM_API_TOKEN"),
    )
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(
                    mime_type="image/jpeg",
                    data=image_bytes,
                ),
                types.Part.from_text(text=get_prompt()),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
        response_mime_type="text/plain",
    )
    generated_content = []
    for chunk in client.models.generate_content_stream(
        model=model,
        # pyre-fixme[6]: For 2nd argument expected `Union[List[Union[List[Union[I...
        contents=contents,
        config=generate_content_config,
    ):
        generated_content.append(chunk.text)
    caption = "".join(generated_content)
    return caption


def main():
    folder = Path(sys.argv[1])
    sequence_name = sys.argv[2]

    # Load the point cloud
    pointcloud_gz = (
        folder
        / "sequences"
        / sequence_name
        / "mps"
        / "slam"
        / "semidense_points.csv.gz"
    )

    mesh = trimesh.load(
        folder / "mask" / sequence_name / "geometry.obj", force="mesh", process=False
    )
    shift = -(mesh.bounds[0] + mesh.bounds[1]) / 2
    mesh.vertices += shift

    with open(pointcloud_gz, "rb") as f:
        with gzip.open(f, "rt") as f1:
            data = StringIO(f1.read())
        pointcloud_pd = pd.read_csv(data)

    mask_folder = folder / "mask" / sequence_name / "mask"
    image_folder = folder / "mask" / sequence_name / "image"

    camera_extrinsics = []
    camera_intrinsics = []
    mask_images = []
    color_images = []
    for file in sorted(mask_folder.iterdir()):
        if file.suffix == ".json":
            loaded_dict = json.loads(open(file, "r").read())
            camera_extrinsics.append(
                np.array(loaded_dict["T_camera_world"], dtype=np.float32)
            )
            camera_intrinsics.append(
                np.array(loaded_dict["intrinsic"], dtype=np.float32)
            )
        elif file.suffix == ".png":
            mask_images.append(np.array(Image.open(file).convert("L")) > 128)
            color_images.append(
                np.array(Image.open(image_folder / file.name).convert("RGB"))
            )

    image_for_caption = mask_images[0][:, :, None] * color_images[0][:, :, :] + (
        1 - mask_images[0][:, :, None]
    ) * np.array([30, 30, 30])
    caption = caption_using_gemini(
        jpg_encode(image_for_caption.astype(np.uint8), do_gray=False)
    )
    print(caption)

    camera_extrinsics = np.stack(camera_extrinsics)
    inv_camera_extrinsics = np.linalg.inv(camera_extrinsics)
    inv_camera_extrinsics[:, :3, 3] = inv_camera_extrinsics[:, :3, 3] + shift
    camera_extrinsics = np.linalg.inv(inv_camera_extrinsics)

    camera_intrinsics = np.stack(camera_intrinsics)
    mask_images = np.stack(mask_images)
    color_images = np.stack(color_images)

    print("Filtering observations")
    points = pointcloud_pd[["px_world", "py_world", "pz_world"]].values + shift
    inv_std = pointcloud_pd["inv_dist_std"].values
    dist_std = pointcloud_pd["dist_std"].values

    stdevs_mask = np.logical_and(inv_std <= 0.005, dist_std <= 0.05)
    points = points[stdevs_mask]
    inv_std = inv_std[stdevs_mask]
    dist_std = dist_std[stdevs_mask]

    print("Visualizing pointcloud")
    visualize_point_cloud(points, "pointcloud.obj")

    print("Filtering point cloud by mask")
    # project the points to the image

    filter_mask = np.ones(len(points), dtype=bool)
    image_data = []

    for i in range(len(color_images)):
        color_images[i, :, :, :] = mask_images[i][:, :, None] * color_images[i][
            :, :, :
        ] + (1 - mask_images[i][:, :, None]) * np.array([30, 30, 30])
        image_data.append(jpg_encode(color_images[i], do_gray=True))

        point_image = project_point_to_image(
            points, camera_intrinsics[i], np.linalg.inv(camera_extrinsics[i]), 800, 800
        )
        point_mask = plot_dots(point_image[:, :2], 800, 800)

        # check the value of the mask at point image rounded to the nearest integer
        mask_i = mask_images[i]
        point_image_valid_z_mask = point_image[:, 2] > 0
        x = np.clip(point_image[:, 0], 0, mask_i.shape[1] - 1).astype(int)
        y = np.clip(point_image[:, 1], 0, mask_i.shape[0] - 1).astype(int)
        point_image_valid_uv_mask = mask_i[y, x]

        filter_mask = np.logical_and(
            filter_mask,
            np.logical_and(point_image_valid_z_mask, point_image_valid_uv_mask),
        )

        color_images[i, :, :, 0] = point_mask
        Image.fromarray(color_images[i]).save(f"dump/image_{i}.png")

    points = points[filter_mask]
    inv_std = inv_std[filter_mask]
    dist_std = dist_std[filter_mask]

    object_point_projections = []
    for i in range(len(color_images)):
        point_image = project_point_to_image(
            points, camera_intrinsics[i], np.linalg.inv(camera_extrinsics[i]), 800, 800
        )
        object_point_projections.append(torch.from_numpy(point_image[:, :2]).float())

    half_bounds = (np.max(points, axis=0) - np.min(points, axis=0)) / 2

    pkl_data = {
        "points_model": torch.from_numpy(points),
        "caption": caption,
        "inv_dist_std": torch.from_numpy(inv_std),
        "dist_std": torch.from_numpy(dist_std),
        "mesh_vertices": torch.from_numpy(mesh.vertices),
        "mesh_faces": torch.from_numpy(mesh.faces),
        "is_nebula": False,
        "bounds": torch.from_numpy(half_bounds),
        "image_data": image_data,
        "Ts_camera_model": torch.from_numpy(camera_extrinsics),
        "camera_params": torch.from_numpy(camera_intrinsics),
        "object_point_projections": object_point_projections,
        "is_dtc": True,
    }

    for key in pkl_data:
        if isinstance(pkl_data[key], torch.Tensor):
            print(f"{key}: {pkl_data[key].shape}")

    print("Visualizing filtered pointcloud")
    visualize_point_cloud(points, "filtered_pointcloud.obj")

    # save pkl file

    with open(f"{sequence_name}.pkl", "wb") as f:
        pickle.dump(pkl_data, f)


if __name__ == "__main__":
    main()
