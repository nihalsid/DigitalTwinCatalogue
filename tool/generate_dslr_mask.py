# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import multiprocessing as mp
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import trimesh


def load_mesh(path):
    scene = trimesh.load(path, force="mesh", process=False)

    mesh = {}
    mesh["vert"] = np.float32(scene.vertices)
    mesh["face"] = np.int32(scene.faces)

    return mesh


def project_mesh(intrinsic, rvec, tvec, mesh):
    dist_coeffs = np.zeros((5, 1), np.float32)

    projection, _ = cv2.projectPoints(mesh["vert"], rvec, tvec, intrinsic, dist_coeffs)

    mesh_projected = {}
    mesh_projected["projection"] = np.squeeze(projection).astype(int)

    return mesh_projected


def vis_projection(
    mesh,
    mesh_projected,
    img_in,
):
    for face in mesh["face"]:
        tri = np.asarray([mesh_projected["projection"][face[id]] for id in range(3)])

        cv2.drawContours(img_in, [tri], 0, (0, 0, 255), -1)

    return img_in


def combine_image(aria_img, mask_img):
    mask_value = mask_img[:, :, 0:3]
    mask_value[mask_img[:, :, 2] > 0] = 255
    mask_value[:, :, 0] = 0
    mask_value[:, :, 2] = 0

    combined_img = 0.7 * aria_img + 0.3 * mask_value

    return combined_img


def process_frame(camera, obj, mesh, img_folder, mask_folder, check_folder):
    T_camera_world = np.array(camera["T_camera_world"])
    T_world_object = np.array(obj["T_world_object"])
    T_camera_object = T_camera_world @ T_world_object

    intrinsic = np.zeros((3, 3), dtype=np.float32)

    intrinsic[0, 0] = float(camera["intrinsic"]["fx"])
    intrinsic[1, 1] = float(camera["intrinsic"]["fy"])
    intrinsic[0, 2] = float(camera["intrinsic"]["cx"])
    intrinsic[1, 2] = float(camera["intrinsic"]["cy"])
    intrinsic[2, 2] = 1.0

    rvec, _ = cv2.Rodrigues(T_camera_object[:3, :3])
    tvec = T_camera_object[:3, 3]

    mesh_projected = project_mesh(intrinsic, rvec, tvec, mesh)

    img_path = os.path.join(img_folder, camera["image"])

    dslr_img = cv2.imread(img_path)
    mask_img = dslr_img.copy() * 0

    mask_img = vis_projection(mesh, mesh_projected, mask_img)

    mask_path = os.path.join(mask_folder, camera["image"])

    cv2.imwrite(mask_path, mask_img)

    combined_img = combine_image(dslr_img, mask_img)

    combined_path = os.path.join(check_folder, camera["image"])

    cv2.imwrite(combined_path, combined_img)


def process_sequence(sequence_path, model_path, output_path):
    metadata_file = os.path.join(sequence_path, "metadata.json")

    with open(metadata_file) as f:
        metadata = json.load(f)

    model_name = metadata["model_name"][0]

    mesh_path = os.path.join(model_path, model_name, "3d-asset.glb")

    image_folder = os.path.join(sequence_path, "png")
    mask_folder = os.path.join(output_path, "mask")
    combined_folder = os.path.join(output_path, "overlay")

    os.makedirs(mask_folder, exist_ok=True)
    os.makedirs(combined_folder, exist_ok=True)

    camera_poses_file = os.path.join(sequence_path, "camera_poses.json")
    object_pose_file = os.path.join(sequence_path, "object_pose.json")

    with open(camera_poses_file) as f:
        camera_poses = json.load(f)

    with open(object_pose_file) as f:
        object_pose = json.load(f)

    mesh = load_mesh(mesh_path)

    cameras = camera_poses["cameras"]
    obj = object_pose["mesh"]

    pool = mp.Pool(int(args.process))

    for camera in cameras:
        pool.apply_async(
            process_frame,
            args=(
                camera,
                obj,
                mesh,
                image_folder,
                mask_folder,
                combined_folder,
            ),
        )

    pool.close()
    pool.join()


def main(args):
    process_sequence(args.sequence_folder, args.model_folder, args.output_folder)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--sequence_folder",
        type=str,
        help="dslr sequence folder",
        required=True,
    )
    parser.add_argument(
        "--model_folder",
        type=str,
        help="model root folder",
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="output folder",
        required=True,
    )
    parser.add_argument(
        "--process",
        type=int,
        help="number of processes to speed up",
        default=30,
    )

    args = parser.parse_args()

    main(args)
