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
import os
from argparse import ArgumentParser
from pathlib import Path

import cv2
import imageio
import numpy as np
import trimesh
from projectaria_tools.core import calibration, data_provider, mps
from projectaria_tools.core.mps import MpsDataPathsProvider
from projectaria_tools.core.mps.utils import get_nearest_pose
from projectaria_tools.core.sensor_data import TimeDomain
from projectaria_tools.core.stream_id import StreamId
from tqdm import tqdm


def save_rgb_frame(data, aria_camera, pinhole_camera, output_folder, frame_id):
    distorted_image = data.image_data_and_record()[0].to_numpy_array()

    undistorted_image = calibration.distort_by_calibration(
        distorted_image, pinhole_camera, aria_camera
    )

    output_file_name = "aria_{id:0=4}.png"
    output_path = os.path.join(output_folder, output_file_name.format(id=frame_id))
    imageio.imwrite(output_path, undistorted_image)

    return undistorted_image


def load_mesh(path):
    scene = trimesh.load(path, force="mesh", process=False)

    mesh = {}
    mesh["vert"] = np.float32(scene.vertices)
    mesh["face"] = np.int32(scene.faces)
    mesh["center"] = np.mean(mesh["vert"], axis=0)

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
    w, h, _ = img_in.shape

    tri_list = np.asarray(
        [
            [mesh_projected["projection"][face[id]] for id in range(3)]
            for face in mesh["face"]
        ]
    )

    skip_check = (
        np.sum(
            np.logical_or(
                np.logical_or(tri_list[:, :, 0] < 0, tri_list[:, :, 0] >= w),
                np.logical_or(tri_list[:, :, 1] < 0, tri_list[:, :, 1] >= h),
            ),
            axis=1,
        )
        == 3
    )

    for id in range(len(tri_list)):
        if skip_check[id]:
            continue

        cv2.drawContours(img_in, [tri_list[id]], 0, (255, 255, 255), -1)

    return img_in


def save_mask(
    mesh,
    pose_info,
    rgb_camera_calib,
    T_world_object,
    width,
    height,
    focal,
    output_folder,
    frame_id,
    device_time_ns,
):
    T_world_device = pose_info.transform_world_device
    T_device_camera = rgb_camera_calib.get_transform_device_camera()
    T_camera_world = (T_world_device @ T_device_camera).inverse().to_matrix()
    T_camera_object = T_camera_world @ T_world_object

    mask_img = np.zeros((height, width, 3), dtype=np.uint8)

    mesh_center_in_camera = (
        T_camera_object[:3, :3] @ mesh["center"] + T_camera_object[:3, 3]
    )

    if mesh_center_in_camera[2] < 0:
        return mask_img

    rvec, _ = cv2.Rodrigues(T_camera_object[:3, :3])
    tvec = T_camera_object[:3, 3]

    intrinsic = np.zeros((3, 3), dtype=np.float32)

    intrinsic[0, 0] = float(focal)
    intrinsic[1, 1] = float(focal)
    intrinsic[0, 2] = float(width * 0.5)
    intrinsic[1, 2] = float(height * 0.5)
    intrinsic[2, 2] = 1.0

    mesh_projected = project_mesh(intrinsic, rvec, tvec, mesh)
    mask_img = vis_projection(mesh, mesh_projected, mask_img)

    output_file_name = "aria_{id:0=4}.png"
    output_path = os.path.join(output_folder, output_file_name.format(id=frame_id))
    imageio.imwrite(output_path, mask_img)

    output_file_name = "transform_{id:0=4}.json"
    output_path = os.path.join(output_folder, output_file_name.format(id=frame_id))
    Path(output_path).write_text(
        json.dumps(
            {
                "T_camera_world": T_camera_world.tolist(),
                "intrinsic": intrinsic.tolist(),
                "device_time_ns": device_time_ns,
            }
        )
    )

    return mask_img


def save_combined_image(rgb_img, mask_img, output_folder, frame_id):
    mask_value = mask_img[:, :, 0:3]
    mask_value[mask_img[:, :, 2] > 0] = 255
    mask_value[:, :, 0] = 0
    mask_value[:, :, 2] = 0

    combined_img = 0.7 * rgb_img + 0.3 * mask_value
    combined_img = combined_img.astype(np.uint8)

    output_file_name = "aria_{id:0=4}.png"
    output_path = os.path.join(output_folder, output_file_name.format(id=frame_id))
    imageio.imwrite(output_path, combined_img)


def process_sequence(
    mps_folder,
    vrs_file,
    mesh_file,
    object_pose_file,
    image_folder,
    mask_folder,
    combined_folder,
    width,
    height,
    focal,
):
    mesh = load_mesh(mesh_file)

    with open(object_pose_file) as f:
        object_pose = json.load(f)
    T_world_object = np.array(object_pose["mesh"]["T_world_object"])

    tmesh = trimesh.load(mesh_file, force="mesh", process=False)
    tmesh.apply_transform(T_world_object)
    trimesh.Trimesh(vertices=tmesh.vertices, faces=tmesh.faces).export(
        Path(combined_folder).parent / "geometry.obj"
    )

    mps_data_paths_provider = MpsDataPathsProvider(mps_folder)
    mps_data_paths = mps_data_paths_provider.get_data_paths()

    trajectory_data = mps.read_closed_loop_trajectory(
        mps_data_paths.slam.closed_loop_trajectory
    )

    online_calibs = mps.read_online_calibration(mps_data_paths.slam.online_calibrations)
    for camera_calib in online_calibs[-1].camera_calibs:
        if camera_calib.get_label() == "camera-rgb":
            last_rgb_camera_calib = camera_calib

    provider = data_provider.create_vrs_data_provider(vrs_file)
    # provider.set_color_correction(True)
    rgb_stream_label = "camera-rgb"
    rgb_stream_id = StreamId("214-1")
    deliver_option = provider.get_default_deliver_queued_options()
    deliver_option.deactivate_stream_all()
    deliver_option.activate_stream(rgb_stream_id)
    deliver_option.set_subsample_rate(rgb_stream_id, args.subsample_rate)
    rgb_frame_count = provider.get_num_data(rgb_stream_id)

    aria_camera = provider.get_device_calibration().get_camera_calib(rgb_stream_label)
    pinhole_camera = calibration.get_linear_camera_calibration(width, height, focal)

    progress_bar = tqdm(total=rgb_frame_count // args.subsample_rate)
    frame_id = 0

    for data in provider.deliver_queued_sensor_data(deliver_option):
        progress_bar.update(1)

        device_time_ns = data.get_time_ns(TimeDomain.DEVICE_TIME) - int(
            last_rgb_camera_calib.get_time_offset_sec_device_camera() * 1e9
        )
        pose_info = get_nearest_pose(trajectory_data, device_time_ns)

        if pose_info:
            rgb_img = save_rgb_frame(
                data, aria_camera, pinhole_camera, image_folder, frame_id
            )
            mask_img = save_mask(
                mesh,
                pose_info,
                last_rgb_camera_calib,
                T_world_object,
                width,
                height,
                focal,
                mask_folder,
                frame_id,
                device_time_ns,
            )
            save_combined_image(rgb_img, mask_img, combined_folder, frame_id)

            frame_id += 1


def main(args):
    metadata_file = os.path.join(args.sequence_folder, "metadata.json")
    with open(metadata_file) as f:
        metadata = json.load(f)

    model_name = metadata["model_name"][0]
    sequence_name = Path(args.sequence_folder).name
    object_pose_file = os.path.join(args.sequence_folder, "object_pose.json")
    mesh_file = os.path.join(args.model_folder, model_name, "3d-asset.glb")
    vrs_file = os.path.join(args.sequence_folder, "video.vrs")
    mps_folder = os.path.join(args.sequence_folder, "mps")

    mask_folder = os.path.join(args.output_folder, sequence_name, "mask")
    image_folder = os.path.join(args.output_folder, sequence_name, "image")
    combined_folder = os.path.join(args.output_folder, sequence_name, "overlay")

    os.makedirs(mask_folder, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(combined_folder, exist_ok=True)

    process_sequence(
        mps_folder,
        vrs_file,
        mesh_file,
        object_pose_file,
        image_folder,
        mask_folder,
        combined_folder,
        args.width,
        args.height,
        args.focal,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--sequence_folder",
        type=str,
        help="path to sequence folder",
    )
    parser.add_argument(
        "--model_folder",
        type=str,
        help="path to model root folder",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="path to output folder",
    )
    parser.add_argument(
        "--width",
        type=int,
        help="image width",
        required=True,
    )
    parser.add_argument(
        "--height",
        type=int,
        help="image height",
        required=True,
    )
    parser.add_argument(
        "--focal",
        type=int,
        help="focal length",
        required=True,
    )
    parser.add_argument(
        "--subsample_rate",
        type=int,
        help="after a data is played, rate - 1 data are skipped",
        default=1,
    )

    args = parser.parse_args()

    main(args)
