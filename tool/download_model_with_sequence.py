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


def download_sequences(sequence_cdn_file, sequence_output_folder, sequence_names):
    sequence_download_cmd = (
        "aria_dataset_downloader -c "
        + sequence_cdn_file
        + " -o "
        + sequence_output_folder
    )

    if sequence_names is not None and len(sequence_names) > 0:
        sequence_download_cmd += " -l " + " ".join(sequence_names)

    os.system(sequence_download_cmd)


def download_models(model_cdn_file, model_output_folder, model_names):
    model_download_cmd = (
        "dtc_object_downloader -c " + model_cdn_file + " -o " + model_output_folder
    )

    if len(model_names) > 0:
        model_download_cmd += " -l " + " ".join(model_names)

    os.system(model_download_cmd)


def get_model_names(sequence_output_folder):
    model_names = set()

    for sequence_folder in os.listdir(sequence_output_folder):
        metadata_file = os.path.join(
            sequence_output_folder, sequence_folder, "metadata.json"
        )
        if not os.path.exists(metadata_file):
            continue
        with open(metadata_file) as f:
            metadata = json.load(f)

        model_names.add(metadata["model_name"][0])

    return model_names


def main(args):
    sequence_output_folder = os.path.join(args.output_folder, "sequences")
    os.makedirs(sequence_output_folder, exist_ok=True)

    model_output_folder = os.path.join(args.output_folder, "models")
    os.makedirs(model_output_folder, exist_ok=True)

    download_sequences(
        args.sequence_cdn_file,
        sequence_output_folder,
        args.sequence_names,
    )

    model_names = get_model_names(sequence_output_folder)

    download_models(args.model_cdn_file, model_output_folder, model_names)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-s",
        "--sequence_cdn_file",
        type=str,
        required=True,
        help="path to sequence cdn file",
    )
    parser.add_argument(
        "-m",
        "--model_cdn_file",
        type=str,
        required=True,
        help="path to model cdn file",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        required=True,
        help="path to output folder",
    )
    parser.add_argument(
        "-l",
        "--sequence_names",
        nargs="+",
        required=False,
        help="a list of sequence names (separated by space) to be downloaded. If not set, all sequences will be downloaded",
    )

    args = parser.parse_args()

    main(args)
