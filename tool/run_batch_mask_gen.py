import os
import sys

from tqdm import tqdm


def run_pickle_gen(job):
    sequence_base_dir, sequence_base, model_folder, output_folder = job
    sequence = os.path.join(sequence_base_dir, sequence_base)
    cmd1 = f"python DigitalTwinCatalogue/tool/generate_aria_mask.py --sequence_folder {sequence} --model_folder {model_folder} --output_folder {output_folder} --width 800 --height 800 --focal 400"
    os.system(cmd1)
    cmd2 = (
        f"python DigitalTwinCatalogue/tool/convert_for_efm.py download {sequence_base}"
    )
    os.system(cmd2)

    if sequence_base.endswith("active"):
        pkl_loc = f"dtc_active/{sequence_base}.pkl"
        output_path = f"yawarnihal/tree/datasets/object_centric_reconstruction/dtc_active/{sequence_base}.pkl"
        os.system("manifold rm " + output_path)
        os.system("manifold put " + pkl_loc + " " + output_path)
    elif sequence_base.endswith("passive"):
        pkl_loc = f"dtc_passive/{sequence_base}.pkl"
        output_path = f"yawarnihal/tree/datasets/object_centric_reconstruction/dtc_passive/{sequence_base}.pkl"
        os.system("manifold rm " + output_path)
        os.system("manifold put " + pkl_loc + " " + output_path)


if __name__ == "__main__":
    sequence_base_dir = sys.argv[1]
    all_sequences = [
        x
        for x in os.listdir(sequence_base_dir)
        if os.path.isdir(os.path.join(sequence_base_dir, x))
    ]
    model_folder = sys.argv[2]
    output_folder = sys.argv[3]
    jobs = []

    for sequence_base in all_sequences:
        jobs.append((sequence_base_dir, sequence_base, model_folder, output_folder))

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(run_pickle_gen, jobs), total=len(jobs)))
