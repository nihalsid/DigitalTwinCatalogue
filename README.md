# DTC Toolkit

This repository hosts an official toolkit for [DTC](https://www.projectaria.com/datasets/dtc/), a high quality dataset for object reconstruction research.

The toolkit offers:

- A sequence download tool that will download user selected sequences and their corresponding 3D object models.
- Three example python codes illustrating how to load and use DTC dataset, including a Rerun visualizer for Aria sequence, object mask generators for Aria and DSLR sequences.

Resources:
- [DTC homepage](https://www.projectaria.com/datasets/dtc/)
- [Download full DTC dataset from projectaria.com](https://www.projectaria.com/datasets/dtc/)
- [DTC Dataset Wiki](https://facebookresearch.github.io/projectaria_tools/docs/open_datasets/digital_twin_catalog)
- [Read DTC paper]()

The following are instructions to run the toolkit on DTC dataset.

## Step 1: Check System Requirement

The codebase is supported on:

* x64 Linux distributions of:
    * Fedora 36, 37, 38
    * Ubuntu jammy (22.04 LTS) and focal (20.04 LTS)
* Mac Intel or Mac ARM-based (M1) with MacOS 11 (Big Sur) or newer

Python 3.9+ (3.10+ if you are on [Apple Silicon](https://support.apple.com/en-us/116943)).


## Step 2: Setup Virtual Environment

Install Python library dependencies in a virtual environment.

```
# 1. Create virtual environment
rm -rf $HOME/dtc_tools_python_env
python3 -m venv $HOME/dtc_tools_python_env
source $HOME/dtc_tools_python_env/bin/activate

# 2. Install Python packages
python3 -m pip install --upgrade pip
pip install -r ${PATH_TO_DTC_REPO}/requirements.txt

```


## Step 3: Sign Up and Get the Download Links File

1. Review the [DTC license agreement](https://www.projectaria.com/datasets/dtc/license/).
    * Examine the specific licenses applicable to the data types you wish to use, such as Sequence and 3D object models.
2. Go to the [DTC website](https://www.projectaria.com/datasets/dtc/) and sign up.
    * Scroll down to the bottom of the page.
    * Enter your email and select **Access the Datasets**.
3. The DTC page will be refreshed to contain instructions and download links
    * The download view is ephemeral, keep the tab open to access instructions and links
    * Download links that last for 14 days
    * Enter your email again on the DTC main page to get fresh links
4. Select the Download button for any of the data types:
    * “Download the DTC Aria Dataset"
    * "Download the DTC 3D Object Model Dataset"
    * "Download the DTC DSLR Dataset"
5. These will swiftly download JSON files with urls that the downloader will use


## Step 4: Download the Data
### Use the [Project Aria Tool](https://www.projectaria.com/tools/) downloader to download some, or all of the data.

```
# 1. Activate your environment
source $HOME/dtc_tools_python_env/bin/activate

# 2. Setup DTC Aria data folder
mkdir -p $HOME/Documents/dtc_aria

# 3. Run the dataset downloader
# Download all DTC Aria sequence data
aria_dataset_downloader -c ${PATH_TO_ARIA_CDN_FILE} -o $HOME/Documents/dtc_aria/ -d 0 1 2 3 6
# Type answer `y`

# Download one DTC Aria sequence data
aria_dataset_downloader -c ${PATH_TO_ARIA_CDN_FILE} -o $HOME/Documents/dtc_aria/ -l BirdHouseRedRoofYellowWindows_active -d 0 1 2 3 6
# Type answer `y`

# 4. Setup DTC DSLR data folder
mkdir -p $HOME/Documents/dtc_dslr

# 5. Run the dataset downloader
# Download all DTC DSLR sequence data
aria_dataset_downloader -c ${PATH_TO_DSLR_CDN_FILE} -o $HOME/Documents/dtc_dslr/ -d 0 1
# Type answer `y`

# Download one DTC DSLR sequence data
aria_dataset_downloader -c ${PATH_TO_DSLR_CDN_FILE} -o $HOME/Documents/dtc_dslr/ -l Airplane_B097C7SHJH_WhiteBlue_Lighting001 -d 0 1
# Type answer `y`

# 6. Setup DTC 3D object model data folder
mkdir -p $HOME/Documents/dtc_model

# 3. Run the dataset downloader
# Download all DTC 3D object model data
dtc_object_downloader  -c {PATH_TO_OBJECT_CDN_FILE} -o $HOME/Documents/dtc_model
# Type answer `y`

# Download one DTC 3D object model data
dtc_object_downloader  -c {PATH_TO_OBJECT_CDN_FILE} -o $HOME/Documents/dtc_model -l Airplane_B097C7SHJH_WhiteBlue
# Type answer `y`
```

### Use the downloader in this repo to download some, or all of the data.

```
# 1. Activate your environment
source $HOME/dtc_tools_python_env/bin/activate

# 2. Setup DTC Aria and 3D object model data folder
mkdir -p $HOME/Documents/dtc_aria_model

# 3. Run the dataset downloader
# Download all DTC Aria sequence with corresponding 3D object model data
python3 ${PATH_TO_DTC_REPO}/download_model_with_sequence.py -s ${PATH_TO_ARIA_CDN_FILE} -m ${PATH_TO_OBJECT_CDN_FILE} -o $HOME/Documents/dtc_aria_model
# Type answer `y`

# Download one DTC Aria sequence with corresponding 3D object model data
python3 ${PATH_TO_DTC_REPO}/download_model_with_sequence.py -s ${PATH_TO_ARIA_CDN_FILE} -m ${PATH_TO_OBJECT_CDN_FILE} -o $HOME/Documents/dtc_aria_model -l BirdHouseRedRoofYellowWindows_active
# Type answer `y`

# 4. Setup DTC DSLR and 3D object model data folder
mkdir -p $HOME/Documents/dtc_dslr_model

# 5. Run the dataset downloader
# Download all DTC DSLR sequence with corresponding 3D object model data
python3 ${PATH_TO_DTC_REPO}/download_model_with_sequence.py -s ${PATH_TO_DSLR_CDN_FILE} -m ${PATH_TO_OBJECT_CDN_FILE} -o $HOME/Documents/dtc_dslr_model
# Type answer `y`

# Download one DTC DSLR sequence with corresponding 3D object model data
python3 ${PATH_TO_DTC_REPO}/download_model_with_sequence.py -s ${PATH_TO_DSLR_CDN_FILE} -m ${PATH_TO_OBJECT_CDN_FILE} -o $HOME/Documents/dtc_dslr_model -l BirdHouseRedRoofYellowWindows_active
# Type answer `y`
```


## Step 5: Run the Dataset Viewer

Viewing Aria rig trajectory, semi-dense point cloud and 3D object model

```
# 1. Activate your environment
source $HOME/dtc_tools_python_env/bin/activate

# 2. Run the dataset visualizer
python ${PATH_TO_DTC_REPO}/visualize_aria.py --sequence_folder $HOME/Documents/dtc_aria/{ARIA_SEQUENCE_NAME} --model_folder $HOME/Documents/dtc_model/{3D_MODEL_NAME}
```


## Step 6: Run the Mask Generation Tool

### Generate object mask for Aria sequence.

```
# 1. Activate your environment
source $HOME/dtc_tools_python_env/bin/activate

# 2. Setup DTC Aria mask folder
mkdir -p $HOME/Documents/dtc_aria_mask

# 3. Run the mask generation
python ${PATH_TO_DTC_REPO}/generate_aria_mask.py --sequence_folder $HOME/Documents/dtc_aria/{ARIA_SEQUENCE_NAME} --model_folder $HOME/Documents/dtc_model/{3D_MODEL_NAME} --output_folder $HOME/Documents/dtc_aria_mask --width 800 --height 800 --focal 400
```

### Generate object mask for DSLR sequence.

```
# 1. Activate your environment
source $HOME/dtc_tools_python_env/bin/activate

# 2. Setup DTC DSLR mask folder
mkdir -p $HOME/Documents/dtc_dslr_mask

# 3. Run the mask generation
python ${PATH_TO_DTC_REPO}/generate_dslr_mask.py --sequence_folder $HOME/Documents/dtc_aria/{ARIA_SEQUENCE_NAME} --model_folder $HOME/Documents/dtc_model/{3D_MODEL_NAME} --output_folder $HOME/Documents/dtc_dslr_mask
```


## License

- [DTC Toolkit](https://www.projectaria.com/datasets/dtc/) (aka. this repository) is released by Meta under the [Apache 2.0 license](LICENSE)
- [DTC dataset](https://www.projectaria.com/datasets/dtc/) is released under the [DTC license agreement](https://www.projectaria.com/datasets/dtc/license/)


## Contributing

Go to [Contributing](CONTRIBUTING.md) and the [Code of Conduct](CODE_OF_CONDUCT.md).
