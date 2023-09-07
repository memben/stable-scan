# Stable-Scan: Retexture 3D Point Clouds Using Stable Diffusion

![Stable-Scan Demo](https://github.com/memben/stable-scan/assets/59774249/e75e2dca-be58-44ab-b069-403f8d8585ba)

Stable-Scan leverages the power of Stable Diffusion to enrich your 3D Point Clouds with new textures.

## Table of Contents

- [Installation](#installation)
  - [Clone Repository](#clone-repository)
  - [Environment Setup](#environment-setup)
  - [LAS File Creation](#las-file-creation)
  - [Stable Diffusion API Setup](#stable-diffusion-api-setup)
- [Usage](#usage)
- [Known Issues](#known-issues)

## Installation

### Clone Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/memben/stable-scan.git
```

### Environment Setup

Set up a virtual environment and install required packages. (Note: Detailed instructions are coming soon.)

```bash
# TODO: Instructions for setting up the .venv and installing requirements
```

### LAS File Creation

The application currently supports only .las files for 3D Point Clouds. You can create .las files using various applications. One such application is [ScanKit](https://github.com/Kenneth-Schroeder/ScanKit), 
also available in the [App Store](https://apps.apple.com/lu/app/scankit/id1581317177).

#### Note
For larger scans, ScanKit will produce multiple .las files. 

```bash
python stablescan.py debug path/to/file1.las path/to/file2.las
```

### Stable Diffusion API Setup

Stable-Scan relies on the Stable Diffusion WebUI API. To set it up:

1. Follow the instructions at the [Stable Diffusion WebUI GitHub repo](https://github.com/AUTOMATIC1111/stable-diffusion-webui.git).
2. Install [sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) as an extension.
3. Download the depth model (`control_v11f1p_sd15_depth [cfd03158]`, to use newer versions, update [webui_api.py](https://github.com/memben/stable-scan/blob/edd5e6998a923bb427184431688240a0b28b6669/webui_api.py#L68)).
4. Start the server:

```bash
./webui.sh --api
```

If you're running the server remotely, use the `--share` flag.

## Usage

Activate your virtual environment and navigate to the project directory.

For CLI options:

```bash
python stablescan.py
```

For a quick demo:

```bash
python stablescan.py control path/to/file.las
```

For full control (recommended), use the `debug` mode:

```bash
python stablescan.py debug path/to/file.las --webui-api SERVER_URL
```

**Workflow:**

1. Navigate to the view you wish to retexture.
2. Press `r`, validate the preview images and if statisfied continue by giving a prompt in CLI.
3. Press `o` to save the state.
4. Optional: Remove untextured points by pressing `x`.
5. Load the saved state by pressing `l`.
6. Navigate to new positions and retexture them.

## Known Issues

- Projection into the point cloud has some known limitations ([See Issue](#TODO-LINK)).
- Style transfer is currently problematic ([See Issue](#TODO-LINK)).

For more information and updates, visit the [Issues section](#TODO-LINK).

We hope you find Stable-Scan useful for your 3D Point Cloud texturing needs! Feel free to contribute and report issues.
