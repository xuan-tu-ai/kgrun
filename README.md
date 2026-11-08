<div align="center">

# kgrun

### Run Python scripts on remote Kaggle GPUs directly from your local IDE.

[![CI](https://github.com/xuan-tu-ai/kgrun/actions/workflows/ci.yml/badge.svg)](https://github.com/xuan-tu-ai/kgrun/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Kaggle API](https://img.shields.io/badge/API-Official%20Kaggle-20BEFF.svg)](https://github.com/Kaggle/kaggle-api)

<p align="center">
  <b>No GPU on your local laptop? No problem.</b><br>
  Write code in VS Code, Cursor, or PyCharm as usual, then run it on free Kaggle GPUs (2× Tesla T4) with a single CLI command.<br>
  Zero SSH setup · Zero tunnel risks · 100% Official Kaggle API.
</p>

</div>

---

## Why kgrun?

Developing deep learning models on a laptop or thin client without an NVIDIA GPU is frustrating:

| Approach | Drawbacks & Friction |
|---|---|
| **Kaggle Web Notebooks** | Clunky in-browser editor, tedious multi-file navigation, disconnected from local Git and IDE shortcuts. |
| **Google Colab Free** | Random disconnections, aggressive idle timeouts, unpredictable GPU assignments (often CPU or K80). |
| **SSH / Reverse Tunnels** (ngrok, cloudflared) | **Violates Kaggle Terms of Service**, high risk of account suspension or permanent bans. |
| **`kgrun` (This tool)** | **Best of both worlds:** Code locally in your favorite IDE, run on Kaggle's free 2× Tesla T4 GPUs via official APIs, stream SSE logs live, and auto-download checkpoints. |

---

## Features

* **Zero Server Provisioning:** No VPS, no cloud credit card, no SSH keys. Uses your existing Kaggle account.
* **Automatic Multi-File Packaging:** Bundles local modules (`src/**`, `configs/**`, helper scripts) and injects them onto the remote host automatically.
* **Live SSE Log Streaming:** Real-time stdout and stderr streamed line-by-line to your terminal with `kaggle kernels logs -f`.
* **Detachable & Resilient:** Press `Ctrl+C` to detach at any time. Your job continues running on Kaggle uninterrupted.
* **Dataset Management:** Upload local datasets with `kgrun data push DIR` and automatically mount them in remote sessions.
* **Auto-Pull Artifacts:** Download weights, logs, and metric CSVs directly into `runs/<timestamp>/` with `--pull`.
* **Reproducibility Stamp:** Generates a complete snapshot (`kgrun_stamp.json`) logging CUDA hardware, PyTorch/driver versions, git/code hash, and CLI arguments.

---

## Architecture

```
 local machine (vs code / terminal)                 remote kaggle cloud
 ───────────────────────────────────                 ───────────────────
  1. kgrun run train.py --pull
     ├─ Read kgrun.toml
     ├─ Bundle source (tar.gz ≤ 750 KB)
     └─ Base64 inject into kgrun_entry.py
                                       ───────►  kaggle kernels push
                                                 (accelerator: 2x Tesla T4)
                                                 │
                                                 ▼
  2. Stream live SSE logs             ◄───────  [Kaggle GPU Container]
     (Ctrl+C detaches safely)                    ├─ Extract to /kaggle/tmp/src
                                                 ├─ Mount /kaggle/input/<dataset>
                                                 ├─ Export KGRUN_DATA & KGRUN_OUT
                                                 ├─ Record kgrun_stamp.json
                                                 └─ Execute: python -u train.py
                                                 │
  3. Automatic Artifact Download                 ▼
     runs/2026-11-08_112549/          ◄───────  /kaggle/working/
     ├─ best_model.pth (44.8 MB)                ├─ best_model.pth
     ├─ metrics.csv                             ├─ metrics.csv
     └─ kgrun_stamp.json                        └─ kgrun_stamp.json
```

---

## 30-Second Quickstart

### 1. Prerequisites
* **Python 3.11+** installed.
* **Kaggle Account:** Phone-verified (required by Kaggle for free GPU access).
* **Kaggle API Key:** Download `kaggle.json` or `credentials.json` from [Kaggle Settings](https://www.kaggle.com/settings) and save it to `~/.kaggle/`.

### 2. Install
```bash
git clone https://github.com/xuan-tu-ai/kgrun.git
cd kgrun
pip install -e .
```

### 3. Initialize Your Project
Navigate to any project directory where you want to train:
```bash
cd /path/to/my-project
kgrun init
```
This detects your Kaggle username and generates `kgrun.toml`.

### 4. Make Your Code Compatible
Ensure your script uses `KGRUN_OUT` for artifacts and `KGRUN_DATA` for inputs:
```python
import os
from pathlib import Path

# Outputs: local fallback to 'outputs', Kaggle mounts to '/kaggle/working'
out_dir = Path(os.environ.get("KGRUN_OUT", "outputs"))
out_dir.mkdir(parents=True, exist_ok=True)

# Data: local fallback to 'data', Kaggle mounts to '/kaggle/input/<dataset>'
data_dir = Path(os.environ.get("KGRUN_DATA", "data"))

# Save model checkpoint
# torch.save(model.state_dict(), out_dir / "best_model.pth")
```

### 5. Launch & Pull Results!
```bash
kgrun run train.py --epochs 30 --batch-size 64 --pull
```

---

## CLI Reference

| Command | Description |
|---|---|
| `kgrun init [--force]` | Initialize `kgrun.toml` in current project folder |
| `kgrun run SCRIPT [ARGS] [--gpu t4\|none] [--pull]` | Package project, push to Kaggle GPU, stream logs |
| `kgrun pull [--pattern "*.pth"]` | Download outputs from the latest run |
| `kgrun logs [-f]` | View or stream remote execution logs |
| `kgrun status` | Check status (`RUNNING`, `COMPLETE`, `ERROR`) |
| `kgrun data push DIR [-m msg] [--slug slug]` | Upload local folder as a Kaggle Dataset |

### Advanced `run` Flags:
* `--gpu t4`: Accelerates on **2× Tesla T4** (default). Use `--gpu none` for standard CPU.
* `--pull`: Automatically downloads output files into `runs/<timestamp>/` upon completion.
* `--no-follow`: Detaches immediately after pushing.
* `--timeout <hours>`: Safety fuse in hours (default: `11h`, maximum Kaggle session: `12h`).

---

## Configuration (`kgrun.toml`)

`kgrun` is project-scoped. Everything is configured in `kgrun.toml` at your project root:

```toml
# Remote kernel identifier on Kaggle
kernel      = "username/my-awesome-model"

# Accelerator: "NvidiaTeslaT4" (2x Tesla T4) or "none" (CPU)
accelerator = "NvidiaTeslaT4"

# Execution timeout in hours (< 12)
timeout_h   = 11

# Outbound internet access (for pip install or downloading pretrained weights)
internet    = true

# Mounted Kaggle dataset slugs
datasets    = ["username/my-dataset"]

# Glob patterns to bundle into the code archive
include     = ["*.py", "src/**", "configs/**", "requirements*.txt"]

# Patterns to strictly exclude from packaging (data belongs in Kaggle Datasets)
exclude     = ["venv/**", ".venv/**", "data/**", "runs/**", "**/__pycache__/**", ".git/**"]

# Optional dependencies to install on container startup
# pip       = "requirements-kaggle.txt"
```

---

## Technical Limits & Best Practices

* **Inline Script Size (1 MiB Limit):** The Kaggle API enforces a strict **1 MiB (1,048,576 bytes)** limit on code uploads. `kgrun` automatically enforces a **750 KB** threshold on the compressed `tar.gz` archive to guarantee the base64-embedded payload stays under the limit. Keep data and weights out of `include`!
* **Container Boot Time:** Kaggle virtual machine provisioning takes approximately 1 to 2 minutes before the script starts. Test on 1 batch locally before pushing long jobs.
* **Weekly Quotas:** Kaggle accounts receive ~30 hours of free GPU time per week (resets weekly). Keep track of your quota in your Kaggle profile.

---

## Repository Structure

```
kgrun/
├── kgrun/                     # Core CLI implementation
│   ├── cli.py                 # Argument parsing and subcommands
│   ├── config.py              # kgrun.toml parser and validator
│   ├── pack.py                # In-memory tar.gz bundler & size checker
│   ├── bootstrap.py           # Remote Kaggle container bootstrap generator
│   ├── runner.py              # Execution orchestrator & log streamer
│   └── data.py                # Dataset creation and versioning
├── examples/                  # Working reference examples
│   ├── sample_project/        # Multi-file import demonstration
│   ├── train_fish.py          # Real-world PyTorch deep learning training
│   └── check_dataset.py       # Dataset mounting verification
├── tests/                     # Unit test suite (pytest)
├── .github/workflows/ci.yml   # Multi-platform CI pipeline
├── pyproject.toml             # Standard Python packaging metadata
├── CONTRIBUTING.md            # Guidelines for open-source contributors
└── LICENSE                    # MIT License
```

---

## Contributing

Contributions make the open-source community an amazing place to learn, inspire, and create! Any contributions you make are **greatly appreciated**.

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details on setting up your local environment and submitting pull requests.

---

## License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for more information.

---

<div align="center">
  <b>Built with ❤️ for machine learning researchers and students worldwide.</b><br>
  If you find kgrun helpful, please consider giving it a ⭐ on GitHub!
</div>
