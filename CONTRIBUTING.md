# Contributing to kgrun

Thank you for your interest in contributing to `kgrun`! We welcome contributions of all kinds: bug reports, documentation improvements, feature requests, and pull requests.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/xuan-tu-ai/kgrun.git
   cd kgrun
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install the package in editable mode with development dependencies:**
   ```bash
   pip install -e .
   pip install pytest pytest-cov
   ```

4. **Run the test suite:**
   ```bash
   pytest tests/
   ```

## Guidelines for Pull Requests

* **Keep It Focused:** A PR should address one specific issue or feature.
* **Code Style:** Follow PEP 8 guidelines. Keep line lengths under 100 characters where practical. Use clear type hints.
* **Preserve Platform Safety:** Do not introduce SSH tunnels, unofficial scrapers, or reverse proxies. `kgrun` strictly uses official APIs to keep user accounts safe.
* **Unit Tests:** Add tests for any new configuration parsing, file matching, or packaging behavior.
* **Documentation:** Update `README.md` if your change adds or modifies command-line flags or configuration options.

## Reporting Issues

When filing an issue, please include:
* Operating System (e.g. Windows 11, macOS Sequoia, Ubuntu 24.04).
* Python version (`python --version`) and Kaggle CLI version (`kaggle --version`).
* Relevant snippet of the log output or error traceback.
* Steps to reproduce.
