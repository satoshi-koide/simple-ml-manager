# ML Manager

A lightweight, local-first experiment tracker for machine learning projects.

This tool integrates with Weights & Biases (`wandb`) for live metric logging while saving all experiment configurations and final metrics locally as `.toml` files.

This gives you the immediate visualization of `wandb` combined with the portability and queryability of a local, file-based database. The `MLProject` class can scan all experiment directories and load them into a `pandas` DataFrame for powerful, local analysis.

## Features

* 🚀 **Start New Runs**: `MLRun.create()` automatically initializes `wandb` and creates a local directory for your run at `base_dir/project_name/run_id`.
* 💾 **Local-First Storage**: Saves experiment configs to `config.toml` and final metrics to `metrics.toml` for every run.
* 📊 **Dual Logging**: `run.add_metrics()` logs to both `wandb` (live) and the local `metrics.toml` (persistence) with a single call.
* 🔍 **Pandas-Powered Analysis**: `MLProject` loads your entire experiment history from local files into a `pandas` DataFrame.
* 🏃‍♂️ **Powerful Querying**: Search and filter experiments locally using pandas query strings (e.g., `` `accuracy` > 0.9 and `model.name` == 'BERT' ``).
* 🔄 **Easy Reloading**: Load any `MLRun` object from its `run_id` to inspect its configuration and metrics.
* 🔖 **Git Commit Tracking**: Automatically records the current git commit hash for reproducibility (if the working directory is in a git repository).

---

## Installation

This project uses a `src` layout. To install it in your local environment in editable mode, clone the repository and run:

```bash
pip install -e .
```

You will also need the core dependencies:

```bash
pip install wandb pandas toml
```

---

## Quick Start

Here is a complete end-to-end example of running two experiments and then analyzing the results.

```python
import os
from ml_manager import MLRun, MLProject

# --- Configuration ---
# !! Change this to your wandb entity !!
WANDB_ENTITY = "your_wandb_entity"
WANDB_PROJECT = "my-demo-project"
BASE_CHECKPOINT_DIR = "./checkpoints"

# --- 1. Run Experiment 1 (CNN) ---
print("--- Running Experiment 1: CNN ---")
config1 = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "model": {"name": "CNN", "layers": 5}
}

# This calls wandb.init() and creates the local dir
run1 = MLRun.create(
    config=config1,
    base_dir=BASE_CHECKPOINT_DIR,
    project_name=WANDB_PROJECT,
    entity=WANDB_ENTITY
)

# ... (your model training loop) ...
print(f"Training CNN... (Run ID: {run1.run_id})")

# Log final metrics. This saves to metrics.toml and calls wandb.log()
final_metrics1 = {"accuracy": 0.92, "f1_score": 0.91, "epoch": 10}
run1.add_metrics(final_metrics1)

# This calls wandb.finish()
run1.finish()
print(f"Find this run at: {run1.get_wandb_url()}")


# --- 2. Run Experiment 2 (BERT) ---
print("\n--- Running Experiment 2: BERT ---")
config2 = {
    "learning_rate": 5e-5,
    "batch_size": 16,
    "model": {"name": "BERT", "layers": 12}
}

run2 = MLRun.create(
    config=config2,
    base_dir=BASE_CHECKPOINT_DIR,
    project_name=WANDB_PROJECT,
    entity=WANDB_ENTITY
)

# ... (your model training loop) ...
print(f"Training BERT... (Run ID: {run2.run_id})")

final_metrics2 = {"accuracy": 0.96, "f1_score": 0.95, "epoch": 5}
run2.add_metrics(final_metrics2)
run2.finish()
print(f"Find this run at: {run2.get_wandb_url()}")


# --- 3. Analyze All Experiments ---
print("\n--- Analyzing Project ---")

# This scans ./checkpoints/{project_name}/{run_id} and loads all config.toml/metrics.toml files
project = MLProject(base_dir=BASE_CHECKPOINT_DIR)

print("\n[Full Experiment DataFrame]")
print(project.df.to_markdown(index=False))

# Search the DataFrame using pandas query syntax
# Note: Use backticks (`) for nested keys like 'model.name'
print("\n[Searching for 'accuracy > 0.95']")
best_runs_df = project.search(query_string="accuracy > 0.95")

print(best_runs_df[['run_id', 'model.name', 'accuracy', 'f1_score']].to_markdown(index=False))

# Load a specific run object from the search result
if not best_runs_df.empty:
    best_run_id = best_runs_df.iloc[0]["run_id"]
    print(f"\n[Loading best run: {best_run_id}]")
    
    best_run: MLRun = project.get_run(best_run_id)
    print(f"Config: {best_run.config}")
    print(f"Metrics: {best_run.metrics}")
```

---

## Detailed Usage (API Reference)

### `MLRun`

Manages a single experiment run.

#### `MLRun.create(config, base_dir, project_name, entity)`
* **Action**: Starts a new experiment.
* **Details**:
    1.  Calls `wandb.init(project=project_name, entity=entity, config=config)`.
    2.  Gets the `run_id` from the `wandb` response.
    3.  Detects and records the current git commit hash (if in a git repository).
    4.  Creates a directory: `{base_dir}/{project_name}/{run_id}`.
    5.  Saves the `config` (plus `_wandb` and `_meta` metadata including git commit) to `{base_dir}/{project_name}/{run_id}/config.toml`.
* **Returns**: An `MLRun` instance with an active `wandb_run` connection.

#### `run.add_metrics(metrics_dict)`
* **Action**: Logs metrics for the run.
* **Details**:
    1.  Updates the `run.metrics` attribute in the object.
    2.  Saves the *entire* `run.metrics` dictionary to `{run.run_dir}/metrics.toml`.
    3.  If `run.wandb_run` is active, calls `wandb_run.log(metrics_dict)` to log to the cloud.

#### `run.finish()`
* **Action**: Finishes the active `wandb` run.
* **Details**: Calls `run.wandb_run.finish()`. This should be called at the end of your training script.

#### `MLRun.load(run_id, base_dir, project_name=None)`
* **Action**: Loads an existing experiment run from local files.
* **Details**:
    1.  Reads `{base_dir}/{project_name}/{run_id}/config.toml`.
    2.  Reads `{base_dir}/{project_name}/{run_id}/metrics.toml` (if it exists).
    3.  If `project_name` is not provided, automatically searches for the `run_id` in all project subdirectories.
* **Returns**: An `MLRun` instance. This instance **does not** have an active `wandb` connection (`wandb_run` is `None`).

#### `run.get_wandb_url()`
* **Action**: Gets the URL for the run on the `wandb` dashboard.
* **Returns**: A string (e.g., `https://wandb.ai/your_entity/your_project/runs/your_run_id`).

### `MLProject`

Manages and queries a collection of runs stored in the `base_dir`.

#### `project = MLProject(base_dir)`
* **Action**: Initializes the project manager.
* **Details**: When initialized, it immediately scans the `base_dir` by calling `project.load_project()`.

#### `project.load_project()`
* **Action**: Scans all subdirectories in `base_dir` for `config.toml` and `metrics.toml` files.
* **Details**: It loads all configs and metrics from the `{base_dir}/{project_name}/{run_id}` directory structure, merges them, and builds a `pandas` DataFrame stored in `project.df`. This method is called automatically on init, but you can call it again to refresh the DataFrame.

#### `project.df`
* **Attribute**: The `pandas.DataFrame` containing all experiment data.
* **Details**: Nested dictionaries from your config (e.g., `{"model": {"name": "CNN"}}`) are flattened into columns using `.` as a separator (e.g., `model.name`).

#### `project.search(query_string, return_objects=False)`
* **Action**: Queries the `project.df` DataFrame.
* **Arguments**:
    * `query_string` (str): A `pandas.DataFrame.query()` string.
        * **Important**: For nested config keys (like `model.name`), you **must** use backticks (`` ` ``) in your query.
        * *Example*: ``"`model.name` == 'BERT' and accuracy > 0.9"``
    * `return_objects` (bool):
        * If `False` (default): Returns a `pandas.DataFrame` (a subset of `project.df`).
        * If `True`: Returns a `List[MLRun]` of loaded `MLRun` objects matching the query.

#### `project.get_run(run_id, project_name=None)`
* **Action**: A convenience method to load a single `MLRun` object from the project's `base_dir`.
* **Arguments**:
    * `run_id` (str): The unique identifier of the run.
    * `project_name` (str, optional): The project name. If not provided, searches all projects.
* **Returns**: An `MLRun` instance.

---

## Testing

This project uses `pytest` and a `src` layout.

1.  **Install development dependencies:**
    ```bash
    pip install pytest pytest-mock
    ```

2.  **Configure `pytest`:**
    To ensure `pytest` can find the `ml_manager` module in the `src` directory, create or update your `pyproject.toml` file in the project root with the following:

    **`pyproject.toml`**
    ```toml
    [tool.pytest.ini_options]
    pythonpath = [
        "src"
    ]
    ```

3.  **Run tests:**
    From the project's root directory, simply run:
    ```bash
    pytest
    ```

