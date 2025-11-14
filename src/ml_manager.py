import wandb
import toml
import os
import glob
import pandas as pd
from typing import Dict, Any, Optional, List
import datetime  # Added for timestamp
import subprocess  # Added for git commands

# --- Helper functions ---

def get_git_commit(cwd: Optional[str] = None) -> Optional[str]:
    """
    Get the current git commit hash if the directory is in a git repository.
    
    Args:
        cwd: Working directory to check. If None, uses current directory.
    
    Returns:
        Git commit hash (str) if in a git repo, None otherwise.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    return None

# --- MLRun class ---

class MLRun:
    """Class to manage a single machine learning run"""

    def __init__(
        self,
        run_id: str,
        config: Dict[str, Any],  # Pure user config
        run_dir: str,
        metrics: Dict[str, Any] = None,
        # --- Change 1: Accept metadata as arguments ---
        created_at: Optional[datetime.datetime] = None,
        wandb_entity: Optional[str] = None,
        wandb_project: Optional[str] = None,
        git_commit: Optional[str] = None,
    ):
        self.run_id = run_id
        self.config = config  # Original config specified by user
        self.run_dir = run_dir
        self.metrics = metrics if metrics is not None else {}
        self.wandb_run = None  # Active run object returned by wandb.init()

        # --- Change 2: Store metadata as instance variables ---
        self.created_at = created_at
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project
        self.git_commit = git_commit

    @classmethod
    def create(
        cls,
        config: Dict[str, Any],
        base_dir: str,
        project_name: str,
        entity: Optional[str] = None,
    ) -> "MLRun":
        """
        Create a new run, call wandb.init(), and save config locally.
        """
        print(f"Creating new run in project '{project_name}'...")
        # 1. Initialize wandb
        wandb_run = wandb.init(
            project=project_name,
            entity=entity,
            config=config,
        )

        # --- Change 3: Generate timestamp (UTC) ---
        created_at_time = datetime.datetime.now(datetime.timezone.utc)
        created_at_str = created_at_time.isoformat()

        # Get git commit if in a git repository
        git_commit = get_git_commit()

        # 2. Get information from wandb
        run_id = wandb_run.id
        wandb_entity = wandb_run.entity
        wandb_project = wandb_run.project
        run_dir = os.path.join(base_dir, project_name, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # 3. Save metadata together in config.toml
        full_config = config.copy()
        # wandb metadata
        full_config["_wandb"] = {
            "entity": wandb_entity,
            "project": wandb_project,
            "run_id": run_id,
        }
        # --- Change 4: Add timestamp metadata ---
        full_config["_meta"] = {
            "created_at": created_at_str
        }
        # Add git commit if available
        if git_commit:
            full_config["_meta"]["git_commit"] = git_commit

        config_path = os.path.join(run_dir, "config.toml")
        with open(config_path, "w") as f:
            toml.dump(full_config, f)
        print(f"Run {run_id} created. Config saved to {config_path}")

        # 4. Create MLRun instance
        instance = cls(
            run_id=run_id,
            config=config, # Pure config
            run_dir=run_dir,
            metrics={},
            created_at=created_at_time, # --- Change 5: Pass datetime object ---
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            git_commit=git_commit,
        )
        instance.wandb_run = wandb_run
        return instance

    @classmethod
    def load(cls, run_id: str, base_dir: str, project_name: Optional[str] = None) -> "MLRun":
        """
        Load local config.toml and metrics.toml from existing run_id.
        If project_name is not provided, searches for run_id in all project subdirectories.
        """
        if project_name:
            run_dir = os.path.join(base_dir, project_name, run_id)
        else:
            # Search for run_id in all subdirectories
            found_dirs = glob.glob(os.path.join(base_dir, "*", run_id))
            if not found_dirs:
                raise FileNotFoundError(
                    f"run_id {run_id} not found in any project under {base_dir}"
                )
            if len(found_dirs) > 1:
                raise ValueError(
                    f"Multiple runs with id {run_id} found: {found_dirs}. Please specify project_name."
                )
            run_dir = found_dirs[0]
        
        config_path = os.path.join(run_dir, "config.toml")

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"config.toml not found for run_id {run_id} in {base_dir}"
            )

        with open(config_path, "r") as f:
            full_config = toml.load(f)

        # --- Change 6: Extract metadata while removing from config ---
        wandb_info = full_config.pop("_wandb", {})
        meta_info = full_config.pop("_meta", {})
        
        # What remains is the original config
        user_config = full_config

        # Parse timestamp
        created_at_obj = None
        created_at_str = meta_info.get("created_at")
        if created_at_str:
            try:
                # Convert from ISO format to datetime object
                created_at_obj = datetime.datetime.fromisoformat(created_at_str)
            except ValueError:
                print(f"Warning (Run {run_id}): Could not parse created_at string: {created_at_str}")

        # Extract git commit
        git_commit = meta_info.get("git_commit")

        # Load metrics.toml as well
        metrics_path = os.path.join(run_dir, "metrics.toml")
        metrics = {}
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as f:
                    metrics = toml.load(f)
            except Exception as e:
                print(f"Warning: Could not load metrics {metrics_path}: {e}")

        # --- Change 7: Pass extracted metadata to constructor ---
        return cls(
            run_id=run_id,
            config=user_config,
            run_dir=run_dir,
            metrics=metrics,
            created_at=created_at_obj,
            wandb_entity=wandb_info.get("entity"),
            wandb_project=wandb_info.get("project"),
            git_commit=git_commit,
        )

    def add_metrics(self, metrics_dict: Dict[str, Any]):
        """
        Register metrics as a dictionary and save to local metrics.toml.
        """
        if not isinstance(metrics_dict, dict):
            print(f"Error (Run {self.run_id}): metrics must be a dictionary.")
            return

        self.metrics.update(metrics_dict)
        print(f"Run {self.run_id}: Metrics updated in memory: {metrics_dict}")

        metrics_path = os.path.join(self.run_dir, "metrics.toml")
        try:
            with open(metrics_path, "w") as f:
                toml.dump(self.metrics, f)
            print(f"Metrics saved to {metrics_path}")
        except Exception as e:
            print(f"Error saving metrics to {metrics_path}: {e}")

        if self.wandb_run:
            self.wandb_run.log(metrics_dict)
            print(f"Logged to wandb: {metrics_dict}")

    def finish(self):
        """Finish the active wandb run"""
        if self.wandb_run:
            self.wandb_run.finish()
            self.wandb_run = None
            print(f"Run {self.run_id} finished.")
        else:
            print(f"Run {self.run_id} was not active. No need to finish.")

    def get_wandb_url(self) -> str:
        """Return URL to wandb dashboard"""
        if self.wandb_entity and self.wandb_project:
            return f"https://wandb.ai/{self.wandb_entity}/{self.wandb_project}/runs/{self.run_id}"
        else:
            return "Could not determine wandb URL (entity or project missing)."

    def __repr__(self):
        ts_str = self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else 'UnknownTime'
        git_str = f", git={self.git_commit[:7]}" if self.git_commit else ""
        return f"<MLRun (id={self.run_id}, created={ts_str}{git_str})>"


# --- MLProject class ---

class MLProject:
    """Project class to manage multiple MLRuns"""

    def __init__(self, base_dir: str = "./checkpoints"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True) # Create folder if it doesn't exist
        self.df = pd.DataFrame()
        self.load_project() # Load on initialization

    def load_project(self):
        """
        Load all config.toml and metrics.toml in base_dir,
        merge them and convert to DataFrame.
        """
        config_paths = glob.glob(os.path.join(self.base_dir, "*", "*", "config.toml"))
        
        all_data = []

        for path in config_paths:
            run_id = os.path.basename(os.path.dirname(path))
            run_dir = os.path.dirname(path)
            
            try:
                # 1. Load config.toml (contains _wandb, _meta)
                with open(path, "r") as f:
                    data = toml.load(f)
                
                data["run_id"] = run_id
                data["run_dir"] = run_dir

                # 2. Load corresponding metrics.toml and merge
                metrics_path = os.path.join(run_dir, "metrics.toml")
                if os.path.exists(metrics_path):
                    try:
                        with open(metrics_path, "r") as f:
                            metrics = toml.load(f)
                        data.update(metrics)
                    except Exception as e:
                        print(f"Warning: Could not load metrics {metrics_path}: {e}")

                all_data.append(data)

            except Exception as e:
                print(f"Error loading {path}: {e}")

        if not all_data:
            print(f"No runs found in {self.base_dir}")
            self.df = pd.DataFrame()
        else:
            # 3. Build DataFrame
            # (Nested keys are expanded to 'model.name' or '_meta.created_at')
            self.df = pd.json_normalize(all_data, sep=".")
            
            # --- Change 8: Convert timestamp column to datetime type ---
            if "_meta.created_at" in self.df.columns:
                self.df["_meta.created_at"] = pd.to_datetime(
                    self.df["_meta.created_at"]
                )
                print("Converted '_meta.created_at' column to datetime type.")
            # ---

            print(f"Loaded {len(self.df)} runs (with metrics) from {self.base_dir}")

    def search(
        self, query_string: str, return_objects: bool = False
    ) -> pd.DataFrame | List[MLRun]:
        """
        Search DataFrame with query string.
        (Example query with timestamp:
         "`_meta.created_at` > '2023-10-27 12:00:00'")
        """
        if self.df.empty:
            print("Search warning: DataFrame is empty.")
            return pd.DataFrame() if not return_objects else []
        
        try:
            results_df = self.df.query(query_string).copy()
        except Exception as e:
            print(f"Query failed: {e}")
            print("---")
            print("Hint: Nested keys (e.g., 'model.name', '_meta.created_at')")
            print("should be enclosed in backticks: `model.name` == 'bert'")
            print("Metrics (e.g., 'accuracy > 0.9') can be searched directly.")
            print("---")
            return pd.DataFrame() if not return_objects else []

        # (Bonus) Add wandb_url column to DataFrame
        url_cols = ["_wandb.entity", "_wandb.project", "run_id"]
        if all(col in results_df.columns for col in url_cols):
            results_df["wandb_url"] = results_df.apply(
                lambda row: f"https://wandb.ai/{row['_wandb.entity']}/{row['_wandb.project']}/runs/{row['run_id']}",
                axis=1,
            )
            
        if return_objects:
            return [
                self.get_run(run_id) for run_id in results_df["run_id"]
            ]
        else:
            return results_df

    def get_run(self, run_id: str, project_name: Optional[str] = None) -> MLRun:
        """Load MLRun object by specifying run_id and optional project_name"""
        return MLRun.load(run_id, self.base_dir, project_name)

    def __len__(self):
        return len(self.df)

    def __repr__(self):
        return f"<MLProject (path={self.base_dir}, runs={len(self)})>"

# =============================================================================
# 🚀 Example Usage
# =============================================================================
if __name__ == "__main__":

    # (Assumes logged into wandb)
    # wandb.login() 

    import shutil
    import time
    DEMO_DIR = "./checkpoints_demo"
    
    # --- 1. Clean up test directory ---
    if os.path.exists(DEMO_DIR):
        print(f"Cleaning up old demo directory: {DEMO_DIR}\n")
        shutil.rmtree(DEMO_DIR)

    # --- 2. Prepare project ---
    WANDB_ENTITY = "causal-rl" # ★ Change to your wandb entity
    WANDB_PROJECT = "mlproject-demo-v2"
    
    # --- 3. Run Experiment 1 (CNN) ---
    print("\n" + "="*30)
    print("--- Running Experiment 1 (CNN) ---")
    config1 = {
        "learning_rate": 0.001,
        "batch_size": 32,
        "model": {"name": "CNN", "layers": 5}
    }
    
    run1 = MLRun.create(
        config=config1,
        base_dir=DEMO_DIR,
        project_name=WANDB_PROJECT,
        entity=WANDB_ENTITY
    )
    print(f"Run 1 Object: {run1}") # Check __repr__
    
    run1.add_metrics({"accuracy": 0.92, "f1_score": 0.91, "epoch": 10})
    run1.finish()

    # --- 4. Run Experiment 2 (BERT) (wait a few seconds) ---
    print("\n...waiting 2 seconds to ensure different timestamps...")
    time.sleep(2)
    
    print("\n" + "="*30)
    print("--- Running Experiment 2 (BERT) ---")
    config2 = {
        "learning_rate": 5e-5,
        "batch_size": 16,
        "model": {"name": "BERT", "layers": 12}
    }
    
    run2 = MLRun.create(
        config=config2,
        base_dir=DEMO_DIR,
        project_name=WANDB_PROJECT,
        entity=WANDB_ENTITY
    )
    print(f"Run 2 Object: {run2}")
    
    run2.add_metrics({"accuracy": 0.96, "f1_score": 0.95, "epoch": 5})
    run2.finish()

    # --- 5. Load project and check metrics and timestamps ---
    print("\n" + "="*30)
    print("--- Loading Project and Checking DataFrame ---")
    
    project = MLProject(base_dir=DEMO_DIR)
    
    print("\n[DataFrame with Metrics and Timestamps]")
    # Verify that '_meta.created_at' column is added
    # (If too many columns, select only relevant columns)
    display_cols = [
        "run_id", 
        "_meta.created_at", 
        "model.name", 
        "accuracy", 
        "learning_rate"
    ]
    # Display only columns that exist in df.columns
    display_cols = [col for col in display_cols if col in project.df.columns]
    
    print(project.df[display_cols].to_markdown(index=False))

    # --- 6. Sort by timestamp ---
    print("\n" + "="*30)
    print("--- Sorting by Timestamp (DESC) ---")
    
    # Sorts correctly because it's datetime type
    sorted_df = project.df.sort_values(by="_meta.created_at", ascending=False)
    print(sorted_df[display_cols].to_markdown(index=False))

    # --- 7. Search by timestamp ---
    print("\n" + "="*30)
    print("--- Searching by Timestamp (only run2) ---")
    
    # Get mid-time between run1 and run2 (simplified)
    if run1.created_at and run2.created_at:
        mid_time = run1.created_at + (run2.created_at - run1.created_at) / 2
        mid_time_str = mid_time.isoformat() # Query with ISO string
        
        query = f"`_meta.created_at` > '{mid_time_str}'"
        print(f"Query: {query}")
        
        results = project.search(query_string=query)
        print(results[display_cols].to_markdown(index=False))