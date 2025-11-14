import pytest
import toml
import os
import time  # Added to create timestamp differences
import datetime  # Added for timestamp testing
from unittest.mock import MagicMock
import pandas.api.types as ptypes  # For pandas type checking

# Import the classes to be tested
# (Assuming they are saved in 'ml_manager.py' in the same directory)
from ml_manager import MLRun, MLProject, get_git_commit

# --- Fixtures (Test Setup) --------------------------------------------------

@pytest.fixture
def mock_wandb_init(mocker):
    """
    Fixture that mocks (replaces with a fake) 'wandb.init'.
    """
    mock_run = MagicMock()
    mock_init = mocker.patch("wandb.init", return_value=mock_run)
    return mock_init, mock_run

# --- Change 1: Modified fixture to create timestamp differences ---
@pytest.fixture
def populated_project_dir(tmp_path, mock_wandb_init):
    """
    Creates a temporary directory with multiple runs
    for testing MLProject.
    ★ Intentionally stagger the creation time of each Run ★
    """
    mock_init, mock_run = mock_wandb_init
    base_dir = str(tmp_path)

    # --- Run 1: CNN ---
    mock_run.id = "run_cnn_1"
    mock_run.entity = "test_entity"
    mock_run.project = "test_project"
    config1 = {"lr": 0.01, "model": {"name": "CNN"}}
    run1 = MLRun.create(config1, base_dir, "test_project", "test_entity")
    run1.add_metrics({"accuracy": 0.90, "f1": 0.88})
    run1.finish()

    # --- Wait to create a clear timestamp difference ---
    time.sleep(0.1)

    # --- Run 2: BERT ---
    mock_run.id = "run_bert_1"
    config2 = {"lr": 0.001, "model": {"name": "BERT"}}
    run2 = MLRun.create(config2, base_dir, "test_project", "test_entity")
    run2.add_metrics({"accuracy": 0.95, "f1": 0.92})
    run2.finish()

    time.sleep(0.1)

    # --- Run 3: CNN (different parameters) ---
    mock_run.id = "run_cnn_2"
    config3 = {"lr": 0.1, "model": {"name": "CNN"}}
    run3 = MLRun.create(config3, base_dir, "test_project", "test_entity")
    run3.add_metrics({"accuracy": 0.80, "f1": 0.75})
    run3.finish()

    # Return the path to the directory with 3 runs and the creation time of the middle run (run2)
    # (for timestamp search testing)
    return base_dir, run2.created_at

@pytest.fixture
def mock_datetime(mocker):
    """
    Fixture that ensures when 'ml_manager' does 'import datetime',
    it gets a module with a mocked datetime.datetime class.
    
    The most robust way to mock C-extensions (datetime).
    """
    # 1. Fixed test time
    test_time = datetime.datetime(2023, 10, 27, 12, 0, 0, tzinfo=datetime.timezone.utc)

    # 2. Create a mock 'datetime' *class*
    #    (conforms to the spec of the real datetime.datetime class)
    mock_dt_class = mocker.MagicMock(spec=datetime.datetime)
    
    # 3. When 'now' is called (used in create), return the fixed time
    mock_dt_class.now.return_value = test_time

    # 4. When 'fromisoformat' is called (used in load),
    #    call the real method (side_effect)
    mock_dt_class.fromisoformat.side_effect = datetime.datetime.fromisoformat

    # 5. Create a mock 'datetime' *module*
    #    (specify 'wraps' so that other features like 'timedelta'
    #     from the real 'datetime' module can still be used)
    mock_dt_module = mocker.MagicMock(wraps=datetime)
    
    # 6. Replace the 'datetime' attribute (class) of the mock module
    #    with the mock class created in (2) above
    mock_dt_module.datetime = mock_dt_class

    # 7. (Most important) Patch so that when 'ml_manager' references 'datetime',
    #    this mock module is returned instead of the real module.
    mocker.patch('ml_manager.datetime', mock_dt_module)

    return test_time


# --- MLRun Class Tests ---------------------------------------------------

# --- Change 3: Added timestamp test to create ---
def test_mlrun_create_with_timestamp(tmp_path, mock_wandb_init, mock_datetime):
    """
    Test that MLRun.create correctly records timestamp (_meta.created_at)
    """
    mock_init, mock_run = mock_wandb_init
    mock_run.id = "test_id_123"
    mock_run.entity = "my_entity"
    mock_run.project = "my_project"
    
    config = {"learning_rate": 0.01}
    base_dir = str(tmp_path)
    
    # Due to 'mock_datetime' fixture,
    # datetime.datetime.now() inside MLRun.create returns fixed time
    fixed_time = mock_datetime 

    # 1. Action
    run = MLRun.create(
        config=config,
        base_dir=base_dir,
        project_name="my_project",
        entity="my_entity"
    )

    # 2. Verification (wandb)
    mock_init.assert_called_with(
        project="my_project",
        entity="my_entity",
        config=config
    )
    assert run.run_id == "test_id_123"

    # 3. Verification (filesystem and config.toml contents)
    config_path = os.path.join(base_dir, "my_project", "test_id_123", "config.toml")
    assert os.path.exists(config_path)

    loaded_config = toml.load(config_path)
    # Check if _meta.created_at is saved as ISO format string
    assert "_meta" in loaded_config
    assert loaded_config["_meta"]["created_at"] == fixed_time.isoformat()
    # Also verify _wandb metadata
    assert loaded_config["_wandb"]["run_id"] == "test_id_123"
    
    # 4. Verification (MLRun instance)
    # Check if instance variable is stored as datetime object
    assert run.created_at == fixed_time
    assert isinstance(run.created_at, datetime.datetime)

def test_mlrun_add_metrics(tmp_path, mock_wandb_init):
    """
    Test that MLRun.add_metrics creates metrics.toml and calls wandb.log
    (no changes from previous version)
    """
    # 1. Setup
    mock_init, mock_run = mock_wandb_init
    mock_run.id = "test_metrics_run"
    config = {"lr": 0.1}
    # mock_datetime not needed as this test is unrelated to timestamp
    run = MLRun.create(config, str(tmp_path), "p", "e")

    # 2. Action
    metrics1 = {"accuracy": 0.9, "epoch": 1}
    run.add_metrics(metrics1)
    
    metrics2 = {"f1_score": 0.88, "epoch": 2}
    run.add_metrics(metrics2, record_to_wandb=True) # Test append/overwrite

    # 3. Verification (filesystem)
    metrics_path = os.path.join(run.run_dir, "metrics.toml")
    loaded_metrics = toml.load(metrics_path)
    assert loaded_metrics["accuracy"] == 0.9
    assert loaded_metrics["f1_score"] == 0.88
    assert loaded_metrics["epoch"] == 2

    # 4. Verification (wandb)
    mock_run.log.assert_called_with(metrics2)

def test_mlrun_add_metrics_no_wandb_logging(tmp_path, mock_wandb_init):
    """
    Test that MLRun.add_metrics with record_to_wandb=False does not call wandb.log
    """
    # 1. Setup
    mock_init, mock_run = mock_wandb_init
    mock_run.id = "test_no_wandb_run"
    config = {"lr": 0.05}
    run = MLRun.create(config, str(tmp_path), "p", "e")
    
    # Reset the mock to clear any previous calls
    mock_run.log.reset_mock()

    # 2. Action - add metrics without logging to wandb
    metrics = {"accuracy": 0.95, "loss": 0.05}
    run.add_metrics(metrics, record_to_wandb=False)

    # 3. Verification (filesystem - should still save)
    metrics_path = os.path.join(run.run_dir, "metrics.toml")
    loaded_metrics = toml.load(metrics_path)
    assert loaded_metrics["accuracy"] == 0.95
    assert loaded_metrics["loss"] == 0.05

    # 4. Verification (wandb - should NOT be called)
    mock_run.log.assert_not_called()

# --- Change 4: Added timestamp test to load ---
def test_mlrun_load_with_timestamp(tmp_path):
    """
    Test that MLRun.load correctly loads timestamp from config.toml
    and metrics.toml
    """
    # 1. Setup: Manually create test files
    run_id = "load_test_run"
    project_name = "test_project"
    project_dir = tmp_path / project_name
    project_dir.mkdir()
    run_dir = project_dir / run_id
    run_dir.mkdir()
    
    # Fixed test time (ISO string)
    # (fromisoformat can handle UTC offsets like +00:00)
    test_time_str = "2023-01-01T10:00:00+00:00"
    # Corresponding datetime object (for parsing verification)
    test_time_obj = datetime.datetime.fromisoformat(test_time_str)

    # Create config.toml
    config_data = {
        "lr": 0.5,
        "model": "manual",
        "_wandb": {"run_id": run_id, "project": "p"},
        "_meta": {"created_at": test_time_str} # ★ Timestamp
    }
    toml.dump(config_data, open(run_dir / "config.toml", "w"))
    
    # Create metrics.toml
    metrics_data = {"accuracy": 0.99, "val_loss": 0.1}
    toml.dump(metrics_data, open(run_dir / "metrics.toml", "w"))

    # 2. Action
    run = MLRun.load(run_id=run_id, base_dir=str(tmp_path))

    # 3. Verification
    assert run.run_id == run_id
    # config (metadata should be removed)
    assert "lr" in run.config
    assert "_wandb" not in run.config
    assert "_meta" not in run.config
    # metrics
    assert "accuracy" in run.metrics
    assert run.metrics["accuracy"] == 0.99
    # ★ Timestamp (should be loaded as datetime object)
    assert run.created_at == test_time_obj
    assert isinstance(run.created_at, datetime.datetime)

def test_mlrun_load_no_metrics_file(tmp_path):
    """
    Test that MLRun.load doesn't error when metrics.toml doesn't exist
    """
    run_id = "no_metrics_run"
    project_name = "test_project"
    project_dir = tmp_path / project_name
    project_dir.mkdir()
    run_dir = project_dir / run_id
    run_dir.mkdir()
    config_data = {
        "lr": 0.1, 
        "_wandb": {"run_id": run_id},
        "_meta": {"created_at": "2023-01-01T00:00:00+00:00"}
    }
    toml.dump(config_data, open(run_dir / "config.toml", "w"))
    
    run = MLRun.load(run_id=run_id, base_dir=str(tmp_path), project_name=project_name)
    
    assert run.run_id == run_id
    assert run.config["lr"] == 0.1
    assert run.metrics == {} # metrics is empty dict
    assert run.created_at is not None # timestamp is loaded

def test_mlrun_load_not_found(tmp_path):
    """
    Test that FileNotFoundError is raised when trying to load non-existent run_id
    """
    with pytest.raises(FileNotFoundError):
        MLRun.load(run_id="non_existent_id", base_dir=str(tmp_path))


# --- Git Commit Tests ---

def test_get_git_commit():
    """
    Test that get_git_commit returns a commit hash when in a git repo
    """
    # This test assumes the test is run within a git repository
    commit = get_git_commit()
    # If we're in a git repo, should return a 40-char hex string
    if commit is not None:
        assert len(commit) == 40
        assert all(c in '0123456789abcdef' for c in commit)

def test_mlrun_create_with_git_commit(tmp_path, mock_wandb_init, mocker):
    """
    Test that MLRun.create correctly records git commit if in a git repo
    """
    mock_init, mock_run = mock_wandb_init
    mock_run.id = "test_git_run"
    mock_run.entity = "test_entity"
    mock_run.project = "test_project"
    
    # Mock get_git_commit to return a fake commit hash
    fake_commit = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0"
    mocker.patch('ml_manager.get_git_commit', return_value=fake_commit)
    
    config = {"lr": 0.01}
    run = MLRun.create(config, str(tmp_path), "test_project", "test_entity")
    
    # Verify git commit is stored in instance
    assert run.git_commit == fake_commit
    
    # Verify git commit is saved to config.toml
    config_path = os.path.join(tmp_path, "test_project", "test_git_run", "config.toml")
    loaded_config = toml.load(config_path)
    assert "_meta" in loaded_config
    assert loaded_config["_meta"]["git_commit"] == fake_commit
    
    run.finish()

def test_mlrun_load_with_git_commit(tmp_path):
    """
    Test that MLRun.load correctly loads git commit from config.toml
    """
    run_id = "git_load_test"
    project_name = "test_project"
    project_dir = tmp_path / project_name
    project_dir.mkdir()
    run_dir = project_dir / run_id
    run_dir.mkdir()
    
    fake_commit = "1234567890abcdef1234567890abcdef12345678"
    
    config_data = {
        "lr": 0.5,
        "_wandb": {"run_id": run_id, "project": project_name},
        "_meta": {
            "created_at": "2023-01-01T00:00:00+00:00",
            "git_commit": fake_commit
        }
    }
    toml.dump(config_data, open(run_dir / "config.toml", "w"))
    
    run = MLRun.load(run_id=run_id, base_dir=str(tmp_path), project_name=project_name)
    
    assert run.git_commit == fake_commit
    assert run.run_id == run_id


# --- MLProject Class Tests -------------------------------------------------

# --- Change 5: Added timestamp type check to project load ---
def test_project_load_and_timestamp_type(populated_project_dir):
    """
    Test that MLProject loads into DataFrame and timestamp column
    has the correct type (datetime64)
    """
    # 1. Action
    base_dir, _ = populated_project_dir # Use only directory path
    project = MLProject(base_dir=base_dir)

    # 2. Verification (basic)
    df = project.df
    assert len(project) == 3
    assert len(df) == 3
    
    # Verify config and metrics columns are loaded
    assert "lr" in df.columns
    assert "model.name" in df.columns
    assert "accuracy" in df.columns
    
    # 3. Verification (★ Timestamp)
    assert "_meta.created_at" in df.columns
    
    # Check if column type is datetime64 (pandas datetime type)
    assert ptypes.is_datetime64_any_dtype(df["_meta.created_at"])
    
    # Verify timestamp is not None (NaT)
    assert not df["_meta.created_at"].isnull().any()

def test_project_load_empty(tmp_path):
    """
    Test that initializing MLProject with empty directory doesn't error
    """
    project = MLProject(base_dir=str(tmp_path))
    assert len(project) == 0
    assert project.df.empty

def test_project_search_by_metric(populated_project_dir):
    """
    Test searching by metric (accuracy)
    """
    base_dir, _ = populated_project_dir
    project = MLProject(base_dir=base_dir)
    results_df = project.search(query_string="accuracy > 0.92")
    assert len(results_df) == 1
    assert results_df.iloc[0]["run_id"] == "run_bert_1"

def test_project_search_by_nested_config(populated_project_dir):
    """
    Test searching by nested config (model.name)
    """
    base_dir, _ = populated_project_dir
    project = MLProject(base_dir=base_dir)
    results_df = project.search(query_string="`model.name` == 'CNN'")
    assert len(results_df) == 2

# --- Change 6: Added new timestamp search test ---
def test_project_search_by_timestamp(populated_project_dir):
    """
    Test searching by timestamp (string comparison)
    """
    # 1. Setup
    base_dir, run2_timestamp = populated_project_dir
    project = MLProject(base_dir=base_dir)
    
    # 2. Action
    # Search for runs created *after* run2 (BERT) (should be run3)
    # In pandas.query, datetime objects can be queried as strings
    query_str = f"`_meta.created_at` > '{run2_timestamp.isoformat()}'"
    
    results_df = project.search(query_string=query_str)
    
    # 3. Verification
    assert len(results_df) == 1
    assert results_df.iloc[0]["run_id"] == "run_cnn_2"

    # 4. Action (reverse query)
    # Search for runs created *before* run2 (BERT) (should be run1)
    query_str_before = f"`_meta.created_at` < '{run2_timestamp.isoformat()}'"
    results_df_before = project.search(query_string=query_str_before)
    
    # 5. Verification
    assert len(results_df_before) == 1
    assert results_df_before.iloc[0]["run_id"] == "run_cnn_1"

def test_project_search_combined(populated_project_dir):
    """
    Test combined search with config and metrics
    """
    base_dir, _ = populated_project_dir
    project = MLProject(base_dir=base_dir)
    
    # CNN model with accuracy >= 0.85 (should be run_cnn_1)
    query = "`model.name` == 'CNN' and accuracy > 0.85"
    results_df = project.search(query_string=query)
    
    assert len(results_df) == 1
    assert results_df.iloc[0]["run_id"] == "run_cnn_1"

def test_project_search_return_objects(populated_project_dir):
    """
    Test that return_objects=True returns list of MLRun objects
    """
    base_dir, _ = populated_project_dir
    project = MLProject(base_dir=base_dir)
    
    results_list = project.search(query_string="lr == 0.001", return_objects=True)
    
    assert isinstance(results_list, list)
    assert len(results_list) == 1
    assert isinstance(results_list[0], MLRun)
    assert results_list[0].run_id == "run_bert_1"
    assert results_list[0].config["lr"] == 0.001

def test_project_get_run(populated_project_dir):
    """
    Test that get_run correctly loads MLRun from run_id
    """
    base_dir, _ = populated_project_dir
    project = MLProject(base_dir=base_dir)
    
    run = project.get_run("run_bert_1")
    
    assert isinstance(run, MLRun)
    assert run.metrics["accuracy"] == 0.95
    assert run.config["model"]["name"] == "BERT"
    # Timestamp should also be loaded
    assert isinstance(run.created_at, datetime.datetime)
