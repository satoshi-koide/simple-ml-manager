import pytest
import toml
import os
import time  # <<< タイムスタンプの差を作るために追加
import datetime  # <<< タイムスタンプのテストのために追加
from unittest.mock import MagicMock
import pandas as pd
import pandas.api.types as ptypes  # <<< pandas の型チェック用

# テスト対象のクラスをインポート
# (同じディレクトリ内の 'ml_manager.py' に保存されていると仮定)
from ml_manager import MLRun, MLProject

# --- Fixtures (テストの準備) --------------------------------------------------

@pytest.fixture
def mock_wandb_init(mocker):
    """
    'wandb.init' をモック(偽物に入れ替え)するフィクスチャ。
    """
    mock_run = MagicMock()
    mock_init = mocker.patch("wandb.init", return_value=mock_run)
    return mock_init, mock_run

# --- 変更点 1: タイムスタンプの差を作るため、フィクスチャを修正 ---
@pytest.fixture
def populated_project_dir(tmp_path, mock_wandb_init):
    """
    MLProject のテスト用に、複数の実行結果 (run) が
    すでに入っている一時ディレクトリを作成する。
    ★ 各 Run の作成時刻を意図的にずらす ★
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

    # --- タイムスタンプに明確な差をつけるために待機 ---
    time.sleep(0.1)

    # --- Run 2: BERT ---
    mock_run.id = "run_bert_1"
    config2 = {"lr": 0.001, "model": {"name": "BERT"}}
    run2 = MLRun.create(config2, base_dir, "test_project", "test_entity")
    run2.add_metrics({"accuracy": 0.95, "f1": 0.92})
    run2.finish()

    time.sleep(0.1)

    # --- Run 3: CNN (別パラメータ) ---
    mock_run.id = "run_cnn_2"
    config3 = {"lr": 0.1, "model": {"name": "CNN"}}
    run3 = MLRun.create(config3, base_dir, "test_project", "test_entity")
    run3.add_metrics({"accuracy": 0.80, "f1": 0.75})
    run3.finish()

    # 3つの run が入ったディレクトリのパスと、中間の run (run2) の作成時刻を返す
    # (タイムスタンプ検索テストのため)
    return base_dir, run2.created_at

@pytest.fixture
def mock_datetime(mocker):
    """
    'ml_manager' が 'import datetime' したときに、
    datetime.datetime クラスがモックされたモジュールを
    返すようにするフィクスチャ。
    
    C-extension (datetime) をモックする最も堅牢な方法。
    """
    # 1. 固定のテスト時刻
    test_time = datetime.datetime(2023, 10, 27, 12, 0, 0, tzinfo=datetime.timezone.utc)

    # 2. モックの 'datetime' *クラス* を作成
    #    (本物の datetime.datetime クラスの仕様に準拠させる)
    mock_dt_class = mocker.MagicMock(spec=datetime.datetime)
    
    # 3. 'now' (create で使用) が呼ばれたら固定時刻を返す
    mock_dt_class.now.return_value = test_time

    # 4. 'fromisoformat' (load で使用) が呼ばれたら、
    #    本物のメソッドを呼び出す (side_effect)
    mock_dt_class.fromisoformat.side_effect = datetime.datetime.fromisoformat

    # 5. モックの 'datetime' *モジュール* を作成
    #    (本物の 'datetime' モジュールが持つ 'timedelta' など
    #     他の機能も使えるように 'wraps' を指定)
    mock_dt_module = mocker.MagicMock(wraps=datetime)
    
    # 6. モックモジュールの 'datetime' 属性 (クラス) を、
    #    上記(2)で作成したモッククラスに差し替える
    mock_dt_module.datetime = mock_dt_class

    # 7. (最重要) 'ml_manager' が 'datetime' を参照したときに、
    #    本物のモジュールの代わりに、このモックモジュールが
    #    返されるようにパッチする。
    mocker.patch('ml_manager.datetime', mock_dt_module)

    return test_time


# --- MLRun クラスのテスト ---------------------------------------------------

# --- 変更点 3: タイムスタンプのテストを create に追加 ---
def test_mlrun_create_with_timestamp(tmp_path, mock_wandb_init, mock_datetime):
    """
    MLRun.create がタイムスタンプ (_meta.created_at) を正しく記録するかテスト
    """
    mock_init, mock_run = mock_wandb_init
    mock_run.id = "test_id_123"
    mock_run.entity = "my_entity"
    mock_run.project = "my_project"
    
    config = {"learning_rate": 0.01}
    base_dir = str(tmp_path)
    
    # 'mock_datetime' フィクスチャにより、
    # MLRun.create 内部の datetime.datetime.now() は固定時刻を返す
    fixed_time = mock_datetime 

    # 1. アクション
    run = MLRun.create(
        config=config,
        base_dir=base_dir,
        project_name="my_project",
        entity="my_entity"
    )

    # 2. 検証 (wandb)
    mock_init.assert_called_with(
        project="my_project",
        entity="my_entity",
        config=config
    )
    assert run.run_id == "test_id_123"

    # 3. 検証 (ファイルシステムと config.toml の内容)
    config_path = os.path.join(base_dir, "test_id_123", "config.toml")
    assert os.path.exists(config_path)

    loaded_config = toml.load(config_path)
    # _meta.created_at が ISO 形式の文字列で保存されているか
    assert "_meta" in loaded_config
    assert loaded_config["_meta"]["created_at"] == fixed_time.isoformat()
    # _wandb メタデータも確認
    assert loaded_config["_wandb"]["run_id"] == "test_id_123"
    
    # 4. 検証 (MLRun インスタンス)
    # インスタンス変数は datetime オブジェクトとして保持されているか
    assert run.created_at == fixed_time
    assert isinstance(run.created_at, datetime.datetime)

def test_mlrun_add_metrics(tmp_path, mock_wandb_init):
    """
    MLRun.add_metrics が metrics.toml を作成し、wandb.log を呼ぶかテスト
    (このテストは前回から変更なし)
    """
    # 1. 準備
    mock_init, mock_run = mock_wandb_init
    mock_run.id = "test_metrics_run"
    config = {"lr": 0.1}
    # タイムスタンプテストと関係ないので mock_datetime は不要
    run = MLRun.create(config, str(tmp_path), "p", "e")

    # 2. アクション
    metrics1 = {"accuracy": 0.9, "epoch": 1}
    run.add_metrics(metrics1)
    
    metrics2 = {"f1_score": 0.88, "epoch": 2}
    run.add_metrics(metrics2) # 追記・上書きのテスト

    # 3. 検証 (ファイルシステム)
    metrics_path = os.path.join(run.run_dir, "metrics.toml")
    loaded_metrics = toml.load(metrics_path)
    assert loaded_metrics["accuracy"] == 0.9
    assert loaded_metrics["f1_score"] == 0.88
    assert loaded_metrics["epoch"] == 2

    # 4. 検証 (wandb)
    mock_run.log.assert_called_with(metrics2)

# --- 変更点 4: タイムスタンプのテストを load に追加 ---
def test_mlrun_load_with_timestamp(tmp_path):
    """
    MLRun.load が config.toml のタイムスタンプと metrics.toml を
    正しく読み込めるかテスト
    """
    # 1. 準備: テスト用のファイルを手動で作成
    run_id = "load_test_run"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    
    # テスト用の固定時刻 (ISO文字列)
    # (fromisoformat は +00:00 のような UTC オフセットを扱える)
    test_time_str = "2023-01-01T10:00:00+00:00"
    # 対応する datetime オブジェクト (パース確認用)
    test_time_obj = datetime.datetime.fromisoformat(test_time_str)

    # config.toml を作成
    config_data = {
        "lr": 0.5,
        "model": "manual",
        "_wandb": {"run_id": run_id, "project": "p"},
        "_meta": {"created_at": test_time_str} # ★ タイムスタンプ
    }
    toml.dump(config_data, open(run_dir / "config.toml", "w"))
    
    # metrics.toml を作成
    metrics_data = {"accuracy": 0.99, "val_loss": 0.1}
    toml.dump(metrics_data, open(run_dir / "metrics.toml", "w"))

    # 2. アクション
    run = MLRun.load(run_id=run_id, base_dir=str(tmp_path))

    # 3. 検証
    assert run.run_id == run_id
    # config (メタデータが削除されているか)
    assert "lr" in run.config
    assert "_wandb" not in run.config
    assert "_meta" not in run.config
    # metrics
    assert "accuracy" in run.metrics
    assert run.metrics["accuracy"] == 0.99
    # ★ タイムスタンプ (datetime オブジェクトとしてロードされているか)
    assert run.created_at == test_time_obj
    assert isinstance(run.created_at, datetime.datetime)

def test_mlrun_load_no_metrics_file(tmp_path):
    """
    MLRun.load が metrics.toml が存在しなくてもエラーにならないかテスト
    """
    # 1. 準備: config.toml のみ作成
    run_id = "no_metrics_run"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    config_data = {"lr": 0.1, "_wandb": {"run_id": run_id}}
    toml.dump(config_data, open(run_dir / "config.toml", "w"))
    
    # 2. アクション
    run = MLRun.load(run_id=run_id, base_dir=str(tmp_path))
    
    # 3. 検証
    assert run.run_id == run_id
    assert run.config["lr"] == 0.1
    # metrics は空の辞書になるはず
    assert run.metrics == {}

def test_mlrun_load_not_found(tmp_path):
    """
    存在しない run_id を load しようとしたら FileNotFoundError が発生するかテスト
    """
    with pytest.raises(FileNotFoundError):
        MLRun.load(run_id="non_existent_id", base_dir=str(tmp_path))

def test_mlrun_load_no_metrics_file(tmp_path):
    """
    MLRun.load が metrics.toml が存在しなくてもエラーにならないかテスト
    """
    run_id = "no_metrics_run"
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    config_data = {
        "lr": 0.1, 
        "_wandb": {"run_id": run_id},
        "_meta": {"created_at": "2023-01-01T00:00:00+00:00"}
    }
    toml.dump(config_data, open(run_dir / "config.toml", "w"))
    
    run = MLRun.load(run_id=run_id, base_dir=str(tmp_path))
    
    assert run.run_id == run_id
    assert run.config["lr"] == 0.1
    assert run.metrics == {} # metrics は空の辞書
    assert run.created_at is not None # タイムスタンプはロードされる

def test_mlrun_load_not_found(tmp_path):
    """
    存在しない run_id を load しようとしたら FileNotFoundError が発生するかテスト
    """
    with pytest.raises(FileNotFoundError):
        MLRun.load(run_id="non_existent_id", base_dir=str(tmp_path))


# --- MLProject クラスのテスト -------------------------------------------------

# --- 変更点 5: タイムスタンプの型チェックを project load に追加 ---
def test_project_load_and_timestamp_type(populated_project_dir):
    """
    MLProject が DataFrame にロードし、タイムスタンプカラムが
    正しい型 (datetime64) になっているかテスト
    """
    # 1. アクション
    base_dir, _ = populated_project_dir # ディレクトリパスのみ使用
    project = MLProject(base_dir=base_dir)

    # 2. 検証 (基本)
    df = project.df
    assert len(project) == 3
    assert len(df) == 3
    
    # config, metrics カラムのロード確認
    assert "lr" in df.columns
    assert "model.name" in df.columns
    assert "accuracy" in df.columns
    
    # 3. 検証 (★ タイムスタンプ)
    assert "_meta.created_at" in df.columns
    
    # カラムの型が datetime64 (pandas の datetime 型) かチェック
    assert ptypes.is_datetime64_any_dtype(df["_meta.created_at"])
    
    # タイムスタンプが None (NaT) でないことを確認
    assert not df["_meta.created_at"].isnull().any()

def test_project_load_empty(tmp_path):
    """
    空のディレクトリで MLProject を初期化してもエラーにならないかテスト
    """
    project = MLProject(base_dir=str(tmp_path))
    assert len(project) == 0
    assert project.df.empty

def test_project_search_by_metric(populated_project_dir):
    """
    メトリクス (accuracy) で検索できるかテスト
    """
    base_dir, _ = populated_project_dir
    project = MLProject(base_dir=base_dir)
    results_df = project.search(query_string="accuracy > 0.92")
    assert len(results_df) == 1
    assert results_df.iloc[0]["run_id"] == "run_bert_1"

def test_project_search_by_nested_config(populated_project_dir):
    """
    ネストした config (model.name) で検索できるかテスト
    """
    base_dir, _ = populated_project_dir
    project = MLProject(base_dir=base_dir)
    results_df = project.search(query_string="`model.name` == 'CNN'")
    assert len(results_df) == 2

# --- 変更点 6: タイムスタンプによる検索テストを新設 ---
def test_project_search_by_timestamp(populated_project_dir):
    """
    タイムスタンプ (文字列比較) で検索できるかテスト
    """
    # 1. 準備
    base_dir, run2_timestamp = populated_project_dir
    project = MLProject(base_dir=base_dir)
    
    # 2. アクション
    # run2 (BERT) よりも *後* に作成された run を検索 (run3 のはず)
    # pandas.query では、datetime オブジェクトは文字列としてクエリできる
    query_str = f"`_meta.created_at` > '{run2_timestamp.isoformat()}'"
    
    results_df = project.search(query_string=query_str)
    
    # 3. 検証
    assert len(results_df) == 1
    assert results_df.iloc[0]["run_id"] == "run_cnn_2"

    # 4. アクション (逆のクエリ)
    # run2 (BERT) よりも *前* に作成された run を検索 (run1 のはず)
    query_str_before = f"`_meta.created_at` < '{run2_timestamp.isoformat()}'"
    results_df_before = project.search(query_string=query_str_before)
    
    # 5. 検証
    assert len(results_df_before) == 1
    assert results_df_before.iloc[0]["run_id"] == "run_cnn_1"

def test_project_search_combined(populated_project_dir):
    """
    config とメトリクスを組み合わせた検索のテスト
    """
    base_dir, _ = populated_project_dir
    project = MLProject(base_dir=base_dir)
    
    # CNN モデルかつ accuracy が 0.85 以上 (run_cnn_1 のはず)
    query = "`model.name` == 'CNN' and accuracy > 0.85"
    results_df = project.search(query_string=query)
    
    assert len(results_df) == 1
    assert results_df.iloc[0]["run_id"] == "run_cnn_1"

def test_project_search_return_objects(populated_project_dir):
    """
    return_objects=True が MLRun のリストを返すかテスト
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
    get_run が run_id から正しく MLRun をロードするかテスト
    """
    base_dir, _ = populated_project_dir
    project = MLProject(base_dir=base_dir)
    
    run = project.get_run("run_bert_1")
    
    assert isinstance(run, MLRun)
    assert run.metrics["accuracy"] == 0.95
    assert run.config["model"]["name"] == "BERT"
    # タイムスタンプもロードされているはず
    assert isinstance(run.created_at, datetime.datetime)