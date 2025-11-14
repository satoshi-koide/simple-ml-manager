import wandb
import toml
import os
import glob
import pandas as pd
from typing import Dict, Any, Optional, List
import datetime  # <<< タイムスタンプのために追加

# --- MLRun クラス ---

class MLRun:
    """単一の機械学習 run を管理するクラス"""

    def __init__(
        self,
        run_id: str,
        config: Dict[str, Any],  # 純粋なユーザーconfig
        run_dir: str,
        metrics: Dict[str, Any] = None,
        # --- 変更点 1: メタデータを引数で受け取る ---
        created_at: Optional[datetime.datetime] = None,
        wandb_entity: Optional[str] = None,
        wandb_project: Optional[str] = None,
    ):
        self.run_id = run_id
        self.config = config  # ユーザーが指定したオリジナルの config
        self.run_dir = run_dir
        self.metrics = metrics if metrics is not None else {}
        self.wandb_run = None  # wandb.init() が返すアクティブな run オブジェクト

        # --- 変更点 2: メタデータをインスタンス変数として保持 ---
        self.created_at = created_at
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project

    @classmethod
    def create(
        cls,
        config: Dict[str, Any],
        base_dir: str,
        project_name: str,
        entity: Optional[str] = None,
    ) -> "MLRun":
        """
        新しい run を作成し、wandb.init() を呼び出し、ローカルに config を保存する。
        """
        print(f"Creating new run in project '{project_name}'...")
        # 1. wandb を初期化
        wandb_run = wandb.init(
            project=project_name,
            entity=entity,
            config=config,
        )

        # --- 変更点 3: タイムスタンプ (UTC) を生成 ---
        created_at_time = datetime.datetime.now(datetime.timezone.utc)
        created_at_str = created_at_time.isoformat()

        # 2. wandb から情報を取得
        run_id = wandb_run.id
        wandb_entity = wandb_run.entity
        wandb_project = wandb_run.project
        run_dir = os.path.join(base_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # 3. config.toml にメタデータも一緒に保存
        full_config = config.copy()
        # wandb メタデータ
        full_config["_wandb"] = {
            "entity": wandb_entity,
            "project": wandb_project,
            "run_id": run_id,
        }
        # --- 変更点 4: タイムスタンプ用メタデータを追加 ---
        full_config["_meta"] = {
            "created_at": created_at_str
        }

        config_path = os.path.join(run_dir, "config.toml")
        with open(config_path, "w") as f:
            toml.dump(full_config, f)
        print(f"Run {run_id} created. Config saved to {config_path}")

        # 4. MLRun インスタンスを作成
        instance = cls(
            run_id=run_id,
            config=config, # 純粋な config
            run_dir=run_dir,
            metrics={},
            created_at=created_at_time, # --- 変更点 5: datetime オブジェクトを渡す ---
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
        )
        instance.wandb_run = wandb_run
        return instance

    @classmethod
    def load(cls, run_id: str, base_dir: str) -> "MLRun":
        """
        既存の run_id からローカルの config.toml と metrics.toml を読み込む。
        """
        run_dir = os.path.join(base_dir, run_id)
        config_path = os.path.join(run_dir, "config.toml")

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"config.toml not found for run_id {run_id} in {base_dir}"
            )

        with open(config_path, "r") as f:
            full_config = toml.load(f)

        # --- 変更点 6: メタデータを抽出しつつ、config から削除 ---
        wandb_info = full_config.pop("_wandb", {})
        meta_info = full_config.pop("_meta", {})
        
        # 残ったものがオリジナルの config
        user_config = full_config

        # タイムスタンプをパース
        created_at_obj = None
        created_at_str = meta_info.get("created_at")
        if created_at_str:
            try:
                # ISO 形式から datetime オブジェクトに変換
                created_at_obj = datetime.datetime.fromisoformat(created_at_str)
            except ValueError:
                print(f"Warning (Run {run_id}): Could not parse created_at string: {created_at_str}")

        # metrics.toml もロードする
        metrics_path = os.path.join(run_dir, "metrics.toml")
        metrics = {}
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, "r") as f:
                    metrics = toml.load(f)
            except Exception as e:
                print(f"Warning: Could not load metrics {metrics_path}: {e}")

        # --- 変更点 7: 抽出したメタデータをコンストラクタに渡す ---
        return cls(
            run_id=run_id,
            config=user_config,
            run_dir=run_dir,
            metrics=metrics,
            created_at=created_at_obj,
            wandb_entity=wandb_info.get("entity"),
            wandb_project=wandb_info.get("project"),
        )

    def add_metrics(self, metrics_dict: Dict[str, Any]):
        """
        メトリクスを辞書で登録し、ローカルの metrics.toml に保存する。
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
        """アクティブな wandb run を終了する"""
        if self.wandb_run:
            self.wandb_run.finish()
            self.wandb_run = None
            print(f"Run {self.run_id} finished.")
        else:
            print(f"Run {self.run_id} was not active. No need to finish.")

    def get_wandb_url(self) -> str:
        """wandb ダッシュボードへの URL を返す"""
        if self.wandb_entity and self.wandb_project:
            return f"https://wandb.ai/{self.wandb_entity}/{self.wandb_project}/runs/{self.run_id}"
        else:
            return "Could not determine wandb URL (entity or project missing)."

    def __repr__(self):
        ts_str = self.created_at.strftime('%Y-%m-%d %H:%M') if self.created_at else 'UnknownTime'
        return f"<MLRun (id={self.run_id}, created={ts_str})>"


# --- MLProject クラス ---

class MLProject:
    """複数の MLRun を管理するプロジェクトクラス"""

    def __init__(self, base_dir: str = "./checkpoints"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True) # フォルダがなければ作成
        self.df = pd.DataFrame()
        self.load_project() # 初期化時にロード

    def load_project(self):
        """
        base_dir 内のすべての config.toml と metrics.toml を読み込み、
        マージして DataFrame に変換する。
        """
        config_paths = glob.glob(os.path.join(self.base_dir, "*", "config.toml"))
        
        all_data = []

        for path in config_paths:
            run_id = os.path.basename(os.path.dirname(path))
            run_dir = os.path.dirname(path)
            
            try:
                # 1. config.toml をロード (これに _wandb, _meta が含まれる)
                with open(path, "r") as f:
                    data = toml.load(path)
                
                data["run_id"] = run_id
                data["run_dir"] = run_dir

                # 2. 対応する metrics.toml もロードしてマージ
                metrics_path = os.path.join(run_dir, "metrics.toml")
                if os.path.exists(metrics_path):
                    try:
                        with open(metrics_path, "r") as f:
                            metrics = toml.load(metrics_path)
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
            # 3. DataFrame を構築
            # (ネストしたキーは 'model.name' や '_meta.created_at' に展開)
            self.df = pd.json_normalize(all_data, sep=".")
            
            # --- 変更点 8: タイムスタンプカラムを datetime 型に変換 ---
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
        DataFrame をクエリ文字列で検索する。
        (タイムスタンプでのクエリ例:
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
            print("ヒント: ネストしたキー (例: 'model.name', '_meta.created_at') は")
            print("バッククォートで囲んでください: `model.name` == 'bert'")
            print("メトリクス (例: 'accuracy > 0.9') はそのまま検索できます。")
            print("---")
            return pd.DataFrame() if not return_objects else []

        # (Bonus) wandb_url カラムを DataFrame に追加
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

    def get_run(self, run_id: str) -> MLRun:
        """run_id を指定して MLRun オブジェクトをロードする"""
        return MLRun.load(run_id, self.base_dir)

    def __len__(self):
        return len(self.df)

    def __repr__(self):
        return f"<MLProject (path={self.base_dir}, runs={len(self)})>"

# =============================================================================
# 🚀 実行例 (使い方)
# =============================================================================
if __name__ == "__main__":

    # (wandb にログインしている前提)
    # wandb.login() 

    import shutil
    import time
    DEMO_DIR = "./checkpoints_demo"
    
    # --- 1. テスト用のクリーンアップ ---
    if os.path.exists(DEMO_DIR):
        print(f"Cleaning up old demo directory: {DEMO_DIR}\n")
        shutil.rmtree(DEMO_DIR)

    # --- 2. プロジェクトの準備 ---
    WANDB_ENTITY = "causal-rl" # ★ ご自身の wandb entity に変更
    WANDB_PROJECT = "mlproject-demo-v2"
    
    # --- 3. 実験 1 (CNN) を実行 ---
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
    print(f"Run 1 Object: {run1}") # __repr__ の確認
    
    run1.add_metrics({"accuracy": 0.92, "f1_score": 0.91, "epoch": 10})
    run1.finish()

    # --- 4. 実験 2 (BERT) を実行 (数秒待機) ---
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

    # --- 5. プロジェクトをロードしてメトリクスとタイムスタンプを確認 ---
    print("\n" + "="*30)
    print("--- Loading Project and Checking DataFrame ---")
    
    project = MLProject(base_dir=DEMO_DIR)
    
    print("\n[DataFrame with Metrics and Timestamps]")
    # '_meta.created_at' カラムが追加されていることを確認
    # (表示するカラムが多すぎる場合は、関連するカラムのみ選択)
    display_cols = [
        "run_id", 
        "_meta.created_at", 
        "model.name", 
        "accuracy", 
        "learning_rate"
    ]
    # df.columns に存在するカラムのみ表示
    display_cols = [col for col in display_cols if col in project.df.columns]
    
    print(project.df[display_cols].to_markdown(index=False))

    # --- 6. タイムスタンプでソート ---
    print("\n" + "="*30)
    print("--- Sorting by Timestamp (DESC) ---")
    
    # datetime 型になっているため、正しくソートできる
    sorted_df = project.df.sort_values(by="_meta.created_at", ascending=False)
    print(sorted_df[display_cols].to_markdown(index=False))

    # --- 7. タイムスタンプで検索 ---
    print("\n" + "="*30)
    print("--- Searching by Timestamp (run2 のみ) ---")
    
    # run1 と run2 の中間時刻を取得 (簡易的)
    if run1.created_at and run2.created_at:
        mid_time = run1.created_at + (run2.created_at - run1.created_at) / 2
        mid_time_str = mid_time.isoformat() # ISO 文字列でクエリ
        
        query = f"`_meta.created_at` > '{mid_time_str}'"
        print(f"Query: {query}")
        
        results = project.search(query_string=query)
        print(results[display_cols].to_markdown(index=False))