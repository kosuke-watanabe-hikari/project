#初期値グリッド生成関数
import itertools

def make_init_sets(init_grid):
    """
    初期値グリッドから全ての初期値組み合わせを生成する

    Parameters
    ----------
    init_grid : dict
        初期値のグリッド。
        例: {"x0": [1, 2], "y0": [3, 4]}

    Returns
    -------
    init_names : list of str
        初期値の名前（キーの順序）
    init_values : list of list
        各初期値の候補リスト
    init_sets : list of tuple
        初期値の全組み合わせ
        例: [(1,3), (1,4), (2,3), (2,4)]
    """
    init_names = list(init_grid.keys())
    init_values = list(init_grid.values())
    init_sets = list(itertools.product(*init_values))
    return init_names, init_values, init_sets

#パラメータセット生成関数
import itertools

def make_param_sets(param_grid):
    """
    パラメータグリッドから全てのパラメータ組み合わせを生成する

    Parameters
    ----------
    param_grid : dict
        パラメータのグリッド。
        例: {"a": [1, 2], "b": [0.1, 0.2]}

    Returns
    -------
    param_names : list of str
        パラメータ名（キーの順序）
    param_values : list of list
        各パラメータの候補リスト
    param_sets : list of tuple
        パラメータの全組み合わせ
    例: [(1,0.1), (1,0.2), (2,0.1), (2,0.2)]
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_sets = list(itertools.product(*param_values))

    return param_names, param_values, param_sets

#オイラー法関数
import numpy as np

def euler_2d(f, g, params, x0, y0, t_max, dt):
    """
    2次元常微分方程式をオイラー法で数値積分する

    dx/dt = f(x, y, params)
    dy/dt = g(x, y, params)

    Parameters
    ----------
    f, g : callable
        状態変数 (x, y) と params を引数に取り、
        それぞれ dx/dt, dy/dt を返す関数
    params : tuple or dict
        系のパラメータ
    x0, y0 : float
        初期値
    t_max : float
        積分の終了時刻
    dt : float
        時間刻み幅

    Returns
    -------
    t : numpy.ndarray
        時間配列（0 から t_max まで）
    traj : numpy.ndarray, shape (N, 2)
        軌道データ。
        traj[:, 0] が x(t), traj[:, 1] が y(t)
    """
    t = np.arange(0, t_max + dt, dt)
    N = len(t)

    traj = np.zeros((N, 2))
    traj[0, 0] = x0
    traj[0, 1] = y0

    for i in range(N - 1):
        x, y = traj[i]
        traj[i+1, 0] = x + dt * f(x, y, params)
        traj[i+1, 1] = y + dt * g(x, y, params)

    return t, traj

#パラメータ探索ループ関数
def run_param_loop(f, g, param_sets, init_sets, t_max, dt):
    """
    パラメータ集合と初期値集合の全組み合わせに対して
    2次元常微分方程式を数値積分し，結果をまとめて返す

    Parameters
    ----------
    f, g : callable
        状態変数 (x, y) と params を引数に取り，
        それぞれ dx/dt, dy/dt を返す関数
    param_sets : list of tuple
        探索するパラメータの組み合わせ
    （make_param_sets の出力）
    init_sets : list of tuple
        探索する初期値の組み合わせ
        （make_init_sets の出力）
    t_max : float
        積分の終了時刻
    dt : float
        時間刻み幅
    Returns
    -------
    results : list of dict
        各計算結果をまとめたリスト。
        各要素は以下のキーを持つ辞書：
            - "params" : パラメータセット
            - "x0", "y0" : 初期値
            - "dt", "t_max" : 数値積分条件
            - "traj" : 軌道データ (N, 2)
    """
    results = []
    for p in param_sets:
        for x0, y0 in init_sets:
            t, traj = euler_2d(
                f=f,
                g=g,
                params=p,
                x0=x0,
                y0=y0,
                dt=dt,
                t_max=t_max,
            )

            results.append({
                "params": p,
                "x0": x0,
                "y0": y0,
                "traj": traj
            })
    return results

#結果ファイルの出力関数
import pickle
from pathlib import Path

def save_experiment(
    results,
    param_grid,
    init_grid,
    model_name,
    exp_number,
    dt,
    t_max,
    project_name="project",
    out_dir="results"
):
    """
    Notebook がどの階層で実行されても、
    project直下に results フォルダを作る

    Parameters
    ----------
    results : list of dict
        run_param_loop の出力
    params_grid : dict
        使用したパラメータグリッド
    init_grid : dict
        使用した初期値グリッド
    model_name : str
        数理モデル名(ディレクトリ名に使用)
    exp_number : int
        実験番号（必須）
    t_max : float
        積分の終了時刻
    dt : float
        時間刻み幅
    out_dir : str or Path, optional
        保存先親ディレクトリ（default: "results"）
    """
    cwd = Path.cwd()
    
    #project フォルダのパスを取得
    try:
        project_index = cwd.parts.index(project_name)
    except ValueError:
        raise RuntimeError(f"{project_name}フォルダがカレントディレクトリのパスに見つかりません。")
    
    #project直下のパスを構築
    project_dir = Path(*cwd.parts[: project_index + 1])
    
    #project/results/model_name
    base_dir =  project_dir / out_dir
    model_dir = base_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    exp_tag = f"{model_name}_experiment_{exp_number:03}"

    #results
    results_path = model_dir / f"{exp_tag}_results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)

    #meta
    meta = {
        "model_name": model_name,
        "exp_number": exp_number,
        "dt": dt,
        "t_max": t_max,
        "param_grid": param_grid,
        "init_grid": init_grid,
    }

    meta_path = model_dir / f"{exp_tag}_meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(meta,f)

    print(f"Saved results -> {results_path}")
    print(f"Saved meta -> {meta_path}")