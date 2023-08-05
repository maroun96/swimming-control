import argparse
import pickle

from pathlib import Path
from statistics import mean

import numpy as np
import pysindy as ps
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import mean_squared_error

from ns2d.utils.pyconfig import Config

from helpers import parse_sim_dir

def train(train_trajectories, train_controls, train_times, threshold, degree=3):
    feature_names = ["u", "f"]
    poly_library = ps.PolynomialLibrary(degree=degree, include_interaction=False)
    
    alpha = 2.0
    sfd = ps.SINDyDerivative(kind="kalman", alpha=alpha)
    
    opt = ps.STLSQ(threshold=threshold)
    
    model = ps.SINDy(optimizer=opt, differentiation_method=sfd, feature_library=poly_library,
                 feature_names=feature_names)
    
    model.fit(x=train_trajectories, t=train_times, u=train_controls, multiple_trajectories=True)
    
    return model

def compute_test_errors(model, test_traj, test_control, test_time):
    x_dot_mse = model.score(x=test_traj, t=test_time, u=test_control, metric=mean_squared_error)

    x0 = [0]
    x_predicted = model.simulate(x0=x0, t=test_time, u=test_control)
    
    x_mse = mean_squared_error(test_traj[:-1], x_predicted)
    
    return (x_mse, x_dot_mse)

def loocv(trajectories, controls, times, thresholds):
    N_traj = len(trajectories)
    assert (len(controls) == N_traj) and (len(times) == N_traj)
    
    x_mse_avg_list = []
    x_dot_mse_avg_list = []
    
    for threshold in tqdm(thresholds):
        x_mse_list = []
        x_dot_mse_list = []
        
        for i in range(N_traj):
            test_traj, test_control, test_time = trajectories[i], controls[i], times[i]
            train_trajectories = [traj for indx,traj in enumerate(trajectories) if indx!=i]
            train_controls = [control for indx,control in enumerate(controls) if indx!=i]
            train_times = [t for indx,t in enumerate(times) if indx!=i]
            
            model = train(train_trajectories=train_trajectories, train_controls=train_controls,
                          train_times=train_times, threshold=threshold)
            
                        
            (x_mse, x_dot_mse) \
            = compute_test_errors(model=model, test_traj=test_traj, test_control=test_control,
                                  test_time=test_time)
            
            x_mse_list.append(x_mse)
            x_dot_mse_list.append(x_dot_mse)
        
        
        x_mse_avg_list.append(mean(x_mse_list))
        x_dot_mse_avg_list.append(mean(x_dot_mse_list))
        
        x_mse_list.clear()
        x_dot_mse_list.clear()
    
    return (x_mse_avg_list, x_dot_mse_avg_list)

def compute_aics(trajectories, controls, times, thresholds):
    N_traj = len(trajectories)
    assert (len(controls) == N_traj) and (len(times) == N_traj)
    
    aic_list = []
    best_model = None
    min_aic = 1000
    
    for threshold in tqdm(thresholds):
        model = train(train_trajectories=trajectories, train_controls=controls,
                          train_times=times, threshold=threshold)
        
        
        x_dot_mse = model.score(x=trajectories, t=times, u=controls, metric=mean_squared_error,
                               multiple_trajectories=True)

        k = np.nonzero(model.coefficients())[0].size
        
        aic = N_traj*np.log(x_dot_mse) + 2*k
         
        aic_list.append(aic)

        if aic < min_aic:
            min_aic = aic
            best_model = model

    
    return aic_list, best_model

def load_dataframes(result_dir: Path):
    df_dict = {}
    for sub_dir in result_dir.glob("*"):
        if not sub_dir.name.startswith("simu_f"):
            continue
        control_inputs = parse_sim_dir(sim_dir=sub_dir)
        df = pd.read_hdf(sub_dir / "obstacles.hdf5")
        df_dict.update({control_inputs: df})
    return df_dict

CFG_PATH = Path("config.yml").resolve()
MAIN_CFG = Config.from_yaml(config_path=CFG_PATH)
RESULTS_DIR_PATH = Path(MAIN_CFG.export_info.results_dirpath).resolve()

# plt.style.use('./presentation.mplstyle')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loocv", default=False, action="store_true", help="leave one out cross validation")
    parser.add_argument("--freq", default=False, action="store_true", help="Use frequency as control input")
    parser.add_argument("--amp", default=False, action="store_true", help="Use amplitude as control input")
    args = parser.parse_args()

    features_names = ["u"]

    assert (args.freq or args.amp) is True

    if args.freq: features_names.append("f")
    if args.amp: features_names.append("a")

    
    df_dict = load_dataframes(result_dir=RESULTS_DIR_PATH)

    trajectories = list(map(lambda df: df["Ux"].to_numpy().reshape(-1, 1), df_dict.values()))
    times = list(map(lambda df: df["time"].to_numpy(), df_dict.values()))
    if args.freq and not args.amp:
        controls = list(map(lambda df: df["frequency"].to_numpy().reshape(-1, 1), df_dict.values()))
    elif args.amp and not args.freq:
        controls = list(map(lambda df: df["amplitude"].to_numpy().reshape(-1, 1), df_dict.values()))
    elif args.freq and args.amp:
        controls = list(map(lambda df: np.concatenate((df["frequency"].to_numpy().reshape(-1, 1),\
         df["amplitude"].to_numpy().reshape(-1, 1)), axis=1), df_dict.values()))
    
    thresholds = np.arange(0, 0.16, 0.01).tolist()

    aic_list, best_model = compute_aics(trajectories=trajectories, controls=controls, times=times, thresholds=thresholds)


    best_model.print()

    with open("best_model.pkl", "wb") as f:
        pickle.dump(obj=best_model, file=f)

    fig, ax = plt.subplots()
    ax.plot(thresholds, aic_list, linestyle="--", marker="o")
    ax.set_xlabel("threshold")
    ax.set_ylabel("AIC")

    plt.tight_layout()
    plt.grid(linestyle="--")
    plt.savefig("test_model.png")
