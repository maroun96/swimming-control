import sys
import shutil
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import petsc4py

from petsc4py import PETSc
from scipy.optimize import minimize
from scipy.integrate import trapezoid

from ns2d.utils.pyconfig import Config
from ns2d.utils.export import ObsExporter

from helpers import SlidingWindow, rbk_interpolation, get_phase
from solver import NSSolver

COMM = PETSc.COMM_WORLD
RANK = COMM.Get_rank()
CFG_PATH = Path("config.yml").resolve()
MAIN_CFG = Config.from_yaml(config_path=CFG_PATH)
RESULTS_DIR_PATH = Path(MAIN_CFG.export_info.results_dirpath).resolve()
CENTROIDS_PATH = Path("/scratch/kmaroun/clustering/simu_f_2.0/centroids.npy").resolve()
WINDOW_LENGTH = 0.2

if __name__ == "__main__":
    petsc4py.init(sys.argv, comm=COMM)

    obs_exporter = ObsExporter()
    obs_exporter.add_exported_param(name=[
        "swimming_frequency",
        "x",
        "y",
        "velocity_x",
        "velocity_y",
        "Cx",
        "Cy",
        "density",
        "surface_area"
    ])

    window = SlidingWindow(window_length=WINDOW_LENGTH, init_zero=True)
    window.add_data(name="Ux")

    ns_solver = NSSolver(main_cfg=MAIN_CFG, comm=COMM)
    ns_solver.add_exported_vec(vec=ns_solver.fields.current_time.local_ux, vector_name="ux")
    ns_solver.add_exported_vec(vec=ns_solver.fields.current_time.local_uy, vector_name="uy")
    ns_solver.add_exported_vec(vec=ns_solver.fields.current_time.local_pressure, vector_name="p")
    ns_solver.add_exported_vec(vec=ns_solver.fields.solid_ls[1].local_levelset_solid, vector_name="solid_level_set")

    centroids = np.load(CENTROIDS_PATH)
    c_dim = centroids.shape[0]


    def objective_function(frequencies: np.ndarray, ns_solver: NSSolver, u_ref: float):
        SIM_PATH = RESULTS_DIR_PATH / f"simu{objective_function.counter}"
        ns_solver.results_dir_path = SIM_PATH
        shutil.copyfile("config.yml", SIM_PATH / "config.yml")

        if RANK == 0:
            fig, ax = plt.subplots()
            x = [1, 2, 3, 4]
            labels = [str(np.around(i, 2)) for i in centroids]
            ax.plot(x, frequencies, linestyle="", marker="x", markersize=10)
            ax.set_xticks(x, labels)
            fig.savefig(SIM_PATH / "control_vec.png")

        window.reset()
        obs_exporter.clear()
        ns_solver.reset(func_count=objective_function.counter)

    
        while (ns_solver.simu_wrap.time < ns_solver.simu_wrap.time_final):
            state = window.data_set_avg["Ux"][-1]
            freq = rbk_interpolation(state=state, control_vector=frequencies, centroids=centroids)
            PETSc.Sys.Print(f"freq = {freq}")
            PETSc.Sys.Print(f"state = {state}")
            ns_solver.obs_wrap.swimming_frequency = (freq, 1)
            obs_exporter.append_obs_data(ns_solver.simu_wrap, ns_solver.obs_wrap)
            phase = get_phase(obs_exporter=obs_exporter)

            ns_solver.step(phase=phase)

            t = ns_solver.simu_wrap.time
            Ux = ns_solver.obs_wrap.velocity_x[0]
            other_data = {"Ux": Ux}
            window.append(t=t, other_data=other_data)
    
        obs_num = ns_solver.obs_wrap.obstacles_number
        obs_data = obs_exporter.obs_data
        columns = ["time", "frequency", "x", "y", "Ux", "Uy", "Fx", "Fy", "density", "surface_area"]

        if RANK==0:
            data_frames = []
            for i in range(obs_num):
                df = pd.DataFrame(np.array(obs_data[i]), columns=columns)
                data_frames.append(df)
                df.to_hdf(SIM_PATH / f"obstacles.hdf5", key=f"obstacle{i+1}", mode="a")

            with open(SIM_PATH / "ux_avg.pkl", "wb") as f:
                pickle.dump(window.data_set_avg["Ux"], f)
            
    
            df = data_frames[0]
            fig, ax = plt.subplots()
            ax.plot(df["time"], df["frequency"])
            fig.savefig(SIM_PATH / "freq.png")

            fig, ax = plt.subplots()
            ax.plot(df["time"], df["Ux"].abs())
            fig.savefig(SIM_PATH / "ux.png")
    
        obs_data_array = np.array(obs_exporter.obs_data[0])
        ux_t = obs_data_array[:, 4]
        n_t = ux_t.shape[0]


        obj_func = (1/n_t)*np.linalg.norm(ux_t-u_ref)

        
        objective_function.counter += 1

        return obj_func
    
    objective_function.counter = 0

    x0 = np.zeros((c_dim,)) #not used by the simplex optimizer
    eps = 1.0
    initial_simplex = np.ones((c_dim+1, c_dim))*2
    initial_simplex[1:][np.diag_indices(n=c_dim)] += eps

    fmin = 0
    fmax = 5
    
    bounds = [(fmin, fmax) for _ in range(x0.shape[0])]


    opt_res = minimize(
        objective_function,
        x0=x0,
        args=(ns_solver, -1),
        method="Nelder-Mead",
        bounds=bounds,
        options={
            "maxfev": 60,
            "adaptive": True,
            "initial_simplex": initial_simplex
        }
    )

    if RANK==0:
        with open(RESULTS_DIR_PATH / "optimal_result.pkl", "wb"):
            pickle.dump(opt_res)