import sys
import shutil
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import petsc4py
from ns2d.utils.pyconfig import Config
from petsc4py import PETSc
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
import scipy.integrate as integrate


from helpers import ObsExporter
from solver import NSSolver

COMM = PETSc.COMM_WORLD
RANK = COMM.Get_rank()
CFG_PATH = Path("config.yml").resolve()
MAIN_CFG = Config.from_yaml(config_path=CFG_PATH)
RESULTS_DIR_PATH = Path(MAIN_CFG.export_info.results_dirpath).resolve()
FUNC_EVAL_COUNTER = 0

def objective_function(frequencies: np.ndarray, ns_solver: NSSolver, u_ref: float):
    SIM_PATH = RESULTS_DIR_PATH / f"simu{objective_function.counter}"
    ns_solver.results_dir_path = SIM_PATH
    shutil.copyfile("config.yml", SIM_PATH / "config.yml")
    t_final = ns_solver.simu_wrap.time_final
    
    dim = frequencies.shape[0]
    t_knots = np.linspace(0, t_final, dim)
    time_spline_func = PchipInterpolator(x=t_knots, y=frequencies, extrapolate=None)

    def time_spline_func_int(t):
        y, _ = integrate.quad(time_spline_func, 0, t)
        return y

    if RANK == 0:
        plt.clf()
        t_plot = np.linspace(0, t_final, 100)
        plt.plot(t_plot, time_spline_func(t_plot))
        plt.savefig(SIM_PATH / "freq.png")

    ns_solver.reset(objective_function.counter)

    PETSc.Sys.Print("Simulation Start !", comm=PETSc.COMM_WORLD)

    for tf in t_knots[1:]:
        PETSc.Sys.Print("Time segment Start !", comm=PETSc.COMM_WORLD)
        ns_solver.simulate_time_seg(t_final=tf, interpolator=time_spline_func_int)
        PETSc.Sys.Print("Time segment Done !", comm=PETSc.COMM_WORLD)
        # if not ns_solver.in_bounds:
        #     break
    
    PETSc.Sys.Print("Simulation Done !", comm=PETSc.COMM_WORLD)
    
    obs_num = ns_solver.obs_wrap.obstacles_number
    obs_data = obs_exporter.obs_data
    columns = ["time", "frequency", "x", "y", "Ux", "Uy", "Fx", "Fy"]

    if RANK==0:
        data_frames = []
        for i in range(obs_num):
            df = pd.DataFrame(np.array(obs_data[i]), columns=columns)
            data_frames.append(df)
            df.to_hdf(SIM_PATH / f"obstacles.hdf5", key=f"obstacle{i+1}", mode="a")
    
    obs_data_array = np.array(ns_solver.obs_exporter.obs_data[0])
    #get velocity array
    ux_t = obs_data_array[:, 4]
    n_t = ux_t.shape[0]

    # lbda1*np.sum(np.maximum(0, frequencies-fmax)**2) + lbda2*np.sum(np.minimum(0, frequencies-fmin)**2)

    obj_func = (1/n_t)*np.linalg.norm(ux_t-u_ref)

    objective_function.counter += 1

    return obj_func
    
objective_function.counter = 0

if __name__ == "__main__":
    petsc4py.init(sys.argv, comm=COMM)

    obs_exporter = ObsExporter()
    obs_exporter.add_exported_param(name=["swimming_frequency","x", "y", "velocity_x", "velocity_y", "Cx", "Cy"])

    ns_solver = NSSolver(main_cfg=MAIN_CFG, comm=COMM, obs_exporter=obs_exporter)
    ns_solver.add_exported_vec(vec=ns_solver.fields.current_time.local_ux, vector_name="ux")
    ns_solver.add_exported_vec(vec=ns_solver.fields.current_time.local_uy, vector_name="uy")
    ns_solver.add_exported_vec(vec=ns_solver.fields.current_time.local_pressure, vector_name="p")
    ns_solver.add_exported_vec(vec=ns_solver.fields.solid_ls[1].local_levelset_solid, vector_name="solid_level_set")


    # frequencies = np.array([1, 1, 1, 1, 1, 1, 1])

    # _ = objective_function(frequencies=frequencies, ns_solver=ns_solver, u_ref=-1)

    # ns_solver.fields_exporter.export_vectors(RESULTS_DIR_PATH / "solution.pvtr", timestep=0)
    t_nodes = 7
    x0 = np.zeros(t_nodes)
    fmin = 0
    fmax = 5

    eps = 0.2

    initial_simplex = np.array([
        [2, 2, 2, 2, 2, 2, 2],
        [2+eps, 2, 2, 2, 2 ,2, 2],
        [2, 2+eps, 2, 2, 2 ,2, 2],
        [2, 2, 2+eps, 2, 2 ,2, 2],
        [2, 2, 2, 2+eps, 2 ,2, 2],
        [2, 2, 2, 2, 2+eps ,2, 2],
        [2, 2, 2, 2, 2 ,2+eps, 2],
        [2, 2, 2, 2, 2 ,2, 2+eps]
    ])

    bounds = [(fmin, fmax) for _ in range(t_nodes)]


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


    
    
    
