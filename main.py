import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import petsc4py
from ns2d.utils.pyconfig import Config
from petsc4py import PETSc
from scipy.interpolate import PchipInterpolator

from helpers import ObsExporter
from solver import NSSolver

COMM = PETSc.COMM_WORLD
RANK = COMM.Get_rank()
CFG_PATH = Path("config.yml").resolve()
RESULTS_DIR_PATH = Path("results/").resolve()
MAIN_CFG = Config.from_yaml(config_path=CFG_PATH)

def objective_function(frequencies: np.ndarray, ns_solver: NSSolver, u_ref: float):
    dim = frequencies.shape[0]
    fmin = 0
    fmax = 3
    lbda1 = 10
    lbda2 = 10
    t_final = ns_solver.simu_wrap.time_final
    t_knots = np.linspace(0, t_final, dim)
    time_spline_func = PchipInterpolator(x=t_knots, y=frequencies, extrapolate=None)

    ns_solver.reset()

    for tf in t_knots[1:]:
        ns_solver.simulate_time_seg(t_final=tf, interpolator=time_spline_func)
        if not ns_solver.in_bounds:
            break
    
    obs_data_array = np.array(ns_solver.obs_exporter.obs_data[0])
    #get velocity array
    ux_t = obs_data_array[:, 4]

    obj_func = np.linalg.norm(ux_t-u_ref) + lbda1*np.maximum(0, frequencies-fmax)**2 + lbda2*np.minimum(0, frequencies-fmin)**2

    return obj_func
    


if __name__ == "__main__":
    petsc4py.init(sys.argv, comm=COMM)

    obs_exporter = ObsExporter()
    obs_exporter.add_exported_param(name=["swimming_frequency","x", "y", "velocity_x", "velocity_y", "Cx", "Cy"])

    ns_solver = NSSolver(main_cfg=MAIN_CFG, results_dir_path=RESULTS_DIR_PATH, obs_exporter=obs_exporter)
    ns_solver.add_exported_vec(vec=ns_solver.fields.current_time.local_ux, vector_name="ux")
    ns_solver.add_exported_vec(vec=ns_solver.fields.current_time.local_uy, vector_name="uy")
    ns_solver.add_exported_vec(vec=ns_solver.fields.current_time.local_pressure, vector_name="p")
    ns_solver.add_exported_vec(vec=ns_solver.fields.solid_ls[1].local_levelset_solid, vector_name="solid_level_set")

    # time_segments = 12

    # t_knots = np.linspace(0, 12, time_segments+1)
    frequencies = np.array([0, 0.75, 1, 1, 1, 1, 1])

    # time_spline_func = PchipInterpolator(x=t_knots, y=frequencies, extrapolate=None)

    # for tf in t_knots[1:]:
    #     ns_solver.simulate_time_seg(t_final=tf, interpolator=time_spline_func)

    _ = objective_function(frequencies=frequencies, ns_solver=ns_solver, u_ref=-1)
    
    obs_num = ns_solver.obs_wrap.obstacles_number
    obs_data = obs_exporter.obs_data
    columns = ["time", "frequency", "x", "y", "Ux", "Uy", "Fx", "Fy"]

    if RANK==0:
        data_frames = []
        for i in range(obs_num):
            df = pd.DataFrame(np.array(obs_data[i]), columns=columns)
            data_frames.append(df)
            df.to_hdf(RESULTS_DIR_PATH / f"obstacles.hdf5", key=f"obstacle{i+1}", mode="a")
    
