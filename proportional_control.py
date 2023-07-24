import sys
from pathlib import Path
from functools import partial

import pandas as pd
import numpy as np

import petsc4py

from petsc4py import PETSc
from scipy.integrate import trapezoid

from ns2d.utils.pyconfig import Config
from ns2d.utils.export import ObsExporter


from solver import NSSolver

COMM = PETSc.COMM_WORLD
RANK = COMM.Get_rank()
CFG_PATH = Path("config.yml").resolve()
MAIN_CFG = Config.from_yaml(config_path=CFG_PATH)
RESULTS_DIR_PATH = Path(MAIN_CFG.export_info.results_dirpath).resolve()
PROP_GAIN = 0.01
U_TARGET = -1

if __name__ == "__main__":
    petsc4py.init(sys.argv, comm=COMM)

    obs_exporter = ObsExporter()
    obs_exporter.add_exported_param(name=["swimming_frequency","x", "y", "velocity_x", "velocity_y", "Cx", "Cy"])

    ns_solver = NSSolver(main_cfg=MAIN_CFG, comm=COMM, obs_exporter=obs_exporter)
    ns_solver.add_exported_vec(vec=ns_solver.fields.current_time.local_ux, vector_name="ux")
    ns_solver.add_exported_vec(vec=ns_solver.fields.current_time.local_uy, vector_name="uy")
    ns_solver.add_exported_vec(vec=ns_solver.fields.current_time.local_pressure, vector_name="p")
    ns_solver.add_exported_vec(vec=ns_solver.fields.solid_ls[1].local_levelset_solid, vector_name="solid_level_set")
    
    frequency = 2.0
    freq_list = []
    time_list = [ns_solver.simu_wrap.time]
    ns_solver.results_dir_path = RESULTS_DIR_PATH
    fmin = 0
    fmax = 4

    while (ns_solver.simu_wrap.time < ns_solver.simu_wrap.time_final):
        u_swim  = ns_solver.obs_wrap.velocity_x[0]
        error = u_swim - U_TARGET
        frequency += PROP_GAIN*error
        if frequency < fmin: frequency = fmin
        if frequency > fmax: frequency = fmax
        freq_list.append(frequency)
        phase = trapezoid(y=freq_list, x=time_list)
        PETSc.Sys.Print(f"phase = {2*np.pi*phase}", comm=COMM)
        PETSc.Sys.Print(f"frequency = {frequency}", comm=COMM)
        ns_solver.step(phase=phase)
        time_list.append(ns_solver.simu_wrap.time)
    
    obs_num = ns_solver.obs_wrap.obstacles_number
    obs_data = obs_exporter.obs_data
    columns = ["time", "frequency", "x", "y", "Ux", "Uy", "Fx", "Fy"]

    if RANK==0:
        data_frames = []
        for i in range(obs_num):
            df = pd.DataFrame(np.array(obs_data[i]), columns=columns)
            data_frames.append(df)
            df.to_hdf(RESULTS_DIR_PATH / f"obstacles.hdf5", key=f"obstacle{i+1}", mode="a")
    
