import sys
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

import petsc4py
from petsc4py import PETSc

from ns2d.utils.pyconfig import Config
from ns2d.utils.export import ObsExporter

from solver import NSSolver
from helpers import get_phase, construct_iterator, smoothstep, get_cumul_energy

COMM = PETSc.COMM_WORLD
RANK = COMM.Get_rank()
CFG_PATH = Path("config.yml").resolve()
MAIN_CFG = Config.from_yaml(config_path=CFG_PATH)
RESULTS_DIR_PATH = Path(MAIN_CFG.export_info.results_dirpath).resolve()
T_SMOOTH = 0.2 #add to config

if __name__ == "__main__":
    petsc4py.init(sys.argv, comm=COMM)

    obs_exporter = ObsExporter()
    
    obs_exporter.add_exported_param(name=[
        "swimming_frequency",
        "maximum_amplitude",
        "x",
        "y",
        "velocity_x",
        "velocity_y",
        "Cx",
        "Cy",
        "power_def",
        "surface_area"
    ])

    ns_solver = NSSolver(main_cfg=MAIN_CFG, comm=COMM)
    ns_solver.add_exported_vec(vec=ns_solver.fields.current_time.local_ux, vector_name="ux")
    ns_solver.add_exported_vec(vec=ns_solver.fields.current_time.local_uy, vector_name="uy")
    ns_solver.add_exported_vec(vec=ns_solver.fields.current_time.local_pressure, vector_name="p")
    ns_solver.add_exported_vec(vec=ns_solver.fields.solid_ls[1].local_levelset_solid, vector_name="solid_level_set")

    freq_list = [1.55]
    amp_list = [0.5]

    ns_solver.simu_wrap.dt = 0.002

    iterator = construct_iterator(freq_list=freq_list, amp_list=amp_list)


    for it, (freq, amp) in enumerate(iterator):
        ns_solver.reset(func_count=it)
        obs_exporter.clear()
        sim_path = RESULTS_DIR_PATH / f"simu_f_{freq}_a_{amp}"
        ns_solver.results_dir_path = sim_path

        ed_list = []
    
        while (ns_solver.simu_wrap.time < MAIN_CFG.simu.time_final):
            ns_solver.obs_wrap.swimming_frequency = (freq, 1)
            reg_amp = amp*smoothstep(t=ns_solver.simu_wrap.time, tsmooth=T_SMOOTH)
            ns_solver.obs_wrap.maximum_amplitude = (reg_amp, 1)
            obs_exporter.append_obs_data(ns_solver.simu_wrap, ns_solver.obs_wrap)
            phase = get_phase(obs_exporter=obs_exporter)
            ns_solver.obs_wrap.phase = (phase, 1)
            ed = get_cumul_energy(obs_exporter=obs_exporter)
            PETSc.Sys.Print(f"ed = {ed}")
            ed_list.append(ed)

            ns_solver.step()

    
            
        obs_num = ns_solver.obs_wrap.obstacles_number
        obs_data = obs_exporter.obs_data
        columns = ["time", "frequency", "amplitude", "x", "y", "Ux", "Uy", "Fx", "Fy", "Pd", "surface_area"]

        if RANK==0:
            data_frames = []
            for i in range(obs_num):
                df = pd.DataFrame(np.array(obs_data[i]), columns=columns)
                data_frames.append(df)
                df.to_hdf(sim_path / f"obstacles.hdf5", key=f"obstacle{i+1}", mode="a")
            
            with open(sim_path / "ed.pkl", "wb") as f:
                pickle.dump(obj=ed_list, file=f)
            
            

            
            

            

    
