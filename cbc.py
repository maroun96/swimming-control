import sys
import pickle
from pathlib import Path
from functools import partial

import pandas as pd
import numpy as np

import petsc4py
from petsc4py import PETSc
from scipy.integrate import simpson

from ns2d.utils.pyconfig import Config
from ns2d.utils.export import ObsExporter


from solver import NSSolver

COMM = PETSc.COMM_WORLD
RANK = COMM.Get_rank()
CFG_PATH = Path("config.yml").resolve()
MAIN_CFG = Config.from_yaml(config_path=CFG_PATH)
RESULTS_DIR_PATH = Path(MAIN_CFG.export_info.results_dirpath).resolve()

def get_time_interval_idx(arr, t_interval_len):
    cumul_sum = np.flip(np.cumsum(np.flip(arr)))
    indx_arr = np.where(cumul_sum >= t_interval_len)[0]
    
    if not indx_arr.size:
        return None
    
    last_indx = indx_arr[-1]
    sum_last_idx = cumul_sum[last_indx]
    sum_after_last_idx = cumul_sum[last_indx+1]
    
    diff1 = np.abs(sum_last_idx-t_interval_len)
    diff2 = np.abs(sum_after_last_idx-t_interval_len)
    
    if diff1 <= diff2:
        return last_indx
    else:
        return last_indx+1

if __name__ == "__main__":
    petsc4py.init(sys.argv, comm=COMM)

    obs_exporter = ObsExporter()
    # obs_exporter.add_exported_param(name=["swimming_frequency","x", "y", "velocity_x", "velocity_y", "Cx", "Cy"])

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

    ns_solver = NSSolver(main_cfg=MAIN_CFG, comm=COMM, obs_exporter=obs_exporter)
    ns_solver.add_exported_vec(vec=ns_solver.fields.current_time.local_ux, vector_name="ux")
    ns_solver.add_exported_vec(vec=ns_solver.fields.current_time.local_uy, vector_name="uy")
    ns_solver.add_exported_vec(vec=ns_solver.fields.current_time.local_pressure, vector_name="p")
    ns_solver.add_exported_vec(vec=ns_solver.fields.solid_ls[1].local_levelset_solid, vector_name="solid_level_set")

    # frequencies = [1.0, 2.0, 3.0, 4.0]
    frequencies = [2.0]

    def phase_func_f(f, t):
        return f*t

    avg_forces = []
    t_int = 0.2

    for it, freq in enumerate(frequencies):
        ns_solver.reset(func_count=it)
        sim_path = RESULTS_DIR_PATH / f"simu_f_{freq}"
        ns_solver.results_dir_path = sim_path

        phase_func = partial(phase_func_f, freq)

        while (ns_solver.simu_wrap.time < ns_solver.simu_wrap.time_final):
            if obs_exporter.obs_data is not None:
                obs_data_array = np.array(obs_exporter.obs_data[0])
                time_array = obs_data_array[:, 0]
                PETSc.Sys.Print(f"t_arr = {time_array}", comm=COMM)
                start_indx = get_time_interval_idx(arr=time_array, t_interval_len=t_int)
                PETSc.Sys.Print(f"start_idx = {start_indx}", comm=COMM)
            
                # if start_indx:
                #     time_interval_arr = obs_data_array[start_indx:, 0]
                #     force_interval_arr = obs_data_array[start_indx:, 6]
                #     t_int_act = time_interval_arr[-1] - time_interval_arr[0]
                #     PETSc.Sys.Print(f"t_arr = {time_interval_arr}", comm=COMM)
                #     # avg_f = (1/t_int_act)*simpson(y=force_interval_arr, x=time_interval_arr)
                #     # avg_forces.append(avg_f)
            
            ns_solver.step(phase=phase_func(ns_solver.simu_wrap.time))

        # ns_solver.simulate_time_seg(phase_func=phase_func)

        obs_num = ns_solver.obs_wrap.obstacles_number
        obs_data = obs_exporter.obs_data
        columns = ["time", "frequency", "x", "y", "Ux", "Uy", "Fx", "Fy", "density", "surface_area"]

        if RANK==0:
            data_frames = []
            for i in range(obs_num):
                df = pd.DataFrame(np.array(obs_data[i]), columns=columns)
                data_frames.append(df)
                df.to_hdf(sim_path / f"obstacles.hdf5", key=f"obstacle{i+1}", mode="a")
            
            with open(file=sim_path / f"average_forces.pkl", mode="wb") as f:
                pickle.dump(obj=avg_forces, file=f)
            

    
