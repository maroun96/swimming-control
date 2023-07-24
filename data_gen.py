import sys
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import petsc4py
from petsc4py import PETSc
from jenkspy import JenksNaturalBreaks
from scipy.interpolate import PchipInterpolator

from ns2d.utils.pyconfig import Config
from ns2d.utils.export import ObsExporter

from solver import NSSolver
from helpers import SlidingWindow, get_phase

COMM = PETSc.COMM_WORLD
RANK = COMM.Get_rank()
CFG_PATH = Path("config.yml").resolve()
MAIN_CFG = Config.from_yaml(config_path=CFG_PATH)
RESULTS_DIR_PATH = Path(MAIN_CFG.export_info.results_dirpath).resolve()
N_CLUSTERS = 4

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

    ns_solver = NSSolver(main_cfg=MAIN_CFG, comm=COMM)
    ns_solver.add_exported_vec(vec=ns_solver.fields.current_time.local_ux, vector_name="ux")
    ns_solver.add_exported_vec(vec=ns_solver.fields.current_time.local_uy, vector_name="uy")
    ns_solver.add_exported_vec(vec=ns_solver.fields.current_time.local_pressure, vector_name="p")
    ns_solver.add_exported_vec(vec=ns_solver.fields.solid_ls[1].local_levelset_solid, vector_name="solid_level_set")

    frequencies = [1.0, 1.5, 2.0, 2.5, 3.0]
    tf_list = [12.0, 8.0, 6.0, 4.8, 4.0]

    # t_knots = np.arange(0, 7)
    # freq_list = [3, 3, 2, 1.9, 1.9, 1.9, 1.9]
    # spline = PchipInterpolator(x=t_knots, y=freq_list, extrapolate=False)

    window = SlidingWindow(window_length=0.2, init_zero=True)
    window.add_data(name="Ux")

    for it, (freq, tf) in enumerate(zip(frequencies, tf_list)):
        ns_solver.reset(func_count=it)
        window.reset()
        obs_exporter.clear()
        sim_path = RESULTS_DIR_PATH / f"simu_f_{freq}"
        ns_solver.results_dir_path = sim_path
    
        while (ns_solver.simu_wrap.time < tf):
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
                df.to_hdf(sim_path / f"obstacles.hdf5", key=f"obstacle{i+1}", mode="a")
            
            with open(sim_path / "ux_avg.pkl", "wb") as f:
                pickle.dump(obj=window.data_set_avg["Ux"], file=f)
            
            # jnb = JenksNaturalBreaks(N_CLUSTERS)
            # jnb.fit(window.data_set_avg["Ux"])
            # centroids = np.array(list(map(lambda x: np.mean(x), jnb.groups_)))
            # np.save(file=sim_path / "centroids.npy", arr=np.flip(centroids))
            
            

            

    
