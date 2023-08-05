import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import do_mpc
import petsc4py

from petsc4py import PETSc
from scipy.interpolate import PchipInterpolator

from ns2d.utils.pyconfig import Config
from ns2d.utils.export import ObsExporter

from solver import NSSolver
from helpers import get_phase, smoothstep


def setup_model():
    model_type = "continuous"
    model = do_mpc.model.Model(model_type)

    u = model.set_variable(var_type="_x", var_name="u", shape=(1,1))
    f = model.set_variable(var_type="_u", var_name="f")

    a1 = 0.368
    a2 = -0.089

    model.set_rhs("u", a1*u**2+a2*f**2)
    model.setup()

    return model

def setup_mpc_optimizer(n_horizon, t_step, model):
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': n_horizon,
        't_step': t_step,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)

    u_ref = -1
    u = model.x["u"]

    mterm = (u-u_ref)**2
    lterm = (u-u_ref)**2

    mpc.set_objective(mterm=mterm, lterm=lterm)

    mpc.set_rterm(
        f=0.01,
    )
    
    # Lower bounds on inputs:
    mpc.bounds['lower','_u', 'f'] = 0
    # Upper bounds on inputs:
    mpc.bounds['upper','_u', 'f'] = 3

    mpc.setup()

    return mpc

def setup_simulator(model, t_step):
    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step = t_step)

    simulator.setup()

    return simulator

def mpc_step(mpc, x0):
    x0 = np.array([x0]).reshape(-1,1)
    u0 = mpc.make_step(x0)
    return u0.item()

def simulator_step(simulator, u0):
    u0 = np.array([u0]).reshape(-1,1)
    x0 = simulator.make_step(u0)
    return x0.item()


COMM = PETSc.COMM_WORLD
RANK = COMM.Get_rank()
CFG_PATH = Path("config.yml").resolve()
MAIN_CFG = Config.from_yaml(config_path=CFG_PATH)
RESULTS_DIR_PATH = Path(MAIN_CFG.export_info.results_dirpath).resolve()
N_HORIZON = 20
MPC_T_STEP = 0.03
T_SMOOTH = 0.2

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
        "power_def",
        "surface_area"
    ])

    ns_solver = NSSolver(main_cfg=MAIN_CFG, comm=COMM)
    ns_solver.add_exported_vec(vec=ns_solver.fields.current_time.local_ux, vector_name="ux")
    ns_solver.add_exported_vec(vec=ns_solver.fields.current_time.local_uy, vector_name="uy")
    ns_solver.add_exported_vec(vec=ns_solver.fields.current_time.local_pressure, vector_name="p")
    ns_solver.add_exported_vec(vec=ns_solver.fields.solid_ls[1].local_levelset_solid, vector_name="solid_level_set")

    ns_solver.results_dir_path = RESULTS_DIR_PATH / "opt_control_simu"

    if RANK == 0:
        model = setup_model()
        mpc = setup_mpc_optimizer(n_horizon=N_HORIZON, t_step=MPC_T_STEP, model=model)
        simulator = setup_simulator(model=model, t_step=MPC_T_STEP)
        x0 = np.array([ns_solver.obs_wrap.velocity_x[0]]).reshape(-1,1)
        simulator.x0 = x0
        mpc.x0 = x0
        mpc.set_initial_guess()
        control_time = 0
        sindy_model_traj = [x0.item()]
        sindy_model_time = [control_time]

    freq = None
    mpi_comm = COMM.tompi4py()
    

    while (ns_solver.simu_wrap.time < 4.0):
        if RANK == 0:
            if ns_solver.simu_wrap.time >= control_time:
                PETSc.Sys.Print(f"x0 = {ns_solver.obs_wrap.velocity_x[0]}", comm=COMM)
                freq = mpc_step(mpc=mpc, x0=ns_solver.obs_wrap.velocity_x[0])
                PETSc.Sys.Print(f"freq = {freq}", comm=COMM)
                control_time += MPC_T_STEP
                sindy_model_time.append(control_time)
                sindy_model_traj.append(simulator_step(simulator=simulator, u0=freq))

        COMM.barrier()
        freq = mpi_comm.bcast(freq, root=0)
        ns_solver.obs_wrap.swimming_frequency = (freq, 1)
        reg_amp = smoothstep(t=ns_solver.simu_wrap.time, tsmooth=T_SMOOTH)
        ns_solver.obs_wrap.maximum_amplitude = (reg_amp, 1)
        obs_exporter.append_obs_data(ns_solver.simu_wrap, ns_solver.obs_wrap)
        phase = get_phase(obs_exporter=obs_exporter)
        ns_solver.obs_wrap.phase = (phase, 1)
        ns_solver.step()
    
    obs_num = ns_solver.obs_wrap.obstacles_number
    obs_data = obs_exporter.obs_data
    columns = ["time", "frequency", "x", "y", "Ux", "Uy", "Fx", "Fy", "Pd", "surface_area"]

    if RANK==0:
        data_frames = []
        for i in range(obs_num):
            df = pd.DataFrame(np.array(obs_data[i]), columns=columns)
            data_frames.append(df)
            df.to_hdf(RESULTS_DIR_PATH / f"obstacles.hdf5", key=f"obstacle{i+1}", mode="a")
        
        fig, ax = plt.subplots()
        ax.plot(sindy_model_time, sindy_model_traj)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("U [m/s]")

        ax.grid(linestyle="--")
        fig.savefig(RESULTS_DIR_PATH /"mpc_sindy.png", dpi=300)

        df = data_frames[0]
        fig, ax = plt.subplots()
        ax.plot(df["time"], df["Ux"])
        ax.set_xlabel("time [s]")
        ax.set_ylabel("U [m/s]")

        ax.grid(linestyle="--")
        fig.savefig(RESULTS_DIR_PATH /"mpc_cfd.png", dpi=300)

        fig, _, _ = do_mpc.graphics.default_plot(mpc.data)
        fig.savefig(RESULTS_DIR_PATH / "default_plot.png", dpi=300)