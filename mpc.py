import sys
from pathlib import Path

import dill as pickle
import do_mpc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import petsc4py
from ns2d.utils.export import ObsExporter
from ns2d.utils.pyconfig import Config
from petsc4py import PETSc

from helpers import MPCConfig, Sindy2MPC, get_phase, smoothstep
from solver import NSSolver


def setup_model(mpc_config: MPCConfig):
    model_type = "continuous"
    mpc_model = do_mpc.model.Model(model_type)

    mpc_model.set_variable(var_type="_x", var_name="ux", shape=(1,1))
    
    if mpc_config.freq_control:
        mpc_model.set_variable(var_type="_u", var_name="f")
    if mpc_config.amp_control:
        mpc_model.set_variable(var_type="_u", var_name="a")

    with open(mpc_config.sindy_model_path, "rb") as f:
        sindy_model = pickle.load(file=f)
    
    sindy2mpc = Sindy2MPC(sindy_model=sindy_model, mpc_model=mpc_model)
    sindy2mpc.set_mpc_model_rhs()
    mpc_model.setup()

    return mpc_model

def setup_mpc_optimizer(mpc_config: MPCConfig, mpc_model):
    mpc = do_mpc.controller.MPC(mpc_model)

    setup_mpc = {
        'n_horizon': mpc_config.n_horizon,
        't_step': mpc_config.t_step,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)

    u_ref = -1
    u = mpc_model.x["ux"]

    mterm = (u-u_ref)**2
    lterm = (u-u_ref)**2

    mpc.set_objective(mterm=mterm, lterm=lterm)

    if (mpc_config.freq_control and not mpc_config.amp_control):
        mpc.set_rterm(f=mpc_config.rterm_freq)
        mpc.bounds['lower','_u', 'f'] = 0
        mpc.bounds['upper','_u', 'f'] = 3
    elif (not mpc_config.freq_control and mpc_config.amp_control):
        mpc.set_rterm(a=mpc_config.rterm_amp)
        mpc.bounds["lower", "_u", "a"] = 0
        mpc.bounds["upper", "_u", "a"] = 1.5
    elif (mpc_config.freq_control and mpc_config.amp_control):
        mpc.set_rterm(f=mpc_config.rterm_freq, a=mpc_config.rterm_amp)
        mpc.bounds['lower','_u', 'f'] = 0
        mpc.bounds['upper','_u', 'f'] = 3
        mpc.bounds["lower", "_u", "a"] = 0
        mpc.bounds["upper", "_u", "a"] = 1.5
    else:
        raise Exception("At least one of freq_control or amp_control should be true")
    
    mpc.setup()

    return mpc

# def setup_simulator(mpc_model, mpc_config: MPCConfig):
#     simulator = do_mpc.simulator.Simulator(mpc_model)
#     simulator.set_param(t_step = mpc_config.t_step)

#     simulator.setup()

#     return simulator

def mpc_step(mpc, x0):
    x0 = np.array([x0]).reshape(-1,1)
    u0 = mpc.make_step(x0)
    return u0.squeeze()

# def simulator_step(simulator, u0):
#     u0 = np.array([u0]).reshape(-1,1)
#     x0 = simulator.make_step(u0)
#     return x0.item()


COMM = PETSc.COMM_WORLD
RANK = COMM.Get_rank()
CFG_PATH = Path("config.yml").resolve()
MPC_CFG_PATH = Path("mpc_config.yml").resolve()
MAIN_CFG = Config.from_yaml(config_path=CFG_PATH)
MPC_CFG = MPCConfig.from_yaml(config_path=MPC_CFG_PATH)
RESULTS_DIR_PATH = Path(MAIN_CFG.export_info.results_dirpath).resolve()

T_SMOOTH = 0.2


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
    #ns_solver.add_exported_vec(vec=ns_solver.fields.current_time.local_pressure, vector_name="p")
    ns_solver.add_exported_vec(vec=ns_solver.fields.solid_ls[1].local_levelset_solid, vector_name="solid_level_set")
    ns_solver.add_exported_vec(vec=ns_solver.fields.next_time.local_vorticity, vector_name="vorticity")

    ns_solver.results_dir_path = RESULTS_DIR_PATH / "opt_control_simu"
    ns_solver.simu_wrap.dt = 0.002

    if RANK == 0:
        mpc_model = setup_model(mpc_config=MPC_CFG)
        mpc = setup_mpc_optimizer(mpc_config=MPC_CFG, mpc_model=mpc_model)
        x0 = np.array([ns_solver.obs_wrap.velocity_x[0]]).reshape(-1,1)

        mpc.x0 = x0
        mpc.set_initial_guess()
        control_time = 0

    freq = 2.0
    amp = 1.0
    mpi_comm = COMM.tompi4py()
    

    while (ns_solver.simu_wrap.time < 4.0):
        if RANK == 0:
            if ns_solver.simu_wrap.time >= control_time:
                PETSc.Sys.Print(f"x0 = {ns_solver.obs_wrap.velocity_x[0]}", comm=COMM)
                u = mpc_step(mpc=mpc, x0=ns_solver.obs_wrap.velocity_x[0])
                if (MPC_CFG.freq_control and not MPC_CFG.amp_control):
                    freq = u.item()
                elif (not MPC_CFG.freq_control and MPC_CFG.amp_control):
                    amp = u.item()
                elif (MPC_CFG.freq_control and MPC_CFG.amp_control):
                    freq = u[0]
                    amp = u[1]
                PETSc.Sys.Print(f"freq = {freq}", comm=COMM)
                PETSc.Sys.Print(f"amp = {amp}", comm=COMM)
                control_time += MPC_CFG.t_step

        if MPC_CFG.freq_control:
            freq = mpi_comm.bcast(freq, root=0)
        if MPC_CFG.amp_control:
            amp = mpi_comm.bcast(amp, root=0)
        ns_solver.obs_wrap.swimming_frequency = (freq, 1)
        reg_amp = amp*smoothstep(t=ns_solver.simu_wrap.time, tsmooth=T_SMOOTH)
        ns_solver.obs_wrap.maximum_amplitude = (reg_amp, 1)
        obs_exporter.append_obs_data(ns_solver.simu_wrap, ns_solver.obs_wrap)
        phase = get_phase(obs_exporter=obs_exporter)
        ns_solver.obs_wrap.phase = (phase, 1)
        ns_solver.step()
    
    obs_num = ns_solver.obs_wrap.obstacles_number
    obs_data = obs_exporter.obs_data
    columns = ["time", "frequency", "amplitude", "x", "y", "Ux", "Uy", "Fx", "Fy", "Pd", "surface_area"]

    if RANK==0:
        data_frames = []
        for i in range(obs_num):
            df = pd.DataFrame(np.array(obs_data[i]), columns=columns)
            data_frames.append(df)
            df.to_hdf(RESULTS_DIR_PATH / f"obstacles.hdf5", key=f"obstacle{i+1}", mode="a")
        

        df = data_frames[0]
        fig, ax = plt.subplots()
        ax.plot(df["time"], df["Ux"])
        ax.set_xlabel("time [s]")
        ax.set_ylabel("U [m/s]")

        ax.grid(linestyle="--")
        fig.savefig(RESULTS_DIR_PATH /"mpc_cfd.png", dpi=300)

        fig, _, _ = do_mpc.graphics.default_plot(mpc.data)
        fig.savefig(RESULTS_DIR_PATH / "default_plot.png", dpi=300)