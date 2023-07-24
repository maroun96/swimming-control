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
from helpers import get_phase

def get_optimal_control():
    model_type = "continuous"
    model = do_mpc.model.Model(model_type)

    u = model.set_variable(var_type="_x", var_name="u", shape=(1,1))
    f = model.set_variable(var_type="_u", var_name="f")

    a1 = 0.588
    a2 = -0.134
    u_ref = -1

    #set right hand side
    model.set_rhs("u", a1*u**2+a2*f**2)
    model.setup()

    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 20,
        't_step': 0.01,
        'store_full_solution': True,
    }
    mpc.set_param(**setup_mpc)

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

    simulator = do_mpc.simulator.Simulator(model)
    simulator.set_param(t_step = 0.01)

    simulator.setup()

    x0 = np.pi*np.array([0.0]).reshape(-1,1)

    simulator.x0 = x0
    mpc.x0 = x0
    mpc.set_initial_guess()

    traj = [x0.item()]
    controls = []
    t = 0
    time = [t]

    N = 600

    for _ in range(N):
        u0 = mpc.make_step(x0)
        x0 = simulator.make_step(u0)
        traj.append(x0.item())
        controls.append(u0.item())
        t += 0.01
        time.append(t)

    spline =PchipInterpolator(x=time[:-1], y=controls)

    return spline

COMM = PETSc.COMM_WORLD
RANK = COMM.Get_rank()
CFG_PATH = Path("config.yml").resolve()
MAIN_CFG = Config.from_yaml(config_path=CFG_PATH)
RESULTS_DIR_PATH = Path(MAIN_CFG.export_info.results_dirpath).resolve()

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

    ns_solver.results_dir_path = RESULTS_DIR_PATH

    if RANK == 0:
        control_spline = get_optimal_control()
        t_test = np.linspace(0, 6, 1000)
        plt.plot(t_test, control_spline(t_test))
        plt.savefig(RESULTS_DIR_PATH / "optimal_control_seq.png")
    else:
        control_spline = None

    COMM.barrier()

    mpi_comm = COMM.tompi4py()
    control_spline = mpi_comm.bcast(control_spline, root=0)

    while (ns_solver.simu_wrap.time < 6.0):
        ns_solver.obs_wrap.swimming_frequency = (control_spline(ns_solver.simu_wrap.time), 1)
        obs_exporter.append_obs_data(ns_solver.simu_wrap, ns_solver.obs_wrap)
        phase = get_phase(obs_exporter=obs_exporter)
        ns_solver.step(phase=phase)
    
    obs_num = ns_solver.obs_wrap.obstacles_number
    obs_data = obs_exporter.obs_data
    columns = ["time", "frequency", "x", "y", "Ux", "Uy", "Fx", "Fy", "density", "surface_area"]

    if RANK==0:
        data_frames = []
        for i in range(obs_num):
            df = pd.DataFrame(np.array(obs_data[i]), columns=columns)
            data_frames.append(df)
            df.to_hdf(RESULTS_DIR_PATH / f"obstacles.hdf5", key=f"obstacle{i+1}", mode="a")