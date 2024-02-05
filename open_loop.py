import sys
import pickle
from pathlib import Path

import numpy as np
import casadi as ca
import pandas as pd
import petsc4py
import matplotlib.pyplot as plt

from petsc4py import PETSc
from ns2d.utils.pyconfig import Config
from ns2d.utils.export import ObsExporter

from solver import NSSolver

from helpers import smoothstep, get_phase, get_cumul_energy

COMM = PETSc.COMM_WORLD
RANK = COMM.Get_rank()
CFG_PATH = Path("config.yml").resolve()
MAIN_CFG = Config.from_yaml(config_path=CFG_PATH)
RESULTS_DIR_PATH = Path(MAIN_CFG.export_info.results_dirpath).resolve()
T_SMOOTH = 0.2 #add to config
T = 12 #time horizon
N = 60 #Number of control intervals

#should be changed to read from pickled sindy model
def build_integrator(time_horizon, n_controls):
    p = ca.MX.sym("p")
    v = ca.MX.sym("v")
    e = ca.MX.sym("e")
    x = ca.vertcat(p,v,e)
    f = ca.MX.sym("f")
    a = ca.MX.sym("a")
    u = ca.vertcat(f, a)

    # Model equations (read from input file)
    pdot = v
    vdot = -0.209*v-0.276*v**2-0.436*v**3-0.067*f*a-0.109*f**2*a**2+0.148*v*f*a-0.261*v**2*f*a-0.071*v*f**2*a-0.344*v*f*a**2
    edot = 0.073*f*a-0.068*f**2*a-0.050*f*a**2+0.050*f**2*a**2+0.017*f**3*a+0.031*e*f*a+0.013*e**2*f*a-0.018*e*f**2*a-0.018*e*f*a**2
    xdot = ca.vertcat(pdot, vdot, edot)

    ode = {'x':x, 'u':u, 'ode':xdot}
    F = ca.integrator('F', 'cvodes', ode, 0, time_horizon/n_controls)
    return F

def solve_nlp(integrator, n_controls):
    # For plotting x and u given w
    x_plot = []
    u_plot = []
    u_list = []

    # Start with an empty NLP
    w=[] 
    w0 = []
    lbw = []
    ubw = []
    g= []
    lbg = []
    ubg = []
    J = 0

    # "Lift" initial conditions
    Xk = ca.MX.sym('X0', 3)
    w.append(Xk)
    lbw.append([0, 0, 0]), ubw.append([0, 0, 0]), w0.append([0, 0, 0])
    x_plot.append(Xk)

    # Formulate the NLP
    for k in range(n_controls):
        # New NLP variable for the control
        Uk = ca.MX.sym('U_' + str(k), 2)
        u_list.append(Uk)
            
        w.append(Uk)
        lbw.append([0, 0])
        ubw.append([3.0, 1.5])
        w0.append([1.0, 0.5])
        u_plot.append(Uk)

        # Integrate till the end of the interval
        Fk = integrator(x0=Xk, u=Uk) 
        Xk_end = Fk['xf'] 
        J = J + 0.001*Uk[0]**2 + 0.001*Uk[1]**2
        if k > 0:
            Uk_prev = u_list[-2]
            J = J + 0.001*(Uk[0]-Uk_prev[0])**2 + 0.01*(Uk[1]-Uk_prev[1])**2

        # New NLP variable for state at end of interval
        Xk = ca.MX.sym('X_' + str(k+1), 3)
        w.append(Xk)


        if k == N-1:
            lbw.append([-4, -ca.inf, -ca.inf])
            ubw.append([-4, ca.inf, ca.inf])
            J = J + Xk[2]
        else:
            lbw.append([-ca.inf, -ca.inf, -ca.inf])
            ubw.append([ca.inf, ca.inf, ca.inf])
                
        w0.append([0, 0, 0])
        x_plot.append(Xk)

        # Add equality constraint
        g.append(Xk_end-Xk)
        lbg.append([0, 0, 0])
        ubg.append([0, 0, 0])

    # Concatenate vectors
    w = ca.vertcat(*w)
    g = ca.vertcat(*g)
    x_plot = ca.horzcat(*x_plot)
    u_plot = ca.horzcat(*u_plot)
    w0 = np.concatenate(w0)
    lbw = np.concatenate(lbw)
    ubw = np.concatenate(ubw)
    lbg = np.concatenate(lbg)
    ubg = np.concatenate(ubg)

    # Create an NLP solver
    prob = {'f': J, 'x': w, 'g': g}
    solver = ca.nlpsol('solver', 'ipopt', prob)

    # Function to get x and u trajectories from w
    trajectories = ca.Function('trajectories', [w], [x_plot, u_plot], ['w'], ['x', 'u']) 

    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
    x_opt, u_opt = trajectories(sol['x'])
    x_opt = x_opt.full() # to numpy array
    u_opt = u_opt.full() # to numpy array

    return x_opt, u_opt


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
    ns_solver.add_exported_vec(vec=ns_solver.fields.next_time.local_vorticity, vector_name="vorticity")
    ns_solver.add_exported_vec(vec=ns_solver.fields.solid_ls[1].local_levelset_solid, vector_name="solid_level_set")

    
    ns_solver.simu_wrap.dt = 0.002
    
        
    ns_solver.results_dir_path = RESULTS_DIR_PATH 
    u_opt = None

    if RANK == 0:
        integrator = build_integrator(time_horizon=T, n_controls=N)
        x_opt, u_opt = solve_nlp(integrator=integrator, n_controls=N)

        np.save(file=ns_solver.results_dir_path / "x_opt.npy", arr=x_opt)
        np.save(file= ns_solver.results_dir_path / "u_opt.npy", arr=u_opt)
            
        
        tgrid = np.linspace(0, T, N+1)
        plt.step(tgrid, np.append(np.nan, u_opt[0]))
        plt.step(tgrid, np.append(np.nan, u_opt[1]))
        plt.savefig(ns_solver.results_dir_path / "controls.png")

        
    mpi_comm = COMM.tompi4py()
    u_opt = mpi_comm.bcast(u_opt, root=0)
        

    def freq_func(time):
        data = u_opt[0]
        time_points = np.linspace(0, T, num=u_opt.shape[1])
        index = np.searchsorted(time_points, time, side="right") - 1
        index = np.clip(index, 0, len(data) - 1)
        return data[index]

    def amp_func(time):
        data = u_opt[1]
        time_points = np.linspace(0, T, num=u_opt.shape[1])
        index = np.searchsorted(time_points, time, side="right") - 1
        index = np.clip(index, 0, len(data) - 1)
        return data[index]
        

    ed_list = []
        
    while (ns_solver.simu_wrap.time < T):
        freq = freq_func(time=ns_solver.simu_wrap.time)
        amp = amp_func( time=ns_solver.simu_wrap.time)
        ns_solver.obs_wrap.swimming_frequency = (freq, 1)
        reg_amp = amp*smoothstep(t=ns_solver.simu_wrap.time, tsmooth=T_SMOOTH)
        ns_solver.obs_wrap.maximum_amplitude = (reg_amp, 1)
        obs_exporter.append_obs_data(ns_solver.simu_wrap, ns_solver.obs_wrap)
        phase = get_phase(obs_exporter=obs_exporter)
        ns_solver.obs_wrap.phase = (phase, 1)
        ed = get_cumul_energy(obs_exporter=obs_exporter)
        ed_list.append(ed)
        PETSc.Sys.Print(f"ed = {ed}")
        ns_solver.step()
        
    obs_exporter.append_obs_data(ns_solver.simu_wrap, ns_solver.obs_wrap)
    ed = get_cumul_energy(obs_exporter=obs_exporter)
    ed_list.append(ed)
    obs_num = ns_solver.obs_wrap.obstacles_number
    obs_data = obs_exporter.obs_data
    columns = ["time", "frequency", "amplitude", "x", "y", "Ux", "Uy", "Fx", "Fy", "Pd", "surface_area"]

    if RANK==0:
        data_frames = []
        for i in range(obs_num):
            df = pd.DataFrame(np.array(obs_data[i]), columns=columns)
            data_frames.append(df)
            df.to_hdf(ns_solver.results_dir_path / f"obstacles.hdf5", key=f"obstacle{i+1}", mode="a")
                
        with open(ns_solver.results_dir_path / "ed.pkl", "wb") as f:
            pickle.dump(obj=ed_list, file=f)

           

    
        