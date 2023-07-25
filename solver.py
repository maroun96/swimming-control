import math
from typing import Union, Callable
from pathlib import Path

import numpy as np
from petsc4py import PETSc

from ns2d.utils.pyconfig import initialize_dmda, FieldsContainer, Config
from ns2d.utils.export import Exporter, ObsExporter

from ns2d.solver.navierstokes import navier_stokes_step, init_navier_stokes, deform_bodies
from ns2d.utils.config import (FieldsStructWrapper, ProcessStructWrapper, GridStructWrapper,
                                SimuStructWrapper, ObsStructWrapper, LinStructWrapper, ArrayStructWrapper)

class NSSolver:
    def __init__(self, main_cfg: Config, comm: PETSc.Comm) -> None:
        da = initialize_dmda(grid=main_cfg.grid, comm=comm)
        da.setUniformCoordinates(xmin=main_cfg.grid.xmin, xmax=main_cfg.grid.xmax, ymin=main_cfg.grid.ymin, 
                         ymax=main_cfg.grid.ymax, zmin = 0, zmax = 0)
        self.fields = FieldsContainer.create_fields(da=da, main_config=main_cfg)

        xmin = main_cfg.grid.xmin
        xmax = main_cfg.grid.xmax
        ymin = main_cfg.grid.ymin
        ymax = main_cfg.grid.ymax
        self._domain_range = ((xmin, xmax), (ymin, ymax))
        self.in_bounds = True

        self.proc_wrap = ProcessStructWrapper(da)
        self.grid_wrap = GridStructWrapper(main_cfg.grid,da)
        self.simu_wrap = SimuStructWrapper(main_cfg.simu)
        self.obs_wrap = ObsStructWrapper(main_cfg)
        self.fields_wrap = FieldsStructWrapper(self.fields,main_cfg)
        self.lin_wrap = LinStructWrapper()
        self.arr_wrap = ArrayStructWrapper()

        
        #Add these parameters to config later
        #self.obs_wrap.maximum_amplitude = (1, 1)
        self.obs_wrap.wavelength = (1, 1)
        self.obs_wrap.amp_coef0 = (0.02, 1)
        self.obs_wrap.amp_coef1 = (-0.12, 1)
        self.obs_wrap.amp_coef2 = (0.2, 1)

        init_navier_stokes(
            self.simu_wrap,
            self.grid_wrap,
            self.fields_wrap,
            self.proc_wrap,
            self.obs_wrap,
            self.lin_wrap,
            self.arr_wrap
        )

    
        self.fields_exporter = Exporter(da=da, grid=main_cfg.grid, comm=comm)
    
        self._global_counter = 0
        self._export_time = 0
        self._export_time_interval = self.simu_wrap.time_snapshots
    
    @property
    def results_dir_path(self):
        return self._results_dir_path

    @results_dir_path.setter
    def results_dir_path(self, p: Union[str, Path]):
        if not isinstance(p, Path):
            p = Path(p)
        self._results_dir_path = p.resolve()
        self._results_dir_path.mkdir(parents=True, exist_ok=True)
    
    def simulate_time_seg(self, phase_func: Callable,  t_final: float = None):
        if t_final is None: t_final = self.simu_wrap.time_final
        while(self.simu_wrap.time < t_final):
            self.step(phase=phase_func(self.simu_wrap.time))
        
    def step(self):
        PETSc.Sys.Print(f"amax = {self.obs_wrap.maximum_amplitude[0]}")

        navier_stokes_step(
            self.simu_wrap,
            self.grid_wrap,
            self.fields_wrap,
            self.proc_wrap,
            self.obs_wrap,
            self.lin_wrap,
            self.arr_wrap
        )
                    
        self._export_vectors()
    
    def add_exported_vec(self, vec: PETSc.Vec, vector_name: str):
        self.fields_exporter.add_vector(vec=vec, vector_name=vector_name)
    
    def _export_vectors(self):
        if (self.simu_wrap.time > self._export_time):
            self._global_counter += 1
            self.fields_exporter.export_vectors(self.results_dir_path / f"solution{self._global_counter}.pvtr", timestep=self.simu_wrap.time)
            self._export_time += self._export_time_interval

        
    def reset(self, func_count: int):
        if func_count > 0:
            init_navier_stokes(
                self.simu_wrap,
                self.grid_wrap,
                self.fields_wrap,
                self.proc_wrap,
                self.obs_wrap,
                self.lin_wrap,
                self.arr_wrap
            )
        self._global_counter = 0
        self._export_time = 0        
            

