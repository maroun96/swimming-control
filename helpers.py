import re
import os
from collections import deque
from dataclasses import dataclass
from itertools import product
from math import dist
from pathlib import Path
from typing import Union

import numpy as np
from dacite import from_dict
from ns2d.utils.export import ObsExporter
from ns2d.utils.pyconfig import load_yaml
from scipy.integrate import trapezoid


class SlidingWindow:
    def __init__(self, window_length=0.1, init_zero=False):
        self.window_length = window_length
        self.t_sliding_window = deque([])
        self.data_set = {}
        self.data_set_avg = {}
        
        self.window_size_flag = False
        self.init_zero = init_zero

        if self.init_zero:
            self.t_sliding_window.append(0)
    
    @property
    def window_length(self):
        return self._window_length
    
    @window_length.setter
    def window_length(self, tl: float):
        self._window_length = tl

    def append(self, t, other_data = None):
        self.t_sliding_window.append(t)
        
        if other_data:
            assert other_data.keys() == self.data_set.keys()
            for k, v in self.data_set.items():
                v.append(other_data[k])
        
        self._cut_to_interval_length()
        
        if self.window_size_flag:
            self._get_avg_values()
    
    def add_data(self, name):
        self.data_set.update({name: deque([])})
        self.data_set_avg.update({name: []})

        if self.init_zero:
            self.data_set[name].append(0)
            self.data_set_avg[name].append(0)
    
    def _cut_to_interval_length(self):
        interval_length = self._get_interval_length()
        if interval_length > self.window_length:
            if not self.window_size_flag: self.window_size_flag = True
            counter = 1
            shifted_interval_length = self._get_interval_length(starting_idx=counter)
            while (shifted_interval_length > self.window_length):
                counter += 1
                shifted_interval_length = self._get_interval_length(starting_idx=counter)
            
            shifted_interval_len_pos = self._get_interval_length(starting_idx=counter-1)
            
            if abs(self.window_length-shifted_interval_length) <= abs(self.window_length-shifted_interval_len_pos):
                for _ in range(counter): 
                    self.t_sliding_window.popleft()
                    for v in self.data_set.values():
                        v.popleft()
            else:
                for _ in range(counter-1):
                    self.t_sliding_window.popleft()
                    for v in self.data_set.values():
                        v.popleft()
                         
    def _get_interval_length(self, starting_idx = 0):
        return self.t_sliding_window[-1] - self.t_sliding_window[starting_idx]
    
    def _get_avg_values(self):
        for k, v in self.data_set.items():
            t_int = self._get_interval_length()
            avg = (1/t_int)*trapezoid(y=v, x=self.t_sliding_window)
            self.data_set_avg[k].append(avg)
    
    def reset(self):
        self.t_sliding_window.clear()
        for v, v_avg in zip(self.data_set.values(), self.data_set_avg.values()):
            v.clear()
            v_avg.clear()
        self.window_size_flag = False

        if self.init_zero:
            self.t_sliding_window.append(0)
            for v, v_avg in zip(self.data_set.values(), self.data_set_avg.values()):
                v.append(0)
                v_avg.append(0)

class Sindy2MPC:
    def __init__(self, sindy_model, mpc_model):
        self._sindy_model = sindy_model
        self._mpc_model = mpc_model
        self._feature_names = self._sindy_model.feature_names
        self._check_feature_names()
        
        self._basis_functions = self._sindy_model.get_feature_names()
        self._coefficients = self._sindy_model.coefficients()
        
        self._n_odes, self._n_basis_functions = self._coefficients.shape

        assert self._n_odes == len(self._mpc_model.x.keys())
        
        self._processed_ftr_list = [[]]*self._n_odes
        
        self._get_processed_ftr_list()
    
    def _check_feature_names(self):
        for feature in self._feature_names:
            assert (feature in self._mpc_model.x.keys()) or (feature in self._mpc_model.u.keys())
            
    def _access_mpc_model_var(self, var_name):
        if var_name in self._mpc_model.x.keys():
            return self._mpc_model.x[var_name]
        elif var_name in self._mpc_model.u.keys():
            return self._mpc_model.u[var_name]  
    
    def _process_ftr(self, split_ftr):
        if '^' in split_ftr:
            var_name, exp = split_ftr.split('^')
            mpc_var = self._access_mpc_model_var(var_name)
            return mpc_var**(int(exp))
        else:
            mpc_var = self._access_mpc_model_var(split_ftr)
            return mpc_var
    
    def _get_processed_ftr_list(self):
        for ode_idx,coef_array in enumerate(self._sindy_model.coefficients()): 
            for ftr, coef in zip(self._sindy_model.get_feature_names(), coef_array):
                if coef != 0:
                    self._processed_ftr_list[ode_idx].append(np.prod(list(map(self._process_ftr, ftr.split()))))
    
    def set_mpc_model_rhs(self):
        for n, x in enumerate(self._mpc_model.x.keys()):
            coefs = self._coefficients[n]
            nnz_idx = coefs.nonzero()
            nnz_coefs = coefs[nnz_idx]
            rhs = 0
            for ftr, coef in zip(self._processed_ftr_list[n], nnz_coefs):
                rhs += ftr*coef
            self._mpc_model.set_rhs(x, rhs)

@dataclass
class MPCConfig:
    n_horizon: int
    t_step: float
    freq_control: bool
    amp_control: bool
    sindy_model_path: Union[str, os.PathLike]

    @classmethod
    def from_yaml(cls, config_path):
        cfg = load_yaml(config_path=config_path)
        return from_dict(data_class=cls, data=cfg)


#change later for multi dimensional case
def rbk_interpolation(state, control_vector, centroids):
    diff = state - centroids
    norms = (diff*diff)
    exp_norm = np.exp(-norms)
    exp_norm_sum = exp_norm.sum()
    state_control = 0

    for c, w in zip(control_vector, exp_norm):
        state_control += c*w 
        
    return state_control/exp_norm_sum

def get_phase(obs_exporter: ObsExporter):
    obs_data = np.array(obs_exporter.obs_data[0])
    time = obs_data[:, 0]
    frequency = obs_data[:, 1]

    return trapezoid(y=frequency, x=time)

def construct_iterator(freq_list: list[float], amp_list: list[float]):
    default_frequency = 2.0
    default_amplitude = 1.0

    if freq_list and amp_list:
        return product(freq_list, amp_list)
    elif freq_list and not amp_list:
        return [(f, default_amplitude) for f in freq_list]
    elif not freq_list and amp_list:
        return [(default_frequency, a) for a in amp_list]
    else:
        return [(default_frequency, default_amplitude)]

def compute_dist(obs_wrap, initial_position: tuple[float, float]):
    current_position = (obs_wrap.x[0], obs_wrap.y[0])
    return dist(current_position, initial_position)


def smoothstep(t, tsmooth):
    k = max(0, min(1, t/tsmooth))
    return k**2*(3-2*k)

def parse_sim_dir(sim_dir: Path):
    dir_name = sim_dir.name
    n_list = re.findall(r"[-+]?(?:\d*\.*\d+)", dir_name)
    freq = n_list[0]
    amp = n_list[1]
    return freq, amp






