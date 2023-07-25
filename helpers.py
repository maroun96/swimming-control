from collections import deque
from itertools import product
from math import dist

import numpy as np
from ns2d.utils.export import ObsExporter
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






