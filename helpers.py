from typing import Union

import numpy as np
from ns2d.utils.config import SimuStructWrapper, ObsStructWrapper

class ObsExporter:
    def __init__(self) -> None:
        self._exported_params = []
        self.obs_data = None
    
    def add_exported_param(self, name: Union[list[str], str]):
        if isinstance(name, str):
            self._exported_params.append(name)
        elif isinstance(name, list):
            self._exported_params.extend(name)

    def _get_obs_row(self, simu_wrap: SimuStructWrapper, obs_wrap: ObsStructWrapper, obs_id: int):
        row = []
        row.append(simu_wrap.time)
        for param_name in self._exported_params:
            param = getattr(obs_wrap, param_name)[obs_id]
            row.append(param)
        
        return np.array(row)
    
    def append_obs_data(self, simu_wrap: SimuStructWrapper, obs_wrap: ObsStructWrapper):
        obstacle_num = obs_wrap.obstacles_number
        if self.obs_data is None:
            self.obs_data = [[]]*obstacle_num
       
        for i in range(obstacle_num):
            self.obs_data[i].append(self._get_obs_row(simu_wrap, obs_wrap, i))
    
    def clear(self):
        if self.obs_data:
            for obs_list in self.obs_data:
                obs_list.clear()

def check_bounds(obs_wrap: ObsStructWrapper, obs_id: int, domain_range: tuple[tuple[float, float], tuple[float, float]]):
    x_markers = obs_wrap.x_markers[obs_id]
    y_markers = obs_wrap.y_markers[obs_id]
    (xmin, xmax), (ymin, ymax) = domain_range

    x_markers_min = np.amin(x_markers)
    x_markers_max = np.amax(x_markers)
    y_markers_min = np.amin(y_markers)
    y_markers_max = np.amax(y_markers)

    x_bool = x_markers_min >= xmin and x_markers_max <= xmax
    y_bool = y_markers_min >= ymin and y_markers_max <= ymax

    return x_bool and y_bool




