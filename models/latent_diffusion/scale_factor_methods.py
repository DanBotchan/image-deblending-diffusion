from abc import (ABC,
                 abstractmethod,
                 )
from typing import NamedTuple

import numpy as np

from utils.exp_ops import ExpConfig


__SCALING_METHOD__ = {}


class BaseScheduler(ABC):
    def __init__(self,
                 chaotic_stage_scale: float,
                 critical_stage_scale: float,
                 refinement_stage_scale: float,
                 critical_stage_start: int,
                 critical_stage_stop: int,
                 **kwargs):
        self.chaotic_stage_scale = chaotic_stage_scale
        self.critical_stage_scale = critical_stage_scale
        self.refinement_stage_scale = refinement_stage_scale
        self.critical_stage_start = critical_stage_start
        self.critical_stage_stop = critical_stage_stop

    @abstractmethod
    def get_scale(self,
                  curr_timestep: int,
                  ) -> float:
        pass

def register_scaling_method(name: str):
    def wrapper(cls):
        if __SCALING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __SCALING_METHOD__[name] = cls
        return cls

    return wrapper


def get_scale_scheduler(exp_config: ExpConfig,
                        **kwargs: object):
    guide_conf, gen_conf = exp_config['guidance'], exp_config['general']
    guide_method = guide_conf['guidance_method']
    if __SCALING_METHOD__.get(guide_method, None) is None:
        raise NameError(f"Name {guide_method} is not defined!")

    return __SCALING_METHOD__[guide_method](
        critical_stage_start=gen_conf['critical_start'],
        critical_stage_stop=gen_conf['critical_stop'],
        chaotic_stage_scale=guide_conf['chaotic'],
        critical_stage_scale=guide_conf['critical'],
        refinement_stage_scale=guide_conf['refine'],
        **kwargs
    )


@register_scaling_method('three_stage')
class ThreeStageScaler(BaseScheduler):
    def get_scale(self,
                  curr_timestep: int,
                  ) -> float:
        if curr_timestep > self.critical_stage_start:
            scale = self.chaotic_stage_scale
        elif self.critical_stage_start >= curr_timestep >=\
            self.critical_stage_stop:
            scale = self.critical_stage_scale
        else:
            scale = self.refinement_stage_scale
        return scale


@register_scaling_method('three_stage_linear')
class ThreeStageLinearScaler(BaseScheduler):
    def __init__(self,
                 chaotic_stage_scale: float,
                 critical_stage_scale: float,
                 refinement_stage_scale: float,
                 critical_stage_start: int,
                 critical_stage_stop: int,
                 **kwargs):
        super().__init__(chaotic_stage_scale=chaotic_stage_scale,
                         critical_stage_scale=critical_stage_scale,
                         refinement_stage_scale=refinement_stage_scale,
                         critical_stage_start=critical_stage_start,
                         critical_stage_stop=critical_stage_stop)
        self.scale_array = self._generate_scale_array(array_size=1000)

    def get_scale(self, curr_timestep):
        if curr_timestep >= len(self.scale_array):
            return self.chaotic_stage_scale
        return self.scale_array[curr_timestep]

    def _generate_scale_array(self, array_size=1000):
        """
        Generate an array with smooth transitions between scale values.

        The array has three regions:
        1. From 0 to critical_stage_stop: refinement_stage_scale to
        critical_stage_scale
        2. From critical_stage_start to critical_stage_stop:
        critical_stage_scale to chaotic_stage_scale
        3. From critical_stage_stop to end: critical_stage_scale to
        chaotic_stage_scale
        """
        scale_array = np.zeros(array_size)

        # From 0 to critical_stage_stop: refinement_stage_scale to
        # critical_stage_scale
        for i in range(0, self.critical_stage_stop + 1):
            # Linear interpolation
            t = i / self.critical_stage_stop if self.critical_stage_stop > 0\
                else 0
            scale_array[i] = (
                                 1 - t) * self.refinement_stage_scale +\
                             t * self.critical_stage_scale

        # From critical_stage_start to critical_stage_stop:
        # critical_stage_scale to chaotic_stage_scale
        for i in range(self.critical_stage_start,
                       self.critical_stage_stop + 1):
            # Linear interpolation
            t = (i - self.critical_stage_start) / (
                self.critical_stage_stop - self.critical_stage_start) if (
                                                                             self.critical_stage_stop - self.critical_stage_start) > 0 else 0
            scale_array[i] = (
                                 1 - t) * self.critical_stage_scale + t\
                             * self.chaotic_stage_scale

        # From critical_stage_stop to end: critical_stage_scale to
        # chaotic_stage_scale
        for i in range(self.critical_stage_stop + 1, array_size):
            # Linear interpolation
            t = (i - self.critical_stage_stop) / (
                array_size - 1 - self.critical_stage_stop) if (
                                                                  array_size
                                                                  - 1 -
                                                                  self.critical_stage_stop) > 0 else 0
            scale_array[i] = (
                                 1 - t) * self.critical_stage_scale + t\
                             * self.chaotic_stage_scale

        return scale_array
