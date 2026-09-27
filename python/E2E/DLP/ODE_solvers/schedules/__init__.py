"""Probability paths, time grids, and output-parameterization conversions."""
from .probability_paths import CondOTPath, CondOTParameterization, VPSDEPath, VPSDEParameterization
TimeSchedule = CondOTPath
LinearVPSDE = VPSDEPath
VPParameterization = VPSDEParameterization
__all__ = ['CondOTPath', 'CondOTParameterization', 'VPSDEPath', 'VPSDEParameterization', 'TimeSchedule', 'LinearVPSDE', 'VPParameterization']
