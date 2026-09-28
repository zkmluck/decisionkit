"""Typed decisions without a GPU.

State in, calibrated distributions out, zero decoded tokens. See README.md.
"""
from .contract import API_VERSION, prepare
from .model import DecisionModel

__all__ = ['API_VERSION', 'DecisionModel', 'prepare']
__version__ = '0.1.0'
