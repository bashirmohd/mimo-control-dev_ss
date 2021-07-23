import numpy as np
from scipy import signal

# DOE phase response, measured
doe_phs_deg = np.array([
    0,  90,  0,
    0,   0, -90,
    180, 0,  0
]).reshape(3, 3)

# Force DOE center output to 0 by design
doe_amp = np.ones_like(doe_phs_deg)
doe_amp[1, 1] = 0

# doe matrix could be in any shape
center_idx = np.where(doe_amp == 0)

# Balanced input beam amplitude, match shape, no center beam
beam_amp = doe_amp


def sim8(beam_phs_deg, beam_amp=beam_amp, doe_phs_deg=doe_phs_deg, doe_amp=doe_amp):
    b = beam_amp * np.exp(1j * np.deg2rad(beam_phs_deg))
    d = doe_amp * np.exp(1j * np.deg2rad(doe_phs_deg))
    s = signal.convolve2d(b, d)
    return np.abs(s * s.conj())
