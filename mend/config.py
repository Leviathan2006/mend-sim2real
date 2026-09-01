"""Constants and run configuration.

The three constants at the bottom are properties of the benchmark, not choices:
SIGMA_GLOBAL is published with the scorer, BAD_REAL is the file the organisers
flagged as a duplicate, and the water/chord values fix the inflow speed implied
by each Reynolds number.
"""
import os
from dataclasses import dataclass, field, asdict
from typing import List, Tuple


@dataclass
class Config:
    data_root: str = os.environ.get("REALPDE_ROOT", "data")
    out_dir: str = "runs"

    H: int = 64
    W: int = 128
    K_IN: int = 10           # observed frames given to a forecaster
    H_OUT: int = 20          # forecast horizon, in frames
    STRIDE: int = 20
    EVAL_WIN: int = 40       # evaluation windows per test case
    ENS: int = 20            # ensemble members used to form intervals
    T_CAP: int = 400         # frames read per file
    N_CASES: int = 40
    SEEDS: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    quick: bool = False

    def __post_init__(self):
        if self.quick:
            self.SEEDS = [0]
            self.EVAL_WIN = 6

    def as_dict(self):
        return asdict(self)


@dataclass
class TrainConfig:
    MODES_Y: int = 16
    MODES_X: int = 24
    WIDTH: int = 32
    LAYERS: int = 4
    LIFT: int = 64
    BATCH: int = 8
    LR: float = 1e-3
    WD: float = 1e-5
    PRE_EP: int = 6          # simulator pretraining epochs
    FT_EP: int = 25          # fine-tuning epochs
    PATIENCE: int = 5
    MT: int = 4              # ensemble members drawn per PNO training step
    L2SP_ALPHA: float = 1e-2
    EWC_LAMBDA: float = 1e2
    PG_TAU: float = 0.90
    PG_ALPHA: float = 1.0
    PG_FRAC: float = 0.10
    PG_MAXK: int = 500

    def quicken(self):
        self.PRE_EP, self.FT_EP = 1, 2
        return self


@dataclass
class MendConfig:
    AUX_EP: int = 30
    AUX_LR: float = 2e-3
    GCLIP: Tuple[float, float] = (0.5, 3.0)      # bounds on the per-point gain
    A_GRID: Tuple[float, ...] = (0.0, 0.25, 0.5, 0.75)   # mean blend, 0 = off
    B_GRID: Tuple[float, ...] = (0.0, 0.5, 0.75, 1.0)    # energy blend, 0 = off
    FIT_WIN: int = 24
    WRAP: Tuple[str, ...] = ("direct fine-tune", "PhysGuard", "PNO")


NU_WATER = 1e-6          # m^2/s
CHORD_M = 0.0734         # m
DT_REAL_S = 0.02         # camera period, 50 fps
DT_SIM_STAR = 0.02002    # simulator step, nondimensional

BAD_REAL = {(7575, 0)}   # organiser-flagged duplicate of 6300_0
SIGMA_GLOBAL = 0.0563870  # published with the scorer, in m/s

# MVPE probe columns as a fraction of the width. The official locations are not
# published, so ours are stated here rather than left implicit.
PROBE_FRAC = (0.60, 0.75, 0.90)

EPS = 1e-12


def u_of(Re):
    """Inflow speed in m/s implied by a Reynolds number."""
    return Re * NU_WATER / CHORD_M


def sim_per_real(Re):
    """Simulator steps per camera frame. Runs from 0.7 to 5.0 across this
    dataset, which is why the two sides cannot be indexed with a shared clock."""
    return DT_REAL_S * u_of(Re) / CHORD_M / DT_SIM_STAR
