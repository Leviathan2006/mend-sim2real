# MEND

Code for the paper on correcting a frozen neural operator's mean and energy
statistics under sim-to-real transfer, evaluated on the RealPDEBench airfoil
pair (paired CFD and time-resolved PIV of flow over a NACA airfoil).

Three things live here:

- **an error budget**, which replaces one statistic of a forecast with its
  measured value and re-scores, bounding what any correction of that statistic
  could achieve;
- **eight baselines** (three classical, five learned sim-to-real methods) run
  through one scorer over five seeds;
- **MEND**, which predicts the two statistics the budget identifies and uses
  them to correct a frozen forecaster without retraining it.

## Data

Download the RealPDEBench airfoil release and point the code at the directory
holding `train_sim/` and `train_real/`:

    export REALPDE_ROOT=/path/to/realpde

The loader finds the two directories at any depth below that root. Files are
`<Re>_<aoa>.h5` with datasets `u` and `v`. One file the organisers flagged as a
duplicate is excluded in `config.py`.

## Install

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt

Torch is not pinned to a CUDA build; install the wheel that matches your driver.

## Run

    python scripts/run_baselines.py     # 8 baselines, 5 seeds -> runs/baselines.json
    python scripts/run_budget.py        # Table 1              -> runs/budget.json
    python scripts/run_mend.py          # MEND + ablations     -> runs/all_results.json
    python scripts/make_tables.py       # LaTeX tables

`run_baselines.py` also writes `runs/frozen.pt`, the frozen weights the other two
scripts wrap. Add `--quick` for a smoke test: one seed, two epochs, few windows.
It exercises every code path in a couple of minutes and produces numbers that
mean nothing.

The full five-seed run takes several hours on one GPU, most of it in the five
fine-tuning runs per seed.

## Layout

    mend/config.py     constants, benchmark and training configuration
    mend/data.py       loading, masking, clock alignment, window cutting
    mend/metrics.py    Rel-L2, TKE, MVPE, SPS
    mend/evaluate.py   one scorer for every method, interval fitting, aggregation
    mend/classical.py  Kennedy-O'Hagan, quantile mapping, delay-embedded DMD
    mend/nets.py       FNO backbone, auxiliary head, losses, training loop
    mend/transfer.py   direct fine-tune, L2-SP, EWC, PhysGuard, PNO
    mend/correct.py    MEND: auxiliary heads, the correction, coefficient search
    mend/budget.py     the error budget
    mend/tables.py     LaTeX emitters

## Notes on the evaluation

Scoring happens in m/s, not in normalised units, because SPS carries an absolute
constant fixed by the organisers. Models train on velocities divided by the
inflow speed implied by each Reynolds number; predictions are converted back
before any metric is computed.

The simulator and the camera do not share a clock. The simulator is
nondimensional with U = L = 1; the measurement is in seconds at 50 fps. The
ratio of simulator steps to camera frames runs from 0.7 to 5.0 across this
dataset, so `Bench.sim_resampled` interpolates the simulator onto the camera's
frame grid before either side is used.

Splits are by Reynolds number. No test condition appears in training at any
angle of attack, so every reported number is transfer rather than interpolation.

Every method forms intervals the same way: a per-point spread fitted on
validation, and an interval level chosen from a shared grid on validation. This
is deliberate. SPS mixes accuracy with interval quality, and without a common
construction a method could win on the interval alone.

## Reproducing the paper's numbers

Seeds 0-4, defaults as committed. The appendix tables come from a separate
three-seed run of the same scripts with `--seeds 0 1 2`; that run trains every
row under identical conditions so the ablations can be read against one another,
which is why its absolute numbers differ slightly from the five-seed table.

Table 1 is computed on one fine-tuned operator from a separate diagnostic run,
so its uncorrected relative L2 is not comparable with the main results table.
Only the differences down each of its columns are meaningful.
