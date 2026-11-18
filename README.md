# Robust Design of Broadband Bandgap Structures by Soft Actor–Critic Reinforcement Learning

This repository contains the code used to train and validate the agent, and to generate the data for the manuscript *"Robust Design of Broadband Bandgap Structures by Soft Actor–Critic Reinforcement Learning"*.

## Repository structure

- **`envs/`** – COMSOL files and the environment used by the agent.
- **`comsolfile/`** – COMSOL simulation setups (`.mph`) for the 4 cm × 4 cm design spaces.

The names of the remaining folders, which start with `DRLvib_revised_`, indicate the size of the design domain and the base material. For example, `DRLvib_revised_3cm_steel_refined` contains the 3 cm × 3 cm steel design space discretized into a 60 × 60 grid.

Each training and validation folder also contains a `log` file with the output recorded during training or validation.

# Bandgap Cell Finder

A browser application for selecting and generating p4m-symmetric unit cells with phononic bandgaps, based on the discrete soft actor–critic (SAC) agent described in

> S. B. Gebrekidan, F. Toth, *Design of broadband bandgap structures by soft actor–critic reinforcement learning* (submitted).

The application works with two kinds of data produced by the training code in this repository:

- **Design libraries**: every unit cell simulated in COMSOL during training or evaluation, together with its dispersion bands. The app ranks these cells by how well their bandgaps cover a user-defined frequency range.
- **Trained agent**: the actor network of the SAC agent. The app runs the walking environment in the browser and generates new unit cells without finite element analysis.

Everything runs client-side in a single HTML file. No server, Python installation, or COMSOL licence is needed to use the app.

## Quick start

1. Open `bandgap_cell_finder.html` in a web browser, or serve it with GitHub Pages.
2. Convert *.npy to .json using `export_for_app.py` (see below).
3. Enter a target stop band in **Target stop band** (e.g. 3000–3500 Hz), and set the cell size and material in **Cell size & material**.
4. Click a ranked design to inspect its unit cell and dispersion diagram, then copy its geometry for COMSOL.

## Modes

### Verified designs

Ranks COMSOL-verified unit cells from one or more design libraries against a target stop band.

| Control | Description |
|---|---|
| Design libraries | Load one or more `*.json` libraries (several files can be selected at once). Each library is listed with its reference cell size, E, ρ, and ν, and can be switched off or removed. |
| Target stop band | Lower and upper frequency of the range the bandgap should cover (Hz). |
| Rank by | *Closest match* (coverage, then edge mismatch), *Most margin* (coverage, then margin around the range), or *Widest gap* (gap-to-midgap ratio). |
| Minimum coverage | Hides designs whose best bandgap covers less than this fraction of the range. |
| Cell size & material | Cell size *a* (cm; 3 cm and 4 cm buttons), base material (soft nylon, steel, or custom E, ρ, ν). Library frequencies are rescaled to these values. |
| Fit the cell size | Instead of a fixed size, computes for each design the cell size that centres the target range in its bandgap. |
| ν tolerance / approximate | Libraries are only used when their Poisson's ratio matches the target within the tolerance; the checkbox also admits libraries with a different ν (marked as approximate). |

### Generate with agent

Runs the trained actor in a JavaScript copy of the walking environment.

| Control | Description |
|---|---|
| Trained agent | Load `agent.json`. Actors with 2 inputs (position only) and with 2 + h × h inputs (position + occupancy grid) are supported; the latter run only on their training grid. |
| Cell size & material | Sets the physical cell size (and grid-cell size) and the material for which matching library data are shown. The agent's state contains no material or size information, so these settings do not change the generated geometry. |
| Grid, steps, designs, seed | Full-cell grid (e.g. 40 × 40), steps per episode (500 in training), number of designs, random seed. |
| Actions | *Sampled* (stochastic policy) or *Most likely* (greedy). |
| Start row | Random row of the first column, as in training, or a fixed row. |

Generated cells are checked for connectivity and deduplicated. If a generated cell already exists in a loaded library with a matching Poisson's ratio, its stored dispersion bands are shown; otherwise it is marked as *not simulated*. **Generated designs must be verified in COMSOL before use.**

### Inspector and export

For the selected design, the inspector shows:

- the full p4m unit cell, optionally tiled 3 × 3 to check connectivity across cell boundaries;
- the dispersion diagram along Γ–X–M–Γ with the matched bandgap and the target range shaded, and a cursor readout of wave vector and frequency;
- the gap edges, gap-to-midgap ratio, cell size, library, and episode.

Export options (copied to the clipboard):

- **COMSOL selection**: the void pixels of the top-left quadrant as `arr1(r,c) … arr4(r,c)` identifiers, in the same format as `GridEnv.conv2com`.
- **Grid (CSV)**: the full 40 × 40 cell (1 = void, 0 = solid).
- **Connected designs (JSON)**: all connected generated designs with cell size, material, and any library match, for batch simulation.

## Preparing data: `export_for_app.py`

The history (`.npy`) and checkpoint (`.pth`) files are Python pickles that a browser cannot read. Convert them once:

```bash
# one library per material and design space
python export_for_app.py --history nylon4cm_history.npy --material nylon --size-cm 4 --library-out lib_nylon_4cm.json
python export_for_app.py --history steel4cm_history.npy --material steel --size-cm 4 --library-out lib_steel_4cm.json
python export_for_app.py --history steel3cm_history.npy --material steel --size-cm 3 --library-out lib_steel_3cm.json

# trained actor
python export_for_app.py --checkpoint SAC_DISCRETE_PER_WO_frame_agent_checkpoint_ep2000.pth --agent-out agent.json
