# Robust Design of Broadband Bandgap Structures by Soft Actor–Critic Reinforcement Learning

This repository contains the code used to train and validate the agent, and to generate the data for the manuscript *"Robust Design of Broadband Bandgap Structures by Soft Actor–Critic Reinforcement Learning"*.

## Repository structure

- **`envs/`** – COMSOL files and the environment used by the agent.
- **`comsolfile/`** – COMSOL simulation setups (`.mph`) for the 4 cm × 4 cm and 3 cm × 3 cm design spaces.

The names of the remaining folders, which start with `DRLvib_revised_`, indicate the size of the design domain and the base material. For example, `DRLvib_revised_3cm_steel_refined` contains the 3 cm × 3 cm steel design space discretized into a 60 × 60 grid.

Each training and validation folder also contains a `log` file with the output recorded during training or validation.
