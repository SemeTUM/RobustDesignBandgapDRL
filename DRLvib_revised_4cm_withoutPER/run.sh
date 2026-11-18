#!/bin/bash
export COMSOL_ROOT=/share/programs/comsol/comsol63

export PATH="$COMSOL_ROOT/bin:$PATH"

exec /home/sgebreki/.conda/envs/comsolenv/bin/python discrete4_wo_PER_steel_sac.py
