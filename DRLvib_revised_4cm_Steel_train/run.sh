#!/bin/bash
export COMSOL_ROOT=/share/programs/comsol/comsol63

export PATH="$COMSOL_ROOT/bin:$PATH"

exec /home/sgebreki/.conda/envs/comsolenv/bin/python dsac_4cm_steel_wPER.py
