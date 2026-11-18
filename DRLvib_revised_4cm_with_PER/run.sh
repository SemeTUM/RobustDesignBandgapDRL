#!/bin/bash
export COMSOL_ROOT=/share/programs/comsol/comsol63

export PATH="$COMSOL_ROOT/bin:$PATH"

exec /home/sgebreki/.conda/envs/comsolenv/bin/python discrete4_wobias_sac.py
