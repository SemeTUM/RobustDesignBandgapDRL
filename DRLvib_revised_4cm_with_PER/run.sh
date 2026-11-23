#!/bin/bash
export COMSOL_ROOT=/share/programs/comsol/comsol63

export PATH="$COMSOL_ROOT/bin:$PATH"

exec python discrete4_nylon_sac.py
