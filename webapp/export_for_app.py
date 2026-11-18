"""Convert training/evaluation outputs into files the browser app can read.

The browser cannot read the pickled history .npy or the torch .pth checkpoint, so this
one-off script turns them into plain JSON. Make one library per material and design space:

  python export_for_app.py --history nylon4cm_history.npy --material nylon --size-cm 4 --library-out lib_nylon_4cm.json
  python export_for_app.py --history steel4cm_history.npy --material steel --size-cm 4 --library-out lib_steel_4cm.json
  python export_for_app.py --history steel3cm_history.npy --material steel --size-cm 3 --library-out lib_steel_3cm.json
  python export_for_app.py --history nylon3cm_history.npy --material nylon --size-cm 3 --library-out lib_nylon_3cm.json

  # trained actor (for generating designs in the browser)
  python export_for_app.py --checkpoint SAC_DISCRETE_..._checkpoint_ep2000.pth --agent-out agent.json

--material sets E, rho and nu to the values in the Methods section; override any of them with
--E0 / --rho0 / --nu0. Training histories (keys 'states', 'actions', 'final_bandgap_reward') and
evaluation histories (keys 'fund_grid', 'final_reward') are both accepted.
"""
import argparse
import json

import numpy as np

MATERIALS = {   # E [Pa], rho [kg/m^3], nu  (nylon as in comsolEnv, steel as in the Methods)
    "nylon": {"name": "Soft nylon", "E": 0.38e9, "rho": 1410.0, "nu": 0.35},
    "steel": {"name": "Steel", "E": 250e9, "rho": 7850.0, "nu": 0.28},
}


def walk(pos, action, half):
    """Same move rule as WalkingMan.step in WalkingEnv.py."""
    r, c = pos
    if action == 0: r += 1
    elif action == 1: r -= 1
    elif action == 2: c -= 1
    elif action == 3: c += 1
    if 0 <= r < half and 0 <= c <= r:
        return (r, c)
    return pos


def design_from_entry(entry, half):
    """Fundamental grid as a 0/1 array with 1 = solid (visited).

    'states' holds the position BEFORE each step, so the last cell is recovered by replaying
    the last action with the env's own move rule.
    """
    if "fund_grid" in entry:                      # env convention: 0 = visited (solid)
        fund = np.asarray(entry["fund_grid"]).reshape(half, half)
        return (fund == 0).astype(np.uint8)
    states = np.asarray(entry["states"])[:, :2].astype(int)
    actions = np.asarray(entry["actions"]).astype(int).ravel()
    path = {tuple(p) for p in states}
    path.add(walk(tuple(states[-1]), int(actions[-1]), half))
    solid = np.zeros((half, half), dtype=np.uint8)
    for r, c in path:
        if c <= r:
            solid[r, c] = 1
    return solid


def resolve_material(args):
    mat = dict(MATERIALS[args.material]) if args.material else {"name": "Custom", "E": None, "rho": None, "nu": None}
    if args.E0 is not None: mat["E"] = args.E0
    if args.rho0 is not None: mat["rho"] = args.rho0
    if args.nu0 is not None: mat["nu"] = args.nu0
    if None in (mat["E"], mat["rho"], mat["nu"]):
        raise SystemExit("Give --material nylon|steel, or all of --E0, --rho0 and --nu0.")
    return mat


def export_library(args):
    half = args.grid // 2
    mat = resolve_material(args)
    size_m = args.a0 if args.a0 is not None else args.size_cm / 100
    history = np.load(args.history, allow_pickle=True)
    designs, seen, skipped = [], set(), 0
    for idx, entry in enumerate(history):
        bands = np.real(np.asarray(entry.get(args.bands_key, 0)))
        if bands.ndim != 2:                       # disconnected designs were never simulated
            skipped += 1
            continue
        # comsolEnv table: 1 extra row (duplicate of k = 0) + 31 k-points (k = 0..3 step 0.1), columns = bands.
        if bands.shape[0] == 32:
            bands = bands[1:]
        solid = design_from_entry(entry, half)
        key = "".join(map(str, solid.ravel()))
        if key in seen:
            continue
        seen.add(key)
        reward = entry.get("final_bandgap_reward", entry.get("final_reward", np.nan))
        designs.append({
            "episode": int(entry.get("episode", idx + 1)),
            "reward": float(np.real(reward)),
            "solid": key,                                             # half*half chars, row-major
            "band_min": np.round(bands.min(axis=0), 2).tolist(),      # per band, over k
            "band_max": np.round(bands.max(axis=0), 2).tolist(),
            "bands": np.round(bands, 1).tolist(),                     # [k][band], for the band plot
        })
    label = args.label or f"{mat['name']}, {size_m * 100:g} cm"
    lib = {"format": "unitcell-library/2", "label": label, "grid": args.grid, "half": half,
           "k_points": len(designs[0]["bands"]) if designs else 0,
           "a0_m": size_m, "material": mat["name"], "E0_Pa": mat["E"], "rho0": mat["rho"], "nu0": mat["nu"],
           "designs": designs}
    with open(args.library_out, "w") as f:
        json.dump(lib, f, separators=(",", ":"))
    print(f"library '{label}': {len(designs)} unique simulated designs written to {args.library_out} "
          f"({skipped} episodes had no band data, {len(history) - skipped - len(designs)} duplicates dropped)")


def export_agent(checkpoint_path, out_path):
    import inspect
    import os
    import torch
    if not os.path.isfile(checkpoint_path):
        raise SystemExit(f"Checkpoint not found: {os.path.abspath(checkpoint_path)}")
    if "weights_only" in inspect.signature(torch.load).parameters:   # PyTorch 1.13 and newer
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except Exception:      # checkpoints holding non-tensor objects
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    else:                      # older PyTorch has no weights_only argument
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict) and "policy_state_dict" in ckpt:
        sd = ckpt["policy_state_dict"]
    elif isinstance(ckpt, dict) and "fc1.weight" in ckpt:              # actor state_dict saved on its own
        sd = ckpt
    else:
        keys = list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt).__name__
        raise SystemExit(f"No actor weights ('policy_state_dict') in {checkpoint_path}. Found: {keys}")
    layers = []
    for name in ["fc1", "fc2", "fc3", "fc4"]:
        # tensor.tolist() avoids .numpy(), which fails when PyTorch was built against a different NumPy
        layers.append({"W": sd[f"{name}.weight"].detach().cpu().tolist(), "b": sd[f"{name}.bias"].detach().cpu().tolist()})
    in_dim = len(layers[0]["W"][0])
    with open(out_path, "w") as f:
        json.dump({"format": "sac-discrete-actor/1", "in_dim": in_dim, "layers": layers}, f)
    side = int(round((in_dim - 2) ** 0.5))
    kind = "position only (r, c)" if in_dim == 2 else f"position + {side}x{side} grid"
    print(f"agent: actor with {in_dim} inputs [{kind}] written to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--history", help="history .npy (pickled list of episode dicts)")
    p.add_argument("--checkpoint", help="agent checkpoint .pth with 'policy_state_dict'")
    p.add_argument("--grid", type=int, default=40, help="full unit-cell grid size (GRID_SIZE)")
    p.add_argument("--size-cm", type=float, default=4.0, help="design-space edge length in cm")
    p.add_argument("--a0", type=float, default=None, help="design-space edge length in m (overrides --size-cm)")
    p.add_argument("--material", choices=sorted(MATERIALS), help="base material preset")
    p.add_argument("--E0", type=float, help="Young's modulus in Pa (overrides the preset)")
    p.add_argument("--rho0", type=float, help="density in kg/m^3 (overrides the preset)")
    p.add_argument("--nu0", type=float, help="Poisson's ratio (overrides the preset)")
    p.add_argument("--label", help='name shown in the app, e.g. "Steel, 3 cm" (default: material and size)')
    p.add_argument("--bands-key", default="bandgap", help="history key holding the band array")
    p.add_argument("--library-out", default="design_library.json")
    p.add_argument("--agent-out", default="agent.json")
    args = p.parse_args()
    if not args.history and not args.checkpoint:
        p.error("give --history, --checkpoint, or both")
    if args.history:
        export_library(args)
    if args.checkpoint:
        export_agent(args.checkpoint, args.agent_out)
