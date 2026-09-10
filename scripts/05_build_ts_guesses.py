#!/usr/bin/env python
"""Step 5 -- build transition-state guesses straight from the reaction graph.

Needs only the saved framework (step 0) and reaction.yaml, none of the
Adsorbates trees. For every reaction, every oxygen pair within
settings.TS_MAX_SPAN by the site rule saved with the framework ("all":
any two first-shell oxygens; "cross": only cross-Al pairs) is tried both
ways round (flip0 / flip1 = which X goes on which O); see
pyntaz/ts_graph.py for the sweep itself. Writes

    <RUN_DIR>/TS_guesses/<i>_rxn/info.json                 reaction + guess manifest
    <RUN_DIR>/TS_guesses/<i>_rxn/pair_<k>/<stem>/<stem>_init.xyz
    <RUN_DIR>/TS_guesses/<i>_rxn/pair_<k>/sweep.traj       every roll, for ase gui

one <stem> folder per surviving torsion x axial combination at its
best-clearance roll (skipped below settings.TS_CLEARANCE_MIN), in the same
shape as Adsorbates/ so the relaxation submitter can be pointed at it.
"""

import json
import os

import _common  # noqa: F401
import layout
import settings
from ase.io import write
from ase.io.trajectory import Trajectory
from pyntaz.reactions import molecule_from_adjlist
from pyntaz.ts_graph import TSGraph, PairSweep, guess_stem

run = layout.RunLayout(settings.RUN_DIR)

framework = layout.load_framework(settings.RUN_DIR)
pairs = framework.find_pairs(settings.TS_MAX_SPAN)
print("%s %s: %d atoms, Al at %s, %d oxygen pairs within %.2f A%s"
      % (framework.code, "-".join(framework.t_labels), len(framework.atoms),
         framework.al_indices, len(pairs), settings.TS_MAX_SPAN,
         " (site rule %r)" % framework.site_rule if len(framework.al_indices) > 1 else ""))
for pair_id, (site_a, site_b) in enumerate(pairs):
    bridge = framework.bridging_si(site_a, site_b)
    print("  %s  %-7s O%-4d %-7s O%-4d %.2f A%s"
          % (layout.pair_dirname(pair_id), site_a["site"], site_a["indices"][0],
             site_b["site"], site_b["indices"][0], framework.span(site_a, site_b),
             "  via " + ", ".join("%s(%d)" % (framework.label_of(t), t) for t in bridge)
             if bridge else ""))

reaction_set = layout.load_reactions(settings.REACTIONS)
sweep_settings = dict(torsion_steps=settings.TS_TORSION_STEPS,
                      roll_steps=settings.TS_ROLL_STEPS,
                      span_tolerance=settings.TS_SPAN_TOLERANCE,
                      scale_range=settings.TS_SITE_SCALE_RANGE)

for reaction in reaction_set.reactions:
    ts = TSGraph(molecule_from_adjlist(reaction["reactant"]),
                 molecule_from_adjlist(reaction["product"]),
                 stretch=settings.TS_BOND_STRETCH)
    print("\n[%d] %s   %s" % (reaction["index"], reaction["reaction"],
                              reaction.get("reaction_family", "")))
    print(ts.report())
    if len(ts.sites()) != 2:
        print("  %d site(s) -- only two-site reactions are handled yet, skipped"
              % len(ts.sites()))
        continue

    reaction_dir = os.path.join(run.ts_guesses, layout.reaction_dirname(reaction["index"]))
    os.makedirs(reaction_dir, exist_ok=True)
    manifest = {}
    first = True
    for pair_id, (site_a, site_b) in enumerate(pairs):
        oxygens = (site_a["indices"][0], site_b["indices"][0])
        pair_dir = os.path.join(reaction_dir, layout.pair_dirname(pair_id))
        os.makedirs(pair_dir, exist_ok=True)
        record = {"sites": [site_a["site"], site_b["site"]],
                  "oxygens": [int(o) for o in oxygens],
                  "span": round(framework.span(site_a, site_b), 3),
                  "guesses": {}}
        summary = []

        trajectory = Trajectory(os.path.join(pair_dir, layout.SWEEP_TRAJECTORY), "w")
        for flip, ordered in enumerate((oxygens, oxygens[::-1])):
            sweep = PairSweep(ts, framework.atoms, ordered, **sweep_settings)
            if first:
                print("  %d rotatable bond(s), %d swept %s; %d axial combination(s); "
                      "%d builds per scale and pair"
                      % (len(sweep.rotatable), len(sweep.torsion_keys),
                         sweep.torsion_keys, len(sweep.axial_grid), sweep.n_builds))
                first = False

            survivors = sweep.survivors()
            n_written = 0
            for survivor in survivors:
                rolls = sweep.rolls(survivor)
                for roll, clearance, molecule in rolls:
                    frame = framework.atoms + molecule
                    frame.info.update(molecule.info, flip=flip)
                    trajectory.write(frame)

                roll, clearance, molecule = max(rolls, key=lambda entry: entry[1])
                if clearance < settings.TS_CLEARANCE_MIN:
                    continue
                stem = guess_stem(flip, survivor.angles,
                                  sweep.axial_index(survivor.axial), roll,
                                  has_axial=bool(sweep.axial_labels))
                config_dir = os.path.join(pair_dir, stem)
                os.makedirs(config_dir, exist_ok=True)
                write(layout.initial_guess_path(config_dir), framework.atoms + molecule)
                record["guesses"][stem] = {
                    "flip": flip, "oxygens": [int(o) for o in ordered],
                    "torsions": [[list(key), angle] for key, angle
                                 in zip(sweep.torsion_keys, survivor.angles)],
                    "axial": {str(label): list(pair) for label, pair in survivor.axial.items()},
                    "scale": survivor.scale, "span": round(survivor.span, 3),
                    "roll": round(roll), "clearance": round(clearance, 3)}
                n_written += 1
            summary.append("flip%d %d/%d fit, %d written"
                           % (flip, len(survivors), sweep.n_builds, n_written))
        trajectory.close()

        print("  %s  %-7s %-7s %.2f A   %s"
              % (layout.pair_dirname(pair_id), site_a["site"], site_b["site"],
                 record["span"], "   ".join(summary)))
        manifest[layout.pair_dirname(pair_id)] = record

    info = dict(reaction)
    info.update({"ts_adjlist": ts.adjlist(),
                 "bond_scales": {"%d-%d" % key: value
                                 for key, value in ts.bond_scales().items()},
                 "max_span": settings.TS_MAX_SPAN,
                 "span_tolerance": settings.TS_SPAN_TOLERANCE,
                 "clearance_min": settings.TS_CLEARANCE_MIN,
                 "pairs": manifest})
    with open(os.path.join(reaction_dir, layout.REACTION_INFO), "w") as handle:
        json.dump(info, handle, indent=2)
    n_guesses = sum(len(record["guesses"]) for record in manifest.values())
    print("  %d guesses in %s" % (n_guesses, reaction_dir))
