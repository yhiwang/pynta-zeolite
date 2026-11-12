"""Step 5 of the workflow: for every reaction, pair each relaxed unique
reactant configuration with each product configuration and lay them out as
transition-state endpoint folders::

    <run_dir>/ts_guesses/<index>_rxn/info.json          reaction record + pair manifest
    <run_dir>/ts_guesses/<index>_rxn/pair_NNNN/initial.xyz  the reactant-side minimum
    <run_dir>/ts_guesses/<index>_rxn/pair_NNNN/final.xyz    the product-side minimum

Gas-phase species are not copied: a pair holds only the two surface
endpoints, the gas names are recorded per side in the manifest and step 6
reads their structures from the unique tree.

Pair folders are numbered rather than named after their endpoints; the
manifest carries the mapping.

Two pairing rules. The X count of the reactant template decides whether the
two endpoints share a site folder (one X) or must differ (two X). On top of
that the *span* -- the largest distance between any framework oxygen of the
reactant site and any oxygen of the product site, read from framework.json
-- must not exceed ``max_span``, so with several Al a reactant on one Al is
never paired with a product on a distant one.
"""

import json
import os
import shutil

from . import runtree
from .placement import is_bidentate_stem
from .reactions import molecule_from_adjlist, is_surface_species, count_surface_sites

INITIAL_XYZ = "initial.xyz"
FINAL_XYZ = "final.xyz"
REACTION_INFO = "info.json"


def species_configs(unique_tree, name):
    """[(site, stem, path)] for one species from the unique tree; raises
    when there are none."""
    species_dir = os.path.join(unique_tree, name)
    if not os.path.isdir(species_dir):
        raise FileNotFoundError("no configs for %s -- expected %s"
                                % (name, species_dir))
    configs = runtree.config_files(species_dir)
    if not configs:
        raise FileNotFoundError("no configs for %s in %s" % (name, species_dir))
    return configs


def split_side(names, adjlists, unique_tree):
    """((surface species name, its configs), [gas names]) for one side of a
    reaction. Exactly one species per side must be bound to the framework."""
    surface, gas = [], []
    for name in names:
        if is_surface_species(molecule_from_adjlist(adjlists[name])):
            surface.append((name, species_configs(unique_tree, name)))
        else:
            gas.append(name)
    if len(surface) != 1:
        raise ValueError("side %s has %d surface species, need exactly 1"
                         % (names, len(surface)))
    return surface[0], gas


def site_oxygens(framework, site, stem):
    """Framework oxygen indices of one config: ``site`` indexes
    ``mono_sites`` for a monodentate stem and ``bi_sites`` for a bidentate
    one."""
    if is_bidentate_stem(stem):
        site_a, site_b = framework.bi_sites[int(site)]
        return [site_a["indices"][0], site_b["indices"][0]]
    return [framework.mono_sites[int(site)]["indices"][0]]


def site_span(framework, oxygens_i, oxygens_f):
    """Largest minimum-image distance between an oxygen of the initial site
    and an oxygen of the final site."""
    return max(float(framework.atoms.get_distance(i, j, mic=True))
               for i in oxygens_i for j in oxygens_f)


def build_reaction_pairs(reaction, adjlists, layout, framework, max_span,
                         verbose=True):
    """Write ``<ts_guesses>/<index>_rxn/`` for one reaction. Returns the
    number of pairs written."""
    reaction_dir = os.path.join(layout.ts_guesses, "%d_rxn" % reaction["index"])
    os.makedirs(reaction_dir, exist_ok=True)

    n_sites = count_surface_sites(molecule_from_adjlist(reaction["reactant"]))
    (initial_name, initial_configs), initial_gas = split_side(
        reaction["reactant_names"], adjlists, layout.unique)
    (final_name, final_configs), final_gas = split_side(
        reaction["product_names"], adjlists, layout.unique)

    if verbose:
        print("\n[%d] %s   (%d X -> %s-site pairing, span <= %.2f A)"
              % (reaction["index"], reaction["reaction"], n_sites,
                 "same" if n_sites == 1 else "different", max_span))
        print("  initial: %-20s %d configs   gas %s"
              % (initial_name, len(initial_configs), ", ".join(initial_gas) or "-"))
        print("  final:   %-20s %d configs   gas %s"
              % (final_name, len(final_configs), ", ".join(final_gas) or "-"))

    pairs = {}
    n_far = 0
    for site_i, stem_i, path_i in initial_configs:
        oxygens_i = site_oxygens(framework, site_i, stem_i)
        for site_f, stem_f, path_f in final_configs:
            if n_sites == 1 and site_i != site_f:
                continue
            if n_sites == 2 and site_i == site_f:
                continue
            span = site_span(framework, oxygens_i,
                             site_oxygens(framework, site_f, stem_f))
            if span > max_span:
                n_far += 1
                continue

            key = "pair_%04d" % len(pairs)
            pair_dir = os.path.join(reaction_dir, key)
            os.makedirs(pair_dir, exist_ok=True)
            shutil.copy2(path_i, os.path.join(pair_dir, INITIAL_XYZ))
            shutil.copy2(path_f, os.path.join(pair_dir, FINAL_XYZ))
            pairs[key] = {"initial": {"species": initial_name,
                                      "site": site_i, "stem": stem_i,
                                      "oxygens": oxygens_i},
                          "final": {"species": final_name,
                                    "site": site_f, "stem": stem_f,
                                    "oxygens": site_oxygens(framework, site_f, stem_f)},
                          "span": round(span, 3)}

    record = dict(reaction)
    record["gas"] = {"initial": initial_gas, "final": final_gas}
    record["max_span"] = max_span
    record["pairs"] = pairs
    with open(os.path.join(reaction_dir, REACTION_INFO), "w") as handle:
        json.dump(record, handle, indent=2)

    if verbose:
        print("  %d endpoint pairs, %d skipped for span" % (len(pairs), n_far))
    return len(pairs)


def build_all_pairs(reaction_set, layout, framework, max_span, verbose=True):
    """Step 5 for every reaction; returns {reaction index: n_pairs}."""
    return {reaction["index"]: build_reaction_pairs(
                reaction, reaction_set.adjlists, layout, framework, max_span, verbose)
            for reaction in reaction_set.reactions}