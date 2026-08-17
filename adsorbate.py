import os
import json

from ase.io import write
from ase.data import covalent_radii, atomic_numbers
from pynta.mol import get_adsorbate

from mol import add_adsorbate_to_site_z, add_adsorbate_to_double_site_z


def estimate_surface_bond_length(slab, ads, site, atom_surf_ind):
    site_symbol = slab[site["indices"][0]].symbol
    ads_symbol = ads[atom_surf_ind].symbol
    return (covalent_radii[atomic_numbers[site_symbol]]
            + covalent_radii[atomic_numbers[ads_symbol]])


def config_stem(tag):
    """Directory and filename stem for one orientation, from its tag.

    Zero-padded so ls sorts numerically. The stem is the only record of
    which orientation a config is, so it has to round-trip: monodentate
    angles are multiples of 15 deg and bidentate phi/psi of 15/30 deg, so
    rounding to whole degrees never collides."""
    if "angle" in tag:
        return "degrees_%03d" % round(tag["angle"])
    return "flip%d_phi%03d_psi%03d" % (int(tag["flip"]),
                                       round(tag["phi"]), round(tag["psi"]))


def place_adsorbate(ads, slab, atom_surf_inds, sites):
    if len(atom_surf_inds) == 1:
        bind = atom_surf_inds[0]
        h = estimate_surface_bond_length(slab, ads, sites[0], bind)
        structures, tags, scores = add_adsorbate_to_site_z(
            slab, ads, bind, sites[0], height=h, best_only=False)
    else:
        b1, b2 = atom_surf_inds
        h1 = estimate_surface_bond_length(slab, ads, sites[0], b1)
        h2 = estimate_surface_bond_length(slab, ads, sites[1], b2)
        structures, tags, scores = add_adsorbate_to_double_site_z(
            slab, ads, atom_surf_inds, sites, h1, h2, best_only=False)

    label = "-".join("O%d" % s["indices"][0] for s in sites)
    print("  %-12s %3d orientations  tail contact %.2f-%.2f A"
          % (label, len(structures), min(scores), max(scores)))
    return structures, tags


def generate_adsorbate_guesses(mol, ads, slab, mol_to_atoms_map,
                               single_sites_lists, double_sites_lists):
    mol_surf_inds = [mol.atoms.index(a) for a in mol.get_adatoms()]
    atom_surf_inds = [mol_to_atoms_map[i] for i in mol_surf_inds]

    if len(atom_surf_inds) == 1:
        sites_lists = single_sites_lists
    elif len(atom_surf_inds) == 2:
        sites_lists = double_sites_lists
    else:
        raise ValueError("only monodentate and bidentate are supported, got %d"
                         % len(atom_surf_inds))

    geos, site_ids, tags = [], [], []
    for i, sites_list in enumerate(sites_lists):
        structures, site_tags = place_adsorbate(ads, slab, atom_surf_inds,
                                                sites_list)
        for structure, tag in zip(structures, site_tags):
            geos.append(structure)
            site_ids.append(i)
            tag = dict(tag)
            tag["site_indices"] = [int(s["indices"][0]) for s in sites_list]
            tags.append(tag)
    return geos, site_ids, tags


def construct_initial_guess_files(mol, mol_name, pynta_path, slab,
                                  single_sites_lists, double_sites_lists):
    spdir = os.path.join(pynta_path, "Adsorbates", mol_name)
    if os.path.exists(spdir):
        print("%s: reusing %s" % (mol_name, spdir))
        found = []
        for dirpath, _, files in os.walk(spdir):
            found += [os.path.join(dirpath, f) for f in files
                      if f.endswith("_init.xyz")]
        return sorted(found)

    ads, mol_to_atoms_map = get_adsorbate(mol)

    if not mol.get_surface_sites():
        ads.pbc = slab.pbc
        ads.center(vacuum=10)
        structs, site_ids, tags = [ads], [None], [{}]
    else:
        print("\n%s" % mol_name)
        structs, site_ids, tags = generate_adsorbate_guesses(
            mol, ads, slab, mol_to_atoms_map,
            single_sites_lists, double_sites_lists)

    surf_index_atom_map = {}
    for i, atm in enumerate(mol.atoms):
        if atm.is_bonded_to_surface():
            surf_index_atom_map[mol_to_atoms_map[i]] = i

    xyzs = []
    manifest = {}
    for structure, site_id, tag in zip(structs, site_ids, tags):
        if site_id is None:
            rel = os.path.join("0", "gas")
        else:
            rel = os.path.join("%02d" % site_id, config_stem(tag))
        d = os.path.join(spdir, rel)
        os.makedirs(d, exist_ok=True)
        xyz = os.path.join(d, os.path.basename(rel) + "_init.xyz")
        write(xyz, structure)
        xyzs.append(xyz)
        manifest[rel] = tag

    sp_dict = {"name": mol_name,
               "adjlist": mol.to_adjacency_list(),
               "atom_to_molecule_atom_map": {v: k for k, v
                                             in mol_to_atoms_map.items()},
               "gratom_to_molecule_surface_atom_map": surf_index_atom_map,
               "nslab": len(slab),
               "configs": manifest}
    with open(os.path.join(spdir, "info.json"), "w") as f:
        json.dump(sp_dict, f, indent=2)

    return xyzs