from pynta.mol import *
# CHANGED: removed `from pysidt.sidt import read_nodes`
# CHANGED: removed `from pynta.sidt import SurfaceBondLengthSIDT`
#          SIDT is a subgraph-isomorphic decision tree fit on metal-surface data.
#          It needs a `metal` identity and has never seen Al-O-Si chemistry, so it
#          would fall back to its Root group. Replaced by a covalent-radii sum.
from pynta.geometricanalysis import generate_site_molecule
# KEPT: still used to build slabmol (the sites-as-a-graph object). Verified to work
#       unchanged on ZeoliteBare sites -- it only reads site["site"],
#       site["morphology"] and site_adjacency, all of which your sites have.
from pynta.calculator import map_harmonically_forced
from ase.data import covalent_radii, atomic_numbers
# ADDED: covalent radii table, used to estimate the site-to-adatom bond length.
import logging 
import os 
import pynta
import json
from joblib import Parallel, delayed
from ase import Atoms
from functions_v2 import rotate_free, closest_framework_per_atom
from mol import add_adsorbate_to_site_z


def construct_initial_guess_files(mol,mol_name,pynta_path,slab,
                               single_site_bond_params_lists,single_sites_lists,double_site_bond_params_lists,double_sites_lists,
                               Eharmtol,Eharmfiltertol,Nharmmin,slab_sites,site_adjacency,pbc,nslab,harm_f_software,harm_f_software_kwargs,
                               nprocs):
    # CHANGED: dropped `metal` (it only fed SIDT). slab_sites and site_adjacency are
    #          KEPT, because generate_site_molecule still needs them.
    """Generate and write initial guesses for adsorbate optimization

    Args:
        mol (_type_): Molecule representation of adsorbate
        mol_name (_type_): name of the adsorbate
        pynta_path (_type_): path to the pynta run directory
        slab (_type_): zeolite ase.Atoms object (your ZeoliteBare.atoms)
        single_site_bond_params_lists (_type_): list of site_bond_params for individual unique sites
        single_sites_lists (_type_): list of individual unique sites
        double_site_bond_params_lists (_type_):  list of site_bond_params for unique pairs of sites
        double_sites_lists (_type_): list of unique pairs of sites
        Eharmtol (_type_): harmonic tolerance for automatic selection
        Eharmfiltertol (_type_): harmonic tolerance for removal
        Nharmmin (_type_): target number of selected configurations after harmonic filtering
        slab_sites (_type_): list of all sites
        site_adjacency (_type_): site adjacency information
        pbc (_type_): periodic boundary tuple
        nslab (_type_): number of framework atoms
        harm_f_software (_type_): software used for harmonic optimizations
        harm_f_software_kwargs (_type_): keyword arguments for harmonic optimizations

    Returns:
        list of files with the corresponding xyzs
    """
    if os.path.exists(os.path.join(pynta_path,"Adsorbates",mol_name)):
        logging.info("Found existing path {0} for {1}".format(os.path.join(pynta_path,"Adsorbates",mol_name),mol_name))
        return [os.path.join(pynta_path,"Adsorbates",mol_name,str(prefix),str(prefix)+"_init.xyz") for prefix in os.listdir(os.path.join(pynta_path,"Adsorbates",mol_name)) if os.path.isdir(os.path.join(pynta_path,"Adsorbates",mol_name,prefix))]
    
    surf_sites = mol.get_surface_sites()

    ads,mol_to_atoms_map = get_adsorbate(mol)
    
    if len(surf_sites) == 0:
        ads.pbc = pbc
        ads.center(vacuum=10)
        structs = [ads]
        admols = []
        # ADDED: gas-phase species have no site graphs. Empty list so the variable
        #        exists for the caller regardless of branch.
        gratom_to_molecule_atom_map = {val:key for key,val in mol_to_atoms_map.items()}
        gratom_to_molecule_surface_atom_map = dict()
    else:
        structs,admols = generate_adsorbate_guesses(mol,ads,slab,mol_to_atoms_map,
                                single_site_bond_params_lists,single_sites_lists,
                                double_site_bond_params_lists,double_sites_lists,
                                Eharmtol,Eharmfiltertol,Nharmmin,slab_sites,site_adjacency,harm_f_software,harm_f_software_kwargs,
                                nprocs=nprocs)
        # CHANGED: dropped `metal` from the call, and now unpacks two values --
        #          generate_adsorbate_guesses returns the admol graphs alongside
        #          the structures so they stay available downstream.


        gratom_to_molecule_atom_map = {val:key for key,val in mol_to_atoms_map.items()}

        surf_index_atom_map = dict()
        for i,atm in enumerate(mol.atoms):
            if atm.is_bonded_to_surface():
                surf_index_atom_map[mol_to_atoms_map[i]] = i
                # NOTE: the copy you pasted had `surf_index_atom_map[[i]] = i`, which
                # raises TypeError (a list is unhashable). Restored pynta's original.

        gratom_to_molecule_surface_atom_map = surf_index_atom_map
        
    xyzs = []
    for i,structure in enumerate(structs):
        prefix = i
        try:
            os.makedirs(os.path.join(pynta_path,"Adsorbates",mol_name,str(prefix)))
        except:
            pass
        xyz = os.path.join(pynta_path,"Adsorbates",mol_name,str(prefix),str(prefix)+"_init.xyz")
        xyzs.append(xyz)
        write(xyz,structure)
        sp_dict = {"name":mol_name, "adjlist":mol.to_adjacency_list(),"atom_to_molecule_atom_map": gratom_to_molecule_atom_map,
                "gratom_to_molecule_surface_atom_map": gratom_to_molecule_surface_atom_map, "nslab": nslab}
        with open(os.path.join(pynta_path,"Adsorbates",mol_name,"info.json"),'w') as f:
            json.dump(sp_dict,f)
        
    return xyzs
    # DECIDE: admols is built and returned by generate_adsorbate_guesses but is not
    # persisted here. If you want the graphs on disk for later inspection, dump
    # [a.to_adjacency_list() for a in admols] into the info.json above, or return
    # them from this function too.
                
def generate_adsorbate_guesses(mol,ads,slab,mol_to_atoms_map,
                               single_site_bond_params_lists,single_sites_lists,double_site_bond_params_lists,double_sites_lists,
                               Eharmtol,Eharmfiltertol,Nharmmin,slab_sites,site_adjacency,harm_f_software,harm_f_software_kwargs,nprocs=1):
    # CHANGED: dropped `metal` (only fed SIDT).
    # CHANGED: now returns (xyzsout, admols) instead of just xyzsout, so the graph
    #          representation of each candidate configuration stays accessible.
    mol_surf_inds = [mol.atoms.index(a) for a in mol.get_adatoms()]
    atom_surf_inds = [mol_to_atoms_map[i] for i in mol_surf_inds]
    nslab = len(slab)
    slabmol,neighbor_sites,ninds = generate_site_molecule(slab, slab_sites, slab_sites, site_adjacency, max_dist=None)
    admols = []
    surface_site_indices = []
    if len(atom_surf_inds) == 1:
        site_bond_params_lists = deepcopy(single_site_bond_params_lists)
        sites_lists = single_sites_lists
        for i,site_bond_params_list in enumerate(site_bond_params_lists):
            site_bond_params_list[0]["ind"] = atom_surf_inds[0]+len(slab)
            
            admol = slabmol.copy(deep=True)
            m = mol.copy(deep=True)
            m_site = m.get_surface_sites()[0]
            bd = list(m_site.bonds.values())[0]
            order = bd.order 
            m_adatom = list(m_site.bonds.keys())[0]
            admol_site_index = [j for j,s in enumerate(neighbor_sites) if sites_match(s,sites_lists[i][0],slab)][0]
            admol_site = admol.atoms[admol_site_index]
            m.remove_bond(m.get_bond(m_site,m_adatom))
            m.remove_atom(m_site)
            admol = admol.merge(m)
            admol.add_bond(Bond(m_adatom,admol_site,order=order))
            admol.update_multiplicity()
            admol.update_atomtypes()
            admol.update_connectivity_values()
            admol.identify_ring_membership()
            admols.append(admol)
            surface_site_indices.append([admol_site_index])
            # KEPT unchanged. In pynta this graph was SIDT's input feature. It is no
            # longer consumed by the height estimate, but it is the only graph-level
            # description of "this adsorbate on this site", so it is retained and
            # returned for inspection / future placement methods.
            
            #add up pulling potential
            for ind in range(len(ads)):
                if ind in atom_surf_inds:
                    continue
                pos = deepcopy(site_bond_params_list[0]["site_pos"])
                pos = np.array(pos) + 8.5*np.array(sites_lists[i][0]["normal"])
                # CHANGED: was `pos[2] += 8.5`. That hardcodes +z as "away from the
                # surface", true for a slab lying in the xy-plane. In a pore the
                # outward direction is the site's Al-O + Si-O bisector, stored in
                # the site dict as "normal".
                # DECIDE: 8.5 Angstrom along any direction in a MOR pore probably
                # lands inside the opposite wall. Consider a smaller value, or
                # deleting this loop -- the pore already confines the tail.
                site_bond_params_list.append({"site_pos": pos,"ind": ind+len(slab), "k": 0.1, "deq": 0.0})

    elif len(atom_surf_inds) == 2:
        site_bond_params_lists = deepcopy(double_site_bond_params_lists)
        sites_lists = double_sites_lists
        for i,site_bond_params_list in enumerate(site_bond_params_lists):
            site_bond_params_list[0]["ind"] = atom_surf_inds[0]+len(slab)
            site_bond_params_list[1]["ind"] = atom_surf_inds[1]+len(slab)
            
            admol = slabmol.copy(deep=True)
            m = mol.copy(deep=True)
            m_sites = m.get_surface_sites()
            bds = [list(m_site.bonds.values())[0] for m_site in m_sites]
            orders = [bd.order for bd in bds]
            m_adatoms = [list(m_site.bonds.keys())[0] for m_site in m_sites]
            admol_site_indices = [[j for j,s in enumerate(neighbor_sites) if sites_match(s,sites_lists[i][0],slab)][0],[j for j,s in enumerate(neighbor_sites) if sites_match(s,sites_lists[i][1],slab)][0]]
            admol_sites = [admol.atoms[admol_site_index] for admol_site_index in admol_site_indices]
            for k,m_site in enumerate(m_sites):
                m.remove_bond(m.get_bond(m_site,m_adatoms[k]))
                m.remove_atom(m_site)
            admol = admol.merge(m)
            for k,m_adatom in enumerate(m_adatoms):
                admol.add_bond(Bond(m_adatom,admol_sites[k],order=orders[k]))
            # CHANGED: pynta reuses `i` as the inner loop variable here, shadowing the
            # outer candidate index. Renamed to `k`. Harmless in pynta (i is not read
            # again in the body) but a real trap for anyone editing this block.
            admol.update_multiplicity()
            admol.update_atomtypes()
            admol.update_connectivity_values()
            admol.identify_ring_membership()
            admols.append(admol)
            surface_site_indices.append(admol_site_indices)
    else:
        raise ValueError("Only monodentate and bidentate guesses currently allowed. The infrastructure can support tridenate and higher, but the filtering process may be very expensive above bidentate.")


    mol_fixed_bond_pairs = [[mol.atoms.index(bd.atom1),mol.atoms.index(bd.atom2)] for bd in mol.get_all_edges() if (not bd.atom1.is_surface_site()) and (not bd.atom2.is_surface_site())]
    atom_fixed_bond_pairs = [[mol_to_atoms_map[pair[0]]+len(slab),mol_to_atoms_map[pair[1]]+len(slab)]for pair in mol_fixed_bond_pairs]
    constraint_list = [{"type": "FixBondLength", "a1": pair[0], "a2": pair[1]} for pair in atom_fixed_bond_pairs]+["freeze slab"]
    # REMOVED: nodes_file / read_nodes / SurfaceBondLengthSIDT model loading.
    
    geos = []
    for i,sites_list in enumerate(sites_lists):
        geo,h1,h2 = place_adsorbate(ads,slab,atom_surf_inds,sites_list)
        # CHANGED: dropped admols[i], surface_site_indices[i], metal, sidt_model.
        #          The graphs are still built above, just no longer threaded into a
        #          function that would ignore them.
        if h1:
            site_bond_params_lists[i][0]["site_pos"] = np.array(site_bond_params_lists[i][0]["site_pos"]) + h1*np.array(sites_list[0]["normal"])
            # CHANGED: was `site_bond_params_lists[i][0]["site_pos"][2] += h1`.
            # This moves the harmonic spring ANCHOR from the site position out to
            # where the bonded atom should sit. Along +z it lands in the framework
            # wall; along the site normal it lands in the pore. Without this fix the
            # adsorbate is PLACED correctly (add_adsorbate_to_site reads "normal")
            # but the spring then drags it into the wall during relaxation.
        if h2:
            site_bond_params_lists[i][1]["site_pos"] = np.array(site_bond_params_lists[i][1]["site_pos"]) + h2*np.array(sites_list[1]["normal"])
            # CHANGED: same fix for the second site of a bidentate pair.
        geos.append(geo)
    # add these two lines temporarily, right here:
    return geos, admols
    
    print("initial geometries")
    print(len(geos))
    geos_out = []
    Eharms = []
    site_bond_params_lists_out = []
    
    inputs = [ (geos[j],[],site_bond_params_lists[j],nslab,constraint_list,None,j,mol_to_atoms_map,None,harm_f_software,harm_f_software_kwargs) for j in range(len(geos))]

    outputs = Parallel(n_jobs=nprocs)(delayed(map_harmonically_forced)(inp) for inp in inputs)
    # DECIDE: harm_f_software is "TBLite" in pynta. Your pipeline uses MACE.
    # Either pass a MACE backend here (if pynta.calculator supports one), or
    # `return geos,admols` just above and hand the geos to your own mace_opt.py.
    for i in range(len(outputs)):
        geo_out,Eharm,_ = outputs[i]
        if geo_out:
            del geo_out["constraints"]
            geo_out = Atoms(**geo_out)
            geo_out.calc = None
            geos_out.append(geo_out)
            Eharms.append(Eharm)
            site_bond_params_lists_out.append(site_bond_params_lists[i])

    print("optimized geometries")
    print(len(geos_out))
    inds = get_unique_sym_struct_indices(geos_out)

    print("after symmetry")
    print(len(inds))

    geos_out = [geos_out[ind] for ind in inds]
    Eharms = [Eharms[ind] for ind in inds]

    if len(atom_surf_inds) == 1: #should be small, don't bother filtering
        xyzsout = geos_out
        site_bond_params_lists_final = [site_bond_params_lists_out[ind] for ind in inds]
        return xyzsout,admols
        # CHANGED: returns the admol graphs too. Note admols is indexed by CANDIDATE
        # SITE (one per entry of sites_lists, before any relaxation or filtering),
        # so it does NOT line up one-to-one with xyzsout, which has been filtered.
    else:
        Einds = np.argsort(np.array(Eharms))
        Emin = np.min(np.array(Eharms))
        xyzsout = []
        site_bond_params_lists_final = []
        for Eind in Einds:
            if Eharms[Eind]/Emin < Eharmtol: #include all TSs with energy close to Emin
                xyzsout.append(geos_out[Eind])
                site_bond_params_lists_final.append(site_bond_params_lists_out[Eind])
            elif Eharms[Eind]/Emin > Eharmfiltertol: #if the energy is much larger than Emin skip it
                continue
            elif len(xyzsout) < Nharmmin: #if the energy isn't similar, but isn't much larger include the smallest until Nharmmin is reached
                xyzsout.append(geos_out[Eind])
                site_bond_params_lists_final.append(site_bond_params_lists_out[Eind])

    return xyzsout,admols
    # CHANGED: same, for the bidentate branch.

def place_adsorbate(ads,slab,atom_surf_inds,sites):
    # CHANGED: dropped admol, admol_site_indices, metal, sidt_model.
    #          They existed only so SIDT could compute h. The graphs still get built
    #          in generate_adsorbate_guesses; they just don't come in here.
    if len(atom_surf_inds) == 1:
        geo = slab.copy()
        h = estimate_surface_bond_length(slab,ads,sites[0],atom_surf_inds[0])
        # CHANGED: was estimate_surface_bond_length(admol,admol_site_indices[0],sidt_model,metal)
        add_adsorbate_to_site_z(geo, ads, atom_surf_inds[0], sites[0], height=h)
        nslab = len(slab)
        placed_ads = geo[nslab:]
        framework = geo[:nslab]
        bind = atom_surf_inds[0]
        per_atom = closest_framework_per_atom(placed_ads, framework)
        print(f"\n--- placed {len(placed_ads)}-atom adsorbate, binding atom = {bind} ---")
        for a, (f_idx, dist) in enumerate(per_atom):
            marker = " <-- binding" if a == bind else ""
            print(f"  ads[{a}] ({placed_ads[a].symbol}): nearest fw[{f_idx}] = {dist:.2f} A{marker}")

        return geo,h,None
    elif len(atom_surf_inds) == 2:
        geo = slab.copy()
        h1 = estimate_surface_bond_length(slab,ads,sites[0],atom_surf_inds[0])
        h2 = estimate_surface_bond_length(slab,ads,sites[1],atom_surf_inds[1])
        # CHANGED: same substitution for both binding atoms.
        ori = get_mic(sites[0]['position'], sites[1]['position'], geo.cell)
        add_adsorbate_to_site_z(geo, deepcopy(ads), atom_surf_inds[0], sites[0], height=h1, orientation=ori)
        if np.isnan(geo.positions).any(): #if nans just ignore orientation and let it optimize
            geo = slab.copy()
            add_adsorbate_to_site(geo, ads.copy(), atom_surf_inds[0], sites[0], height=h1, orientation=None)
        return geo,h1,h2
    else:
        raise ValueError
    

def estimate_surface_bond_length(slab,ads,site,atom_surf_ind):
    # CHANGED: complete replacement of the SIDT evaluation.
    #
    # Old signature: (admol, admol_site_index, sidt_model, metal)
    # Old behaviour: label the site atom and the adatom in the 2D admol graph, run
    #                the subgraph-isomorphic decision tree, return the learned bond
    #                length. The tree was fit on metal-surface data and needs a
    #                `metal` identity, so for Al-O-Si sites it would match nothing
    #                and fall back to its Root group (pynta logs a warning for this).
    #
    # New behaviour: sum the covalent radii of the two atoms that will be bonded.
    # BOTH symbols are looked up dynamically from the structures, so this works for
    # any framework atom (O here, but nothing assumes it) and any binding element --
    # not just O and C. Crude, but this is only an initial guess which the harmonic
    # relaxation then refines.
    #
    # NOTE the one trade-off: SIDT could give context-dependent lengths (same C on a
    # bridge vs an ontop site). Covalent radii cannot -- C-on-O is always the same
    # number. Acceptable here because all your sites are chemically similar BAS
    # oxygens and the guess gets relaxed anyway.
    site_symbol = slab[site["indices"][0]].symbol   # framework atom at this site
    ads_symbol  = ads[atom_surf_ind].symbol         # the adsorbate's binding atom
    L = covalent_radii[atomic_numbers[site_symbol]] + covalent_radii[atomic_numbers[ads_symbol]]
    return L