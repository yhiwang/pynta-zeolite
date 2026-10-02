#!/usr/bin/env python
"""DFT step 4: isolated-atom single points, the E0s MACE needs.

    python dft/04_isolated_atoms.py      # ATOM_SUBMIT = False: write inputs + job.sh only

One job per element in ATOM_ELEMENTS, in ``<RUN_DIR>/DFT_atoms/<element>/``:
a single atom in an ATOM_BOX vacuum box, with the step 2 settings (SP_VASP)
plus ATOM_VASP_CHANGES -- spin polarized, NUPDOWN fixed to ATOM_MAGMOM so
every atom sits in its ground spin state. The D3 term is zero for a single
atom, so keeping IVDW changes nothing but keeps the settings identical.

Same bookkeeping as step 2 (job.id guard; done / in the queue / failed).
Step 3 picks the finished atoms up and adds them to train.extxyz as
``config_type=IsolatedAtom``, which is where MACE reads its E0s from.
"""

import importlib
import os
import sys

import _common  # noqa: F401
import dft_settings as ds
import layout
import settings
from ase import Atoms
from ase.calculators.vasp import Vasp
from ase.io import write
from vasp_io import finished

submit_sp = importlib.import_module("02_submit_sp")     # reuse its job template
ATOMS_DIR = "DFT_atoms"                                 # <RUN_DIR>/DFT_atoms/<element>/


def write_job(folder, element):
    magmom = ds.ATOM_MAGMOM[element]
    atoms = Atoms(element, positions=[[c / 2 for c in ds.ATOM_BOX]],
                  cell=ds.ATOM_BOX, pbc=True)
    atoms.set_initial_magnetic_moments([magmom])
    atoms.info.update(element=element, config_type="IsolatedAtom")
    os.makedirs(folder, exist_ok=True)
    write(os.path.join(folder, layout.DFT_CONFIG), atoms, format="extxyz")
    params = dict(ds.SP_VASP, **ds.ATOM_VASP_CHANGES)
    params["nupdown"] = magmom
    Vasp(directory=folder, **params).write_input(atoms)
    with open(os.path.join(folder, layout.JOB_SCRIPT), "w") as handle:
        handle.write(submit_sp.JOB_TEMPLATE.format(
            name="atom_%s" % element, dependency="",
            account=ds.SP_SLURM_ACCOUNT, partition=ds.SP_SLURM_PARTITION,
            tasks=ds.ATOM_SLURM_TASKS, memory=ds.SP_SLURM_MEMORY,
            time=ds.ATOM_SLURM_TIME, err=layout.JOB_ERR,
            module=ds.VASP_MODULE, binary=ds.VASP_BINARY))


def main():
    potpaw = os.path.join(ds.VASP_PP_PATH, "potpaw_PBE")
    if not os.path.isdir(potpaw):
        sys.exit("no POTCARs at %s -- link the lab set there first" % potpaw)
    os.environ["VASP_PP_PATH"] = ds.VASP_PP_PATH
    root = os.path.join(settings.RUN_DIR, ATOMS_DIR)
    in_queue = submit_sp.queued_ids()

    for element in ds.ATOM_ELEMENTS:
        folder = os.path.join(root, element)
        if finished(folder):
            print("%-3s done" % element)
            continue
        id_path = os.path.join(folder, layout.JOB_ID)
        if os.path.isfile(id_path):
            with open(id_path) as handle:
                job_id = handle.read().strip()
            if in_queue is None or job_id in in_queue:
                print("%-3s in the queue (job %s)" % (element, job_id))
            else:
                print("%-3s failed (job %s): see %s/vasp.out, delete job.id to resubmit"
                      % (element, job_id, folder))
            continue
        write_job(folder, element)
        if ds.ATOM_SUBMIT:
            print("%-3s submitted, job %s" % (element, submit_sp.submit(folder)))
        else:
            print("%-3s would submit (%s)" % (element, folder))

    if not ds.ATOM_SUBMIT:
        print("\ndry run: check one INCAR (ISPIN = 2, NUPDOWN), then set "
              "ATOM_SUBMIT = True in dft/dft_settings.py")


if __name__ == "__main__":
    main()