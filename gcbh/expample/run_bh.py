import sys
import os
import numpy as np
from ase.io import read
from gcbh2 import GrandCanonicalBasinHopping
from pygcga2 import randomize, rand_clustering
from catkit.gen.surface import SlabGenerator
from ase.build import bulk
from catkit.gen.adsorption import AdsorptionSites
from ase.constraints import FixAtoms
import random
from ase import Atoms
from pygcga.checkatoms import CheckAtoms
from ase.build import root_surface, fcc111_root
from catkit.gratoms import Gratoms
import json
import glob

filescopied=['opt.py'] # files required to complete an optimization
bond_range={('C','Pt'):[1.2,10],('Pt','Pt'):[1,10.0],('C','C'):[1.9,10],('C','O'):[0.6,10],('Pt','O'):[1.5,10],('O','O'):[1.9,10]}
name = glob.glob('start.traj')
slab_clean = read(name[0])

bh_run=GrandCanonicalBasinHopping(temperature=2000.0, t_nve=700,atoms=slab_clean, bash_script="optimize.sh", files_to_copied=filescopied, restart=True, chemical_potential="chemical_potentials.dat")

cell = slab_clean.get_cell()
a = cell[0,0]
b = cell[1,0]
c = cell[1,1]
tol = 1.5
boundary = np.array([[-tol,-tol],[a+tol,-tol],[a+b+tol,c+tol],[b-tol,c+tol]])

bh_run.add_modifier(randomize, name="randomize", dr=1, bond_range=bond_range, max_trial=50, weight=1)
bh_run.add_modifier(rand_clustering, name="rand_cluster", dr=1, n=5, boundary=boundary , bond_range=bond_range, max_trial=100, weight=1)

bh_run.run(4000)
