from ase.io import *
from ase.io.trajectory import TrajectoryWriter
from ase.neighborlist import neighbor_list, natural_cutoffs, NeighborList
import subprocess
import numpy as np
from pygcga.checkatoms import CheckAtoms
from ase.calculators.singlepoint import SinglePointCalculator as SPC
from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.ase import AseAtomsAdaptor
# from mpi4py import MPI
from lammps import PyLammps
from ase.calculators.lammpslib import LAMMPSlib
from ase.optimize import BFGS
import re
import os

atoms = read('input.traj')
n = len(atoms)
ase_adap = AseAtomsAdaptor()
atoms_ = ase_adap.get_structure(atoms)
ld = LammpsData.from_structure(atoms_,atom_style='atomic')
ld.write_file("struc.data")

os.system('mpirun -n 8 lmp_mpi < in.opt')

images = read('md.lammpstrj',':')
traj = TrajectoryWriter('opt.traj','a')

f = open("log.lammps","r")
Lines = f.readlines()
patten= r'(\d+\s+\-+\d*\.?\d+)'
e_pot = []
for i,line in enumerate(Lines):
    s = line.strip()
    match= re.match(patten, s)
    if match != None:
        D = np.fromstring(s, sep=' ')
        e_pot.append(D[1])

f_all=[]
for atoms in images:
    f = atoms.get_forces()
    f_all.append(f)

for i,atoms in enumerate(images):
    an = atoms.get_atomic_numbers()
    an = [78 if x==1 else x for x in an]
    an = [6 if x==2 else x for x in an]
    an = [8 if x==3 else x for x in an]
    atoms.set_atomic_numbers(an)
    traj.write(atoms, energy=e_pot[i], forces=f_all[i])

os.system('rm log.lammps md.lammpstrj struc.data')

atoms = read('opt.traj@-1')
forces = atoms.get_forces()
e = atoms.get_potential_energy()
C_ndx = [atom.index for atom in atoms if atom.symbol == 'C']
O_ndx =[atom.index for atom in atoms if atom.symbol == 'O']
Pt_ndx = [atom.index for atom in atoms if atom.symbol == 'Pt']

bond_range={('C','Pt'):[1.2,10],('Pt','Pt'):[1,10.0],('C','C'):[1.5,10],('C','O'):[0.9,10],('Pt','O'):[1.9,10],('O','O'):[1.5,10]}
inspector=CheckAtoms(bond_range=bond_range)

nat_cut = natural_cutoffs(atoms, mult=0.85)
nl = NeighborList(nat_cut, self_interaction=False, bothways=True)
nl.update(atoms)

flag=0
for c_ndx in C_ndx:
    indices, offsets = nl.get_neighbors(c_ndx)
    x = np.intersect1d(indices, Pt_ndx)
    # print(c_ndx,x)
    if x.size==0:
        flag=1
        break

cutoff = natural_cutoffs(atoms, 1.1)
nl = NeighborList(cutoff , self_interaction=False,  bothways=True)
nl.update(atoms)

O_ndx_sort=[]
for i in C_ndx:
    indices, offsets = nl.get_neighbors(i)
    O_ndx_sort.append(np.intersect1d(indices,O_ndx))

pos = atoms.get_positions()
d_co = []
atoms.set_pbc([1,1,0])
for i,j in zip(C_ndx, O_ndx_sort):
    d = atoms.get_distance(i,j, mic=True, vector=False)
    d_co.append(d)
    
d_co = np.array(d_co)
deltaf = lambda df: 4.77*df-5.37
delta = deltaf(d_co)
correction = sum(delta)

if inspector.is_good(atoms, quickanswer=True) and flag==0:
    atoms.set_calculator(SPC(atoms, energy=e+correction, forces=forces))
    write("optimized.traj",atoms) # mandatory
else:
    atoms.set_calculator(SPC(atoms, energy=100000000, forces=forces))
    write("optimized.traj",atoms) # mandatory
