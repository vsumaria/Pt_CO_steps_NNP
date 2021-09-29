#!/usr/bin/env python
import re
import numpy as np
from ase import Atoms
from ase.build import molecule as ase_create_molecule
from ase.data import covalent_radii, chemical_symbols
from pygcga.topology import Topology
from pygcga.checkatoms import CheckAtoms
import sys, os
from pygcga.utilities import NoReasonableStructureFound
from ase.atom import Atom
from ase.constraints import FixAtoms
from ase.constraints import Hookean
from ase.constraints import FixBondLengths
from ase.constraints import FixedLine
try:
    from ase.constraints import NeverDelete
except ImportError:
    NeverDelete = type(None)
from ase.neighborlist import NeighborList
from ase.neighborlist import neighbor_list
from ase.neighborlist import natural_cutoffs
import random
import numpy as np
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.ase import AseAtomsAdaptor
from lammps import PyLammps
from ase.io import *
from ase.io.trajectory import TrajectoryWriter
import json

BOND_LENGTHS = dict(zip(chemical_symbols, covalent_radii))
debug_verbosity = True

def selecting_index(weights=None):
    """This function accept a list of float as input, all the inputs (weights) are non-negative number.
    Then, this function choose on number by the weight specified"""
    if weights is None:
        raise RuntimeError("weights is None?")
    _sumofweights = np.asarray(weights).sum()
    if _sumofweights < 1.0e-6:
        raise NoReasonableStructureFound("Sum of weights are too small, I can not find a good site")
    _w = np.array(weights)/_sumofweights
    r = np.random.uniform(low=0.0, high=1.0)
    for i in range(len(_w)):
        if i == 0 and r <= _w[0]:
            return 0
        elif sum(_w[:i+1]) >= r > sum(_w[:i]):
            return i
    return len(_w) - 1


def add_CO_on_surfaces(surface,surf_ind,element="C", zmax=1.9, zmin= 1.8, minimum_distance=0.7,
                         maximum_distance=1.4,bond_range=None,max_trials=200):
    """
    This function will add an atom (specified by element) on a surface. Z axis must be perpendicular to x,y plane.
    :param surface: ase.atoms.Atoms object
    :param element: element to be added
    :param zmin and zmax: z_range, cartesian coordinates
    :param minimum_distance: to ensure a good structure returned
    :param maximum_distance: to ensure a good structure returned
    :param max_trials: maximum nuber of trials
    :return: atoms objects
    """
    pos = surface.get_positions()
    cell=surface.get_cell()
    if np.power(cell[0:2, 2],2).sum() > 1.0e-6:
        raise RuntimeError("Currently, we only support orthorhombic cells")
    inspector=CheckAtoms(min_bond=minimum_distance,max_bond=maximum_distance,bond_range=bond_range)
    n_choose = np.random.choice(surf_ind)
    for _ in range(max_trials):
        x=pos[n_choose,0]+random.uniform(-1,1)
        y=pos[n_choose,1]+random.uniform(-1,1)
        z=pos[n_choose,2]+random.uniform(zmin,zmax)
        t=surface.copy()
        t.append(Atom(symbol=element,position=[x,y,z],tag=1))
        t.append(Atom(symbol='O',position=[x,y,z+1.175],tag=1))
        if inspector.is_good(t, quickanswer=True):
            return t
    raise NoReasonableStructureFound("No good structure found at function add_atom_on_surfaces")

def move_CO_on_surfaces(surface,surf_ind,element="C", zmax=1.9, zmin= 1.8, minimum_distance=0.7,
                         maximum_distance=1.4,bond_range=None,max_trials=200):
    """
    This function will add an atom (specified by element) on a surface. Z axis must be perpendicular to x,y plane.
    :param surface: ase.atoms.Atoms object
    :param element: element to be added
    :param zmin and zmax: z_range, cartesian coordinates
    :param minimum_distance: to ensure a good structure returned
    :param maximum_distance: to ensure a good structure returned
    :param max_trials: maximum nuber of trials
    :return: atoms objects
    """
    zmin=1.75
    an = surface.get_atomic_numbers()
    pos = surface.get_positions()
    if 6 not in an:
        raise NoReasonableStructureFound("No CO adsorbed on the surface")
    c_n=[atom.index for atom in surface if atom.symbol=='C']
    o_n=[atom.index for atom in surface if atom.symbol=='O']
    print(c_n)
    print(o_n)
    c_choose = np.random.choice(c_n)
    index = np.argwhere(c_n==c_choose)[0][0]
    print(c_choose)
    print(index)

    # nc= natural_cutoffs(surface,1.5)
    # nl = NeighborList(nc)
    # nl.update(surface)
    # indices, offsets = nl.get_neighbors(c_choose)
    # o_choose=np.intersect1d(o_n,indices)

    o_choose = o_n[index]
    del_array=[c_choose,o_choose]
    del_array = np.flip(np.sort(del_array),axis=0)
    for i in del_array:
        del surface[i]

    pos = surface.get_positions()
    cell=surface.get_cell()
    # if np.power(cell[0:2, 2],2).sum() > 1.0e-6:
    #     raise RuntimeError("Currently, we only support orthorhombic cells")
    inspector=CheckAtoms(min_bond=minimum_distance,max_bond=maximum_distance,bond_range=bond_range)
    n_choose = np.random.choice(surf_ind)
    for _ in range(max_trials):
        x=pos[n_choose,0]+random.uniform(-1,1)
        y=pos[n_choose,1]+random.uniform(-1,1)
        z=pos[n_choose,2]+random.uniform(zmin,zmax)
        t=surface.copy()
        t.append(Atom(symbol=element,position=[x,y,z],tag=1))
        t.append(Atom(symbol='O',position=[x,y,z+1.175],tag=1))
        if inspector.is_good(t, quickanswer=True):
            return t
    raise NoReasonableStructureFound("No good structure found at function add_atom_on_surfaces")

def randomize(surface, dr, bond_range, max_trial=10):

    c_indx = [atom.index for atom in surface if atom.symbol=='C']
    o_indx = [atom.index for atom in surface if atom.symbol=='O']
    pt_indx = [atom.index for atom in surface if atom.symbol=='Pt']

    for _ in range(max_trial):
        ro = surface.get_positions()
        rn = ro.copy()
        t = surface.copy()

        for i,n in enumerate(c_indx):
            disp = np.random.uniform(-1., 1., [1, 3])
            rn[n,:] = ro[n,:]+dr*disp
            rn[o_indx[i],:] = ro[o_indx[i],:]+dr*disp

        t.set_positions(rn)
        inspector=CheckAtoms(bond_range=bond_range)
        if inspector.is_good(t, quickanswer=True):
            return t
        raise NoReasonableStructureFound("No good structure found at function add_atom_on_surfaces")    

def randomize_all(surface, dr, bond_range, max_trial=10):
    
    fixed = surface.constraints[0].index

    for _ in range(max_trial):
        ro = surface.get_positions()
        rn = ro.copy()
        t = surface.copy()
        for i,n in enumerate(range(len(surface))):
            if i in fixed:
                continue
            disp = np.random.uniform(-1., 1., [1, 3])
            rn[n,:] = ro[n,:]+dr*disp

        t.set_positions(rn)
        inspector=CheckAtoms(bond_range=bond_range)
        if inspector.is_good(t, quickanswer=True):
            return t
        raise NoReasonableStructureFound("No good structure found at function add_atom_on_surfaces")

def md_nve(surface, calc, T=500, timestep=1, steps=30):
    t = surface.copy()
    t.set_calculator(calc)
    MaxwellBoltzmannDistribution(t, temperature_K=T, force_temp=True)
    dyn = VelocityVerlet(t, timestep * units.fs)
    dyn.run(steps)
    return t

def nve_n2p2(atoms, z_fix, N):

    f = open('Current_Status.json')
    info = json.load(f)
    nstep = info['nsteps']
   
    os.system('mpirun -n 4 python nve_mod.py')
    images = read('md.lammpstrj',':')
    trajw = TrajectoryWriter('md%05i.traj'% nstep,'a')

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
        trajw.write(atoms, energy=e_pot[i], forces=f_all[i])

    t = read('md%05i.traj@-1'%nstep)

    if not os.path.isdir('md_files'):
        os.mkdir('md_files')
    
    os.system('mv md%05i.traj md_files/'% nstep)
    os.system('rm md.lammpstrj log.lammps')
    # os.system('rm md.traj')
    return t
