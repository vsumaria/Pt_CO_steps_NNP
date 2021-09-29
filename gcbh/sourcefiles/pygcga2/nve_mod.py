#!/usr/bin/env python
from lammps import PyLammps
from ase.io import *
from ase.io.trajectory import TrajectoryWriter
import json
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.lammps.data import LammpsData

f = open('Current_Status.json')
info = json.load(f)
nstep = info['nsteps']
T = info['t_nve']

atoms = read('local_minima.db@-1')
z_fix = 18.5
N = 300

n = len(atoms)
ase_adap = AseAtomsAdaptor()
atoms_ = ase_adap.get_structure(atoms)
ld = LammpsData.from_structure(atoms_,atom_style='atomic')
ld.write_file("struc.data")

L = PyLammps()
L.command("units  metal")
L.command("dimension  3")
L.command("atom_modify map yes")
L.command("box tilt large")
L.command("boundary  p p f")
L.command("atom_style  atomic")
L.command("read_data  struc.data")
L.command("pair_style nnp showew no showewsum 10000 maxew 500000 resetew yes cflength 1 cfenergy 1 emap 1:Pt,2:C,3:O")
L.command("pair_coeff * * 6")

# fix bottom atoms
L.command("region slab block EDGE EDGE EDGE EDGE 0 "+ str(z_fix))
L.command("group fixed_slab region slab")
L.command("fix freeze fixed_slab setforce 0.0 0.0 0.0")

L.command("region slab2 block EDGE EDGE EDGE EDGE "+ str(z_fix)+" 100")
L.command("group moving_slab region slab2")
L.command("info region slab2 atom")

L.command("dump 1 all custom 1 md.lammpstrj id type x y z fx fy fz")
L.command("thermo 1")
L.command("thermo_style custom step pe")

#minimize
L.command("velocity moving_slab create "+ str(T) + " 4928459 rot yes dist gaussian")
L.command("fix 1 moving_slab nve")
L.command("timestep 0.001")
L.command("run "+str(N))
