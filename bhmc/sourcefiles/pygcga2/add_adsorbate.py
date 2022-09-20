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

def generate_H_site(r0, distance=None, width=None, center_of_mass=None):
    """
    r0 :
        is the position of Pt atom, the equilibrium distance of H and Pt is 1.36+0.70=2.06
    center_of_mass:
        it is the center of the cluster, if the center_of_mass is specified, the position of new atom will be placed
        on the outer direction from center_of_mass
    """
    if distance is None:
        distance = BOND_LENGTHS['Pt'] + BOND_LENGTHS['H']
    if width is None:
        width = distance*0.15
    dr = np.random.normal(loc=distance, scale=width)
    if center_of_mass is None:
        theta = np.arccos(1.0-2.0*np.random.uniform(low=0.0, high=1.0))
    else:
        # new H atoms only exist in the z+ direction
        theta = np.arccos((1.0-np.random.uniform(low=0.0, high=1.0)))
    phi = np.random.uniform(low=0.0, high=np.pi * 2.0)
    dz = dr * np.cos(theta)
    dx = dr * np.sin(theta) * np.cos(phi)
    dy = dr * np.sin(theta) * np.sin(phi)
    dp0 = np.array([dx, dy, dz])
    try:
        assert np.abs(np.sqrt(np.power(dp0, 2).sum()) - dr) < 1.0e-3 # check math is correct
    except AssertionError:
        print("Whoops, Bad things happened when generate new H positions")
        exit(1)
    if center_of_mass is None:
        return dp0 + np.array(r0)  # returned value is a possible site for H atom
    else:
        pout = r0 - np.array(center_of_mass)
        x, y, z = pout
        # we will find a rotation matrix to rotate the z+ direction to pout here.
        if np.power(pout, 2).sum() < 1.0e-1:
            # r0 is too close to the center of mass? this is a bad Pt site
            # if the cluster is a small cluster, this might be correct
            return None
        if (y ** 2 + z ** 2) < 1.0e-6:
            theta_y = np.pi / 2.0
            RotationMatrix = np.array([[np.cos(theta_y), 0.0, np.sin(theta_y)],
                                       [0.0, 1.0, 0.0],
                                       [-1.0 * np.sin(theta_y), 0.0, np.cos(theta_y)]])
        else:
            # fint the rotation matrix
            with np.errstate(invalid='raise'):
                # try:
                theta_x = -1.0 * np.arccos(z / np.sqrt(y ** 2 + z ** 2))
                theta_y = np.arcsin(x / np.sqrt(x ** 2 + y ** 2 + z ** 2))
                # except:
                #     print "x= %.8f, y=%.8f, z=%.8f" % (x,y,z)
                #     print z / (y ** 2 + z ** 2)
                #     print x / (x ** 2 + y ** 2 + z ** 2)
                #     print("something erorr in the math here")
                #     exit(1)
            M_x = np.array([[1.0, 0.0, 0.0],
                            [0.0, np.cos(theta_x), -1.0 * np.sin(theta_x)],
                            [0.0, np.sin(theta_x), np.cos(theta_x)]])
            M_y = np.array([[np.cos(theta_y), 0.0, np.sin(theta_y)],
                            [0.0, 1.0, 0.0],
                            [-1.0 * np.sin(theta_y), 0.0, np.cos(theta_y)]])
            RotationMatrix = np.matmul(M_y, M_x)
        # RotationMatrix could rotate the [0,0,1] to pout direction.
        dp1 = np.matmul(RotationMatrix, dp0)
        try:
            # check math is correct
            assert np.abs(np.power(dp1, 2).sum() - np.power(dr, 2)) < 1.0e-3
        except AssertionError:
            _r2 = np.power(dp1, 2).sum()
            print("_r2=%.3f" % _r2)
            print("r**2=%.3f" % dr ** 2)
            exit(1)
        return dp1 + np.array(r0)


def add_molecule_on_cluster(cluster, molecule='H', anchor_atom=None, metal="Pt",
                            verbosity=False, maxcoordination=12, weights=None,
                            max_attempts_1=500,
                            max_attempts_2=500,
                            minimum_bond_distance_ratio=0.7,
                            maximum_bond_distance_ratio=1.4,
                            outer_sphere=True,
                            distribution=0):
    """
    :param cluster: ase.atoms.Atoms object
    :param molecule: string. I should be able get the molecule from ase.build.molecule
    :param anchor_atom: string if there are more than 2 atoms in the molecule, which atom will form bond with metal,
           This parameter currently is not used.
    :param metal: string; metal element; currently on support one element cluster (no alloy allowed)
    :param maxcoordination: the maximum coordination for 'metal',the anchoring is attempted only if CN of meta
            is smaller than maxcoordition number
    :param weights: list of floats. If the weights are specified, I will not check the coordination numbers, the
           anchoring sites will be determined by the weights.
           The length of weights must be same as the number of atoms in the 'cluster' object
    :param verbosity: print details to standard out
    :param max_attempts_1: how many attempts made to find a suitable anchor site on cluster
    :param max_attempts_2: how many attempts made to anchor the molecule on the cluster
    :param minimum_bond_distance_ratio: The minimum bond distance will calculated by this ratio multiplied by the
           sum of atomic radius (got from ase.data)
    :param maximum_bond_distance_ratio: bla,bla
    :param outer_sphere: if True, the molecule will be added in the outer sphere region of the cluster, this is good
    option for large and spherical clusters (center of the mass is used). Other wise, use False, for small cluster
    :param distribution: 0 or 1: if distribution=0, the maximum coordination number of metal
    is maxcoordination; if distribution=1, the probability is uniformly distributed for different metal atoms.
    :return: ase.atoms.Atoms object
    This function will add one molecule on the cluster and return a new cluster.
    ++ using of weights ++
    If weights are not specified, the anchoring sites will be determined by the coordination number of the metal atoms,
    otherwise, the anchoring sites will be chosen by the weights. For example, the weights could be determined by atomic
    energies from high-dimensional neural network.
    ++ adding molecule rather than single atom++
    Currently this function does not support add a molecule (more than 2 atoms), so the anchor_atom is ignored.
    ++ the total attempts will be max_attempts_1 x max_attempts_2
    ++ minimum_bond_distance
    """
    if cluster is None:
        raise RuntimeError("adding molecule on the None object")
    try:
        _mole = ase_create_molecule(molecule)
        if _mole.get_number_of_atoms() == 1:
            anchor_atom = molecule.strip()
        elif anchor_atom is None:
            raise RuntimeError("You must set anchor_atom when a molecule (natoms > 1) is adsorbed")
        else:
            assert anchor_atom in _mole.get_chemical_symbols()
    except KeyError as e:
        raise RuntimeError("I can not find the molecule {}".format(e))

    cluster_topology = Topology(cluster, ratio=1.3)

    symbols = cluster.get_chemical_symbols()
    props = [] # The probabilities of attaching _mole to sites
    adsorption_sites=[] # adsorption_sites is a list of elements which can absorb molecule
    if isinstance(metal,str):
        adsorption_sites.append(metal)
    elif isinstance(metal, list):
        for m in metal:
            adsorption_sites.append(m)
    else:
        raise RuntimeError("metal=str or list, for the adsorption sites")

    # setup the probabilities for adsorbing _mole
    for index, symbol in enumerate(symbols):
        if weights is not None:
            props.append(weights[index])
        elif symbol not in adsorption_sites:
            props.append(0.0)
        else:
            if distribution == 0:
                _cn = cluster_topology.get_coordination_number(index)
                props.append(max([maxcoordination-_cn, 0.0]))
            else:
                props.append(1.0)

    atoms_checker=CheckAtoms(max_bond=maximum_bond_distance_ratio, min_bond=minimum_bond_distance_ratio)
    for n_trial in range(max_attempts_1):
        if verbosity:
            sys.stdout.write("Try to find a metal atoms, trial number %3d\n" % n_trial)
        attaching_index = selecting_index(props)
        site_symbol = symbols[attaching_index]
        if site_symbol not in adsorption_sites:
            continue
        else:
            for n_trial_2 in range(max_attempts_2):
                p0 = cluster.get_positions()[attaching_index]
                if outer_sphere:
                    p1 = generate_H_site(p0, distance=BOND_LENGTHS[site_symbol]+BOND_LENGTHS[anchor_atom],
                                         width=0.05, center_of_mass=cluster.get_center_of_mass())
                else:
                    p1 = generate_H_site(p0, distance=BOND_LENGTHS[site_symbol]+BOND_LENGTHS[anchor_atom],
                                         width=0.05, center_of_mass=None)
                if p1 is None:
                    # I did not find a good H position
                    continue
                if _mole.get_number_of_atoms() == 1:
                    _mole.set_positions(_mole.get_positions() - _mole.get_center_of_mass() + p1)
                    new_atoms = cluster.copy()
                    for _a in _mole:
                        _a.tag = 1
                        new_atoms.append(_a)
                    if atoms_checker.is_good(new_atoms, quickanswer=True):
                        return new_atoms
                else:
                    # if _mole has more than 1 atoms. I will try to rotate the _mole several times to give it a chance
                    # to fit other atoms
                    for n_rotation in range(20):
                        new_atoms = cluster.copy()
                        # firstly, randomly rotate the _mole
                        phi = np.rad2deg(np.random.uniform(low=0.0, high=np.pi * 2.0))
                        theta = np.rad2deg(np.arccos(1.0 - 2.0 * np.random.uniform(low=0.0, high=1.0)))
                        _mole.euler_rotate(phi=phi, theta=theta, psi=0.0, center="COP")
                        # get the positions of anchored atoms
                        possible_anchored_sites = [i for i, e in enumerate(_mole.get_chemical_symbols())
                                                   if e == anchor_atom]
                        decided_anchored_site = np.random.choice(possible_anchored_sites)
                        p_anchored_position =_mole.get_positions()[decided_anchored_site]
                        # make sure the anchored atom will locate at p1, which is determined by the function
                        _mole.set_positions(_mole.get_positions()+(p1-p_anchored_position))
                        for _a in _mole:
                            _a.tag = 1
                            new_atoms.append(_a)
                        if atoms_checker.is_good(new_atoms, quickanswer=True):
                            return new_atoms
                        else:
                            continue
    raise NoReasonableStructureFound("No reasonable structure found when adding an addsorbate")

def remove_one_adsorbate(structures=None, molecule="H", adsorbate_weights=None):
    """
    :param structures: atoms object
    :param molecule: a symbol for element
    :param adsorbate_weights:
    :return:
    """

    atoms = structures.copy()
    nat_cut = natural_cutoffs(atoms, mult=0.9)
    nl = NeighborList(nat_cut, self_interaction=False, bothways=True)
    nl.update(atoms)

    ndx_mol = [atom.index for atom in atoms if atom.symbol==molecule]
    C_ndx = [atom.index for atom in atoms if atom.symbol=='C']

    flag=[]
    for i in ndx_mol:
        indices, offsets = nl.get_neighbors(i)
        x = np.intersect1d(indices, C_ndx)
        if len(x)!=0:
            flag.append(0)
        else:
            flag.append(1)

    flag = np.array(flag)

    if np.all(flag == 0):
        return structures

    ndx_mol_sel = [ndx_mol[i] for i in range(len(flag)) if flag[i]==1]
    print(ndx_mol_sel)

    sel_ndx = random.choice(ndx_mol_sel)

    print('index to delete is '+str(sel_ndx))
    del structures[sel_ndx]

    # delete_single_atom_with_constraint(structures, sel_ndx)
    return structures

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
