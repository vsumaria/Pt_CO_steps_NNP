from ase.db import connect
from ase.io import *
import glob

def n2p2_data(atoms, energy, forces, name):
	out_str = ''
	out_str += 'begin\n'
	out_str += 'comment ' + name + '\n'
	
	cell = atoms.get_cell()
	cell_template = 'lattice {:10.6f} {:10.6f} {:10.6f}\n'
	for c in cell:
		out_str += cell_template.format(c[0], c[1], c[2])

	atom_template = 'atom {:10.6f} {:10.6f} {:10.6f} {:2s} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f}\n'
	forces = atoms.get_forces()
	for a in atoms:
		force = forces[a.index]
		out_str += atom_template.format(a.position[0], a.position[1], a.position[2],
										a.symbol, 0.0, 0.0,
										force[0], force[1], force[2])
	out_str += 'energy {:10.6f}\n'.format(energy)
	out_str += 'charge 0.0\n'
	out_str += 'end\n'
	return out_str

name = glob.glob('*.traj')
db = read(name[0],':')

with open('input.data', 'w') as input_data:
	for atoms in db:
                name = 'test'
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                input_data.write(n2p2_data(atoms, energy, forces, name))
