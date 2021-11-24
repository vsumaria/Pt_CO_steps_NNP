import sys
import numpy as np
from ase.calculators.calculator import all_changes
from ase.calculators.calculator import Calculator
import pynnp
import os

class N2P2Calculator(Calculator):
    
    def __init__(self, **kwargs):
        Calculator.__init__(self, **kwargs)
        self.implemented_properties = {
            'energy' : self.calculate_energy_and_forces,
            'forces' : self.calculate_energy_and_forces}
        self.results = {}

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        # Calculator essentially does: self.atoms = atoms
        Calculator.calculate(self, atoms, properties, system_changes)

        has_results = [p in self.results for p in properties]
        if (len(system_changes) > 0) or (not np.all(has_results)):
            if ('energy' in properties) and ('forces' in properties):
                # Forces evaluation requires energy. No need to compute
                # energy twice.
                del properties[properties.index('energy')]
            for p in properties:
                if p in self.implemented_properties:
                    self.implemented_properties[p](self.atoms)
                else:
                    raise NotImplementedError(
                        "Property not implemented: {}".format(p))

    def calculate_energy_and_forces(self, atoms):

        def n2p2_data():
            out_str = ''
            out_str += 'begin\n'
            out_str += 'comment test \n'

            n = len(atoms)
            cell = atoms.get_cell()
            cell_template = 'lattice {:10.6f} {:10.6f} {:10.6f}\n'
            for c in cell:
                out_str += cell_template.format(c[0], c[1], c[2])

                atom_template = 'atom {:10.6f} {:10.6f} {:10.6f} {:2s} {:10.6f} {:10.6f} {:10.6f} {:10.6f} {:10.6f}\n'
                forces = np.zeros([n,3])
            for a in atoms:
                force = forces[a.index]
                out_str += atom_template.format(a.position[0], a.position[1], a.position[2], a.symbol, 0.0, 0.0, force[0], force[1], force[2])

            energy = 0
            out_str += 'energy {:10.6f}\n'.format(energy)
            out_str += 'charge 0.0\n'
            out_str += 'end\n'

            return out_str

        with open('input.data', 'w') as input_data:
            input_data.write(n2p2_data())

        os.system('cp ~/nnp/* .')
        p = pynnp.Prediction()
        p.setup()
        p.readStructureFromFile()
        p.predict()
        s = p.structure

        e_nnp = s.energy
        natoms = s.numAtoms

        f = np.zeros([natoms,3])
        for i,atom in enumerate(s.atoms):
            f[i,:] = atom.f.r

        self.results['energy'] = e_nnp
        self.results['forces'] = f

        os.system('rm weights* input.nn scaling.data input.data')
