import numpy as np
import random
import os
import sys
from ase import Atoms
path=os.path.realpath("/home/vsumaria/software/gcbh/sourcefiles/")
sys.path.insert(0,path)
from pygcga.checkatoms import CheckAtoms

def initial_structure_gen(surface, cov , nsurf, coordinates, connectivities, normals, bond_range, max_trial=10, dr=0.05):
    
    '''
    surface = Clean Pt surface to generate a CO adsorbed stuctures
    nsurf = no. of surface Pt atoms (from graph)
    cov = coverage of CO to needed
    coordinates, connectivities, normals = obtained from catkit grapths
    bond_range = distance criteria for atoms
    '''

    t = surface.copy()
    ads_ind = []
    # for xyz in range(max_trial):
    while True:
        nC = round(cov*nsurf) # number of CO to be adsorbed
        
        for _ in range(nC):
            for _ in range(max_trial):
                s = np.random.choice(len(coordinates),1)[0]
                coordinate = coordinates[s,:]
                connectivity = connectivities[s]
                r = random.uniform(-dr,dr)
                coordinate[0]=coordinate[0]+r
                coordinate[1]=coordinate[1]+r
                normal = normals[s,:]
                #need to find a better way to do this: will change with the changing terrace facet
                if connectivity==1:
                    d_CO = 1.158
                    h = 1.848
                if connectivity==2:
                    d_CO = 1.182
                    h = 1.46
                if  connectivity==3:
                    d_CO = 1.193
                    h = 1.336
                if  connectivity==4:
                    d_CO = 1.204
                    h = 1.13
                ads_copy = Atoms('CO',positions=[(0, 0, 0), (0, 0, d_CO)],cell=[[4,0,0],[4,0,0],[0,0,4]])
                ads_copy.rotate([0, 0, 1], normal, center=[0,0,0])
                c_ads = coordinate + (normal*h)
                ads_copy.translate(c_ads)
                t_copy = t.copy()
                t.extend(ads_copy)

                inspector=CheckAtoms(bond_range=bond_range)
                if inspector.is_good(t, quickanswer=True):
                    break
                else:
                    t = t_copy.copy()
        return t
    print('Could not find any structures')
