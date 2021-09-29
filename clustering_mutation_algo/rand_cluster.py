#!/usr/bin/env python
import matplotlib.path as mplPath
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
from shapely.geometry import Polygon, Point
from collections import defaultdict
import random
from shapely.geometry.linestring import LineString
import numpy as np
import sys, os

def voronoi_polygons(voronoi, diameter):
    """
    Generate voronoi cells using the centriods.
    """
    centroid = voronoi.points.mean(axis=0)
    ridge_direction = defaultdict(list)
    for (p, q), rv in zip(voronoi.ridge_points, voronoi.ridge_vertices):
        u, v = sorted(rv)
        if u == -1:
            t = voronoi.points[q] - voronoi.points[p] # tangent
            n = np.array([-t[1], t[0]]) / np.linalg.norm(t) # normal
            midpoint = voronoi.points[[p, q]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - centroid, n)) * n
            ridge_direction[p, v].append(direction)
            ridge_direction[q, v].append(direction)

    for i, r in enumerate(voronoi.point_region):
        region = voronoi.regions[r]
        if -1 not in region:
            # Finite region.
            yield Polygon(voronoi.vertices[region])
            continue
        # Infinite region.
        inf = region.index(-1)              # Index of vertex at infinity.
        j = region[(inf - 1) % len(region)] # Index of previous vertex.
        k = region[(inf + 1) % len(region)] # Index of next vertex.
        if j == k:
            # Region has one Voronoi vertex with two ridges.
            dir_j, dir_k = ridge_direction[i, j]
        else:
            # Region has two Voronoi vertices, each with one ridge.
            dir_j, = ridge_direction[i, j]
            dir_k, = ridge_direction[i, k]
        length = 2 * diameter / np.linalg.norm(dir_j + dir_k)
        # Polygon consists of finite part plus an extra edge.
        finite_part = voronoi.vertices[region[inf + 1:] + region[:inf]]
        extra_edge = [voronoi.vertices[j] + dir_j * length,
                      voronoi.vertices[k] + dir_k * length]
        yield Polygon(np.concatenate((finite_part, extra_edge)))

def random_point_within(poly,n=40):
    """
    Randomly generate 'n' centroid points within the cell boundary
    """
    min_x, min_y, max_x, max_y = poly.bounds

    X = []
    Y=[]
    for i in range(n):
        x = random.uniform(min_x, max_x)
        X.append(x)
        x_line = LineString([(x, min_y), (x, max_y)])
        x_line_intercept_min, x_line_intercept_max = x_line.intersection(poly).xy[1].tolist()
        y = random.uniform(x_line_intercept_min, x_line_intercept_max)
        Y.append(y)
    return ([X, Y])

def rand_clustering(surface, dr,n=4,boundary=None):
    """
    surface = input strucutre
    dr = max distance used while randomly displacing adsorbates
    n = number of centroid points
    boundary = array of points definding the unit cell unless stated otherwise.
    """
    
    cell = surface.get_cell()
    a = cell[0,0]
    b = cell[1,0]
    c = cell[1,1]
    tol = 1.5
    
    org_boundary = np.array([[0,0],[a,0],[a+b,c],[b,c]])
    
    if boundary is None:
        boundary = np.array([[-tol,-tol],[a+tol,-tol],[a+b+tol,c+tol],[b-tol,c+tol]])

    boundary_polygon = Polygon(boundary)
    points = random_point_within(boundary_polygon,n=n)
    
    temp_var = random_point_within(boundary_polygon,n=n)
    points=[]
    for elem in zip(temp_var[0],temp_var[1]):
        points.extend([elem])

    points = [list(ele) for ele in points] 
    points = np.array(points)

    org_boundary_polygon = Polygon(org_boundary)
    pos = surface.get_positions()
    C_ndx = [atom.index for atom in surface if atom.symbol == 'C']
    O_ndx = [atom.index for atom in surface if atom.symbol == 'O']
    posC = np.zeros((len(C_ndx),2))
    posO = np.zeros((len(C_ndx),2))
    i=0
    for c,o in zip(C_ndx,O_ndx):
        pc = pos[c,0:2]
        po = pos[o,0:2]
        posC[i] = pc
        posO[i] = po
        i=i+1

    # points = [[points[0][0],points[1][0]],[points[0][1],points[1][1]],[points[0][2],points[1][2]],[points[0][3],points[1][3]]]
    diameter = np.linalg.norm(boundary.ptp(axis=0))
    boundary_polygon = Polygon(boundary)
    X = []
    Y = []
    for p in voronoi_polygons(Voronoi(points), diameter):
        try:
            x, y = zip(*p.intersection(boundary_polygon).exterior.coords)
        except (ValueError):
            raise NoReasonableStructureFound("No good structure found at function add_atom_on_surfaces")
        X.append(np.array(x))
        Y.append(np.array(y))
    
    ro = surface.get_positions()
    rn = ro.copy()
    t = surface.copy()
    count=0

    for x,y in zip(X,Y):
        disp = np.random.uniform(-1., 1., [1, 3]) #difine displacement from particular tesslation region/ cluster
        #find and move atoms in the cluster
        comb = np.vstack((x,y))
        comb = comb.T
        bbPath = mplPath.Path(comb)
    
        for i,n in enumerate(C_ndx):
            pc = pos[n,0:2]      
            if bbPath.contains_point((pc[0],pc[1])):
                rn[n,:] = ro[n,:]+dr*disp
                rn[O_ndx[i],:] = ro[O_ndx[i],:]+dr*disp
        count=count+1

    t.set_positions(rn)

    
    return t
    
