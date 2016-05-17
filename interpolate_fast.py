import numpy as np
import interpolate_slow

def construct_vertices(xy , uv):
    """
    Preparation method for fast interpolation. This method calculates vertices and weights once, such that
    the interpolation can be done much quicker than e.g. scipy.interpolate.griddata
    :param xy: X,Y coordinates of FEM mesh
    :param uv: X,Y coordinates of the BEM mesh
    :return: Vertices, Weights
    """
    import scipy.spatial.qhull as qhull
    tri = qhull.Delaunay(xy)
    simplex = tri.find_simplex(uv)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uv - temp[:, 2]
    bary = np.einsum('njk,nk->nj', temp[:, :2, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

def fast_interpolate(values, vertices, weights):
    """
    Fast interpolation method (linear), to be executed after construct_vertices
    :param values: Values of the FEM solution at the FEM mesh points
    :param vertices: from construct_vertices
    :param weights: from construct_vertices
    :return: Interpolated values at the BEM mesh points.
    """
    return np.einsum('nj,nj->n', np.take(values, vertices), weights)



