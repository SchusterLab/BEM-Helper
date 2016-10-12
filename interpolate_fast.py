import numpy as np
from . import interpolate_slow

def construct_vertices_from_coordinates(xy , uv):
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

def interpolate_from_vertices(values, vertices, weights):
    """
    Fast interpolation method (linear), to be executed after construct_vertices
    :param values: Values of the FEM solution at the FEM mesh points
    :param vertices: from construct_vertices
    :param weights: from construct_vertices
    :return: Interpolated values at the BEM mesh points.
    """
    return np.einsum('nj,nj->n', np.take(values, vertices), weights)

def construct_vertices_from_grid(c, filename, physical_group):
    """
    Set up for fast interpolation of a single face (physical group). Usually followed by
    :param c: diagnose.ElementCoordinates instance
    :param filename: Filename of a single file containing FEM BC data
    :param physical_group: diagnose.ElementCoordinate.PhysicalGroupx instance. This is instantiated after running
    running diagnose.ElementCoordinate.get_element_coordinates(...,...)
    :return: Vertices, Weights, XY points of the BEM mesh, Solution on the FEM mesh
    """
    x1_unique, x2_unique = c.get_unique_points(physical_group['x'], physical_group['y'], physical_group['z'])
    #print x1_unique.shape, x2_unique.shape

    XY_eval = np.hstack((x1_unique.reshape(x1_unique.shape[0],1), x2_unique.reshape(x2_unique.shape[0],1)))
    # Then we need the FEM mesh points, make sure we get the right corresponding surface.
    elements, nodes, elem_solution = interpolate_slow.get_maxwell_boundary_data(filename)
    xdata, ydata, Udata = interpolate_slow.prepare_for_interpolation(elements, nodes, elem_solution)

    XY_data = np.hstack((xdata.reshape(xdata.shape[0],1), ydata.reshape(ydata.shape[0],1)))
    # Evaluate the vertices and weights at the BEM mesh points specified in XY_eval
    vertices, weights = construct_vertices_from_coordinates(XY_data, XY_eval)

    return vertices, weights, XY_eval, Udata






