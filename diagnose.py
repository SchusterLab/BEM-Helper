import numpy as np
from tabulate import tabulate

class Timer:
    def __init__(self):
        print "Timer class initiated"
        self.__t_i = None
        self.__t_f = None
        self.__elapsed = None

    def tic(self):
        self.__t_i = time.time()

    def toc(self):
        self.__t_f = time.time()

        if self.__t_i is None:
            raise Exception("toc function called without tic")

        self.__elapsed = self.__t_f - self.__t_i

        print "Total time elapsed is {:.2e} s".format(self.__elapsed)

    def save(self, save_string):
        setattr(self, save_string, self.__elapsed)

    def report(self):
        X = self.__dict__.keys()
        X1 = list()
        for x in X:
            if not(x.startswith('_')):
                X1.append(x)

        Y = [self.__dict__[x] for x in X1]
        # Sort the entries in the report according to the times.
        Z = sorted(zip(X1,Y), key=lambda a_entry: a_entry[1])

        print tabulate(zip(np.array(Z)[:,0], np.array(np.array(Z)[:,1], dtype=float)),
                       headers=["Operation", 'Time (s)'],
                       tablefmt="rst", floatfmt=".3f", numalign="center", stralign='left')

class Residuals:
    def __init__(self, verbose):
        self.residuals = array([])
        self.verbose = verbose
        self.report_every = 50
        self.look_back = 100
        self.convergence_criterion = 0.1

    def append_residual(self, residual):
        if self.verbose:
            if len(self.residuals)%self.report_every==1:
                print "#{:d}\t{:.2e}".format(len(self.residuals), residual)
        self.residuals = append(self.residuals, array([residual]))

    def monitor_convergence(self, residual):
        self.append_residual(residual)
        if len(self.residuals) > self.look_back:
            pct_change = abs(self.residuals[-int(self.look_back)] - self.residuals[-1])/(self.residuals[-int(self.look_back)]) * 100.
        else:
            pct_change = 100.

        if pct_change < self.convergence_criterion:
            print "Only {:.2f}% change in residuals over last {1} samples. Final tolerance {2} after {3} samples."\
                .format(pct_change, int(self.look_back), residual, len(self.residuals))
            return 0

    def calculate_residual(self, solution):
        return solution
  
    def monitor_conjugate_gradient_residual(self, solution):
        residual = self.calculate_residual(solution)
        self.append_residual(residual)

class ElementCoordinates:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def get_element_coordinates(self, elements, domain_indices):
        for k in np.unique(domain_indices):
            # Select elements with domain index k
            indices = np.where(domain_indices == k)[0]
            xs, ys, zs = np.array([]), np.array([]), np.array([])
            for l in indices:
                xs = np.append(xs, elements[l].geometry.corners[0,:])
                ys = np.append(ys, elements[l].geometry.corners[1,:])
                zs = np.append(zs, elements[l].geometry.corners[2,:])

            setattr(self, "PhysicalGroup%d"%k, {"x" : xs, "y" : ys, "z" : zs})

    def get_two_variables(self, x, y, z):
        constant_coordinate = [len(np.unique(np.diff(k)))==1 for k in [x, y, z]]
        if self.verbose:
            print "{:s} is constant".format(np.array(['x', 'y', 'z'])[np.array(constant_coordinate)][0])
        coordinates = np.arange(0,3)[np.logical_not(constant_coordinate)]
        non_constant = np.array([x,y,z])[coordinates]
        return non_constant[0], non_constant[1]

    def get_unique_points(self, x, y, z):
        x1, x2 = self.get_two_variables(x, y, z)
        unique_points, unique_indices = np.unique(x1 + 1j*x2, return_index=True)
        return x1[unique_indices], x2[unique_indices]
