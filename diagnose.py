from numpy import *
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

        print tabulate(zip(array(Z)[:,0], array(array(Z)[:,1], dtype=float)),
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
        #return dot(self.lhs.weakform(), solution) - self.rhs

    def monitor_conjugate_gradient_residual(self, solution):
        residual = self.calculate_residual(solution)
        self.append_residual(residual)