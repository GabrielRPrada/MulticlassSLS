import numpy as np

EPSILON = np.nextafter(0, 1)

class LineSet(object):
    def __init__(self, lines):
        self.lines = np.array(lines).flatten()
        for i in range(len(lines)-1):
            assert lines[i].size == lines[i+1].size # garante que todos os segmentos de reta tem o mesmo tamanho
        self.dim = lines[0].size
        self.n_lines = len(lines)
        self.shape = (self.dim, self.n_lines)

    def __add__(self, other):
        assert len(self.lines) == len(other.lines)
        return LineSet([self.lines[i] + other.lines[i] for i in range(len(self.lines))])

    def __neg__(self):
        return LineSet([-line for line in self.lines])

    def __sub__(self, other):
        return self + (-other)
    
    def __str__(self):
        return f"[{', '.join([str(l) for l in self.lines])}]"

    def __iter__(self):
        return iter(self.lines)
    
    def hyper(self):
        return LineSet([l.hyper() for l in self.lines])

    def discriminant(self, x):
        acc = 0
        for line in self.lines:
            acc += 1/(line.pdist(x)+EPSILON)
        
        return acc
    
    def discriminant_gradient(self, x):
        acc = np.zeros(self.dim*self.n_lines*2)
        i = 0
        for line in self.lines:
            grad = line.pdist_gradient(x)
            acc[i:len(grad)] += grad / (line.pdist(x)**2)
            i += 1
        
        return -acc
