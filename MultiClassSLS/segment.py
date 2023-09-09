import numpy as np

def dist(x,y):
    return np.linalg.norm(x-y,2)

class LineSegment(object):
    def __init__(self, p, q):
        p = np.array(p)
        q = np.array(q)
        assert p.size == q.size
        self.size = p.size
        self.p = p
        self.q = q
        self.measure = dist(p,q)

    def __add__(self, other):
        return LineSegment(self.p + other.p, self.q + other.q)

    def __neg__(self):
        return LineSegment(-self.p, -self.q)

    def __sub__(self, other):
        return self + (-other)
    
    def pdist(self, x):
        x = np.array(x)
        assert x.size == self.size
        return (dist(x, self.p) + dist(x, self.q) - self.measure) / 2

    def pdist_gradient(self, x):
        x = np.array(x)
        assert x.size == self.size
        gradient = np.empty(self.size)
        dist_p = dist(x, self.p)
        dist_q = dist(x,self.q)
        for i in range(self.size):
            gradient[2*i] = (self.measure*(x[i]-self.p[i]) + dist_p*(self.q[i] - self.p[i]))/(2*dist_p*self.measure)
            gradient[2*i + 1] = (self.measure*(x[i]-self.q[i]) + dist_q*(self.p[i] - self.q[i]))/(2*dist_q*self.measure)