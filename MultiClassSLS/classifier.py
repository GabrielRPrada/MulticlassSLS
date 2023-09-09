import numpy as np
from segment import LineSegment
from set import LineSet 

class MultiClassSLS(object):
    def __init__(self, n_lines=2):
        self.n_lines = n_lines
    
    def _set_lines(self, lines, classes):
        assert len(classes) == len(lines)
        classes = np.array(classes, dtype=int)
        self.classes = classes
        self.lines = {}
        for i in np.unique(classes):
            self.lines[i] = [x for (j,x) in enumerate(lines) if classes[j] == i][0]
    
        return self

    def _cross_entropy(self, x, y):
        acc = 0
        for i, point in enumerate(x):
            s = np.log(np.sum([np.exp(lines.discriminant(point)) for lines in self.lines.keys()]))
            disc = self.lines[y[i]].discriminant(point)
            acc -= disc + s
        
        return acc

    def _cross_entropy_gradient(self, x, y):
        acc = []
        
        for lines in self.lines.keys():
            acc.append(np.zeros((2*lines.dim*lines.n_lines)))

        for i, point in enumerate(x):
            s = np.log(np.sum([np.exp(lines.discriminant_gradient(point)) for lines in self.lines.keys()]))
            disc = self.lines[y[i]].discriminant_gradient(point)
            acc[y[i]] -= disc + s
        
        return acc

    def _adjust_lines(self, delta):
        copy = MultiClassSLS(n_lines = self.n_lines)

        lines = self.lines
        for i in range(len(delta)):
            lines[i] += delta[i]
        
        copy._set_lines(lines, self.classes)
        return copy

    def train(self, x, y, max_iter = 5, disp_min = 0.05):

        learning_rate = -0.5
        for i in range(max_iter):
            grad = self._cross_entropy_gradient(x, y)
            grad_flatten = np.concatenate(grad, axis=0)
            m = -np.linalg.norm(grad_flatten)**2
            if m < disp_min:
                break
            t = - learning_rate*m
            alpha = -0.5
            step = [alpha*g for g in grad]
            objective = self._cross_entropy(x, y)
            while objective - self._adjust_lines(step) < alpha*t:
                alpha /= 2
                step = [alpha*g for g in grad]
            self = self._adjust_lines(step)

        return self