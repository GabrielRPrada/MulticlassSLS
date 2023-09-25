import numpy as np
from .segment import LineSegment
from .set import LineSet 

class MultiClassSLS(object):
    def __init__(self, n_lines=2):
        self.n_lines = n_lines
    
    def __str__(self):
        text = ""
        for i in range(len(self.classes)):
            text += f"Class {self.classes[i]}: {self.lines[i]}\n"
        return text

    def _set_lines(self, lines, classes, hyper=True):
        assert len(classes) == len(lines)
        classes = np.array(classes, dtype=int)
        self.classes = classes
        self.lines = {}
        for i in np.unique(classes):
            if hyper:
                self.lines[i] = [x.hyper() for (j,x) in enumerate(lines) if classes[j] == i][0]
            else:
                self.lines[i] = [x for (j,x) in enumerate(lines) if classes[j] == i][0]
        return self

    def _cross_entropy(self, x, y):
        acc = 0
        for i, point in enumerate(x):
            s = np.log(np.sum([np.exp(lines.discriminant(point)) for lines in self.lines.values()]))
            disc = self.lines[y[i]].discriminant(point)
            acc -= disc + s
        
        return acc

    def _cross_entropy_gradient(self, x, y):
        acc = []
        
        for lines in self.lines.values():
            acc.append(np.zeros((2*lines.dim*lines.n_lines)))

        for i, point in enumerate(x):
            s = np.log(np.sum([np.exp(lines.discriminant_gradient(point)) for lines in self.lines.values()]))
            disc = self.lines[y[i]].discriminant_gradient(point)
            acc[y[i]] -= disc + s
        
        return acc

    def _adjust_lines(self, delta):
        copy = MultiClassSLS(n_lines = self.n_lines)

        lines = self.lines.copy()
        for i in range(len(delta)):
            steps = []
            d = self.lines[i].dim
            for j in range(int(len(delta[i])/(d*2))):
                p = delta[i][j*d*2:j*d*2+d*2:2]
                q = delta[i][j*d*2+1:j*d*2+d*2:2]
                steps.append(LineSegment(p, q))

            lines[i] += LineSet(steps)
        
        lines_flatten = [l for l in lines.values()]
        copy._set_lines(lines_flatten, self.classes, False)
        return copy

    def train(self, x, y, max_iter = 5, disp_min = 0.05):
        hyper_x = np.c_[x, np.zeros(x.shape[0])]
        learning_rate = 0.5
        iter = max_iter
        if max_iter == 0:
            iter = float('inf')
        while iter > 0:
            iter -= 1
            grad = self._cross_entropy_gradient(hyper_x, y)
            grad_flatten = np.concatenate(grad, axis=0)
            m = np.linalg.norm(grad_flatten)**2
            print(m)
            if m < disp_min:
                break
            t = learning_rate*m
            alpha = -0.5
            step = [alpha*g for g in grad]
            objective = self._cross_entropy(hyper_x, y)
            print(objective)
            copy = self._adjust_lines(step)
            while objective - copy._cross_entropy(hyper_x, y) < alpha*t:
                alpha /= 2
                step = [alpha*g for g in grad]
                copy = self._adjust_lines(step)
            self = self._adjust_lines(step)

        return self