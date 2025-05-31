import math

class SimpleRandom:
    def __init__(self, seed=1):
        self.modulus = 2**32
        self.a = 1664525
        self.c = 1013904223
        self.state = seed

    def rand(self):
        self.state = (self.a * self.state + self.c) % self.modulus
        return self.state

    def rand_float(self):
        return self.rand() / self.modulus

    def uniform(self, low, high):
        return low + (high - low) * self.rand_float()

    def normal(self, mean=0, std=1):
        # Box-Muller transform for normal distribution
        if not hasattr(self, '_has_spare'):
            self._has_spare = False
        
        if self._has_spare:
            self._has_spare = False
            return self._spare * std + mean
        
        self._has_spare = True
        u = self.rand_float()
        v = self.rand_float()
        mag = std * math.sqrt(-2.0 * math.log(u))
        self._spare = mag * math.cos(2.0 * math.pi * v)
        return mag * math.sin(2.0 * math.pi * v) + mean
