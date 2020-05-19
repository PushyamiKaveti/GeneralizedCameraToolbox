# -*- coding: utf-8 -*-
"""
Created on Sat May 12 10:20:03 2020

@author: Pushyami Kaveti
"""
class Ransac:
    def __init__(self, N_trials=1e8, thresh=0.0001, P=0.99, sam_size = 17, max_iter = 1000000):
        self.Num_trials = N_trials
        self.threshold = thresh
        self.prob = P
        self.iteration = 0
        self.sample_size = sam_size
        self.max_iterations = max_iter
    
    def run(self):
        pass
    
    
    