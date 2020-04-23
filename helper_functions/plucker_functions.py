# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:48:33 2020

@author: Pushyami Kaveti
"""
import numpy as np


class Plucker:
    
    def from_points(self, p1, p2):
        '''
        Constructor for creating plucker vector from two 
        3D points in space
    
        Parameters
        ----------
        p1 : 
        p2 : TYPE
            DESCRIPTION.

        '''
        self.l = p2 - p1
        self.m = np.cross(p1, self.l )
        
    def from_point_dir(self, p1, d):
        '''
        Constructor for creating plucker vector from a 
        3D point in space and the direction of the ray
        typ should be 'point-dir'

        Parameters
        ----------
        p1 : TYPE
            DESCRIPTION.
        d : TYPE
            DESCRIPTION.

        '''
        self.l = d
        self.m = np.cross(p1,d)
        
    def from_dir_moment(self, d, m):
        '''
        from a direction vector and moment vector 
        typ should be 'dir-moment'

        Parameters
        ----------
        d : TYPE
            DESCRIPTION.
        m : TYPE
            DESCRIPTION.
        '''
        self.l = d
        self.m = m
        
    def __init__(self, p1, p2, typ='default'):
        if typ == 'default':
            self.from_points(p1, p2)            
        if typ=='point-dir':
            self.from_point_dir(p1,p2)
        elif typ == 'dir-moment':
            self.from_dir_moment(p1,p2)


    def distance_to_origin(self):
        '''
        Method to compute the Distance of the plucker 
        line from the origin. d=|w|/|q|

        Returns
        -------
        d : TYPE
            DESCRIPTION.

        '''
        d = np.norm(self.m)/ np.linalg.norm(self.l)
        return d
    
    def closest_point_to_origin(self):
        '''
         Method to compute the Distance of the plucker 
         line from the origin. p = l x m / |l|^2

        Returns
        -------
        p : TYPE
            DESCRIPTION.

        '''
        p = np.cross(self.l , self.m)/ np.dot(self.l, self.l)
        return p
    
    @staticmethod
    def reciprocal_product(pl1, pl2):
        '''
        Method to compute the the reciprocal product
        of two plucker vectors pl1 and pl2
        (l1,m1) * (l2 , m2) = l1^ . m2 + l2^ . m1
  
        Parameters
        ----------
        pl1 : TYPE
            DESCRIPTION.
        pl2 : TYPE
            DESCRIPTION.

        Returns
        -------
        rec_prod : TYPE
            DESCRIPTION.

        '''
        u1 = pl1.l / np.linalg.norm(pl1.l)
        u2 = pl2.l / np.linalg.norm(pl2.l)
        rec_prod = np.dot(u1,pl2.m) + np.dot(u2, pl1.m)
        return rec_prod

    def gen_point(self, lamda):
        '''
        Method to generate a point on the plucker line
        p = (l x m / |l|^2) + lamda *  l^
        '''
        u1 = self.l / np.linalg.norm(self.l)
        p = self.closest_point_to_origin() + lamda * u1
        return p

    @staticmethod
    def is_parallel(pl1, pl2):
        '''
        Method to check if two plucker lines are parallel 
        l1 x l1 = 0
       '''
        v = np.linalg.norm( np.cross(pl1.l, pl2.l) ) < 10*np.finfo(float).eps 
        return v

    @staticmethod
    def point_of_intersection(pl1, pl2):
        '''
        Method to calculate the point of intersection 
        between two plucker lines. If the lines are not
        parallel and not co-planar then the point of 
        intersection is given by
        p = ((m1.l2) I + l1 m2.T - l2 m1.T ) * (l1 x l2)/ |l1 x l2|^2

        Parameters
        ----------
        pl1 : TYPE
            DESCRIPTION.
        pl2 : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        if Plucker.is_parallel(pl1, pl2) and Plucker.reciprocal_product(pl1, pl2) > 10*np.finfo(float).eps :
            return 0
        else:
            tmp = np.dot(pl1.m, pl2.l) * np.eye(3) + np.outer(pl1.l, pl2.m) - np.outer(pl2.l , pl1.m)
            c = np.cross(pl1.l,pl2.l)
            p = tmp @ c / np.dot(c,c)
            return p

    @staticmethod
    def point_of_intersection_plane(pl, plane):
        '''
        Method to find the point of intersection of 
        the plucker vector with a plane
        '''
        pass
           
            