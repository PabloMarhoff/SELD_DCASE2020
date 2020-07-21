#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from acoular.tprocess import digest
from acoular.signals import Int
from acoular import Grid
from numpy import pi, array, cos, sin, arange, arccos, repeat, concatenate
from traits.api import property_depends_on,Property,cached_property

class SphericalGrid_Equiangular( Grid ):
    """
    Provides a spherical grid for beamforming.
    Ratio of the Gridpoints should be Azimuth:Elevation = 2:1
    """
       
    phi = Property(depends_on=['_npoints_azi'])

    theta = Property(depends_on=['_npoints_ele'])
    
    
    def __init__(self,np_azi,np_ele):
        self.set_npoints_azi(np_azi)
        self.set_npoints_ele(np_ele)

    def set_npoints_azi(self,np_azi):
        self._npoints_azi = np_azi
        
    def set_npoints_ele(self,np_ele):
        self._npoints_ele = np_ele
    
    # def _get_npoints_azi(self):
    #     return self._npoints_azi
    
    # def _get_npoints_ele(self):
    #     return self._npoints_ele
        
#    npoints = Int(_npoints_azi*_npoints_ele)
    #: how many grid points

    digest = Property(depends_on = ['_steer_obj.digest','phi','theta',
                                    '_npoints_azi','_npoints_ele'] )

    @cached_property
    def _get_digest( self ):
        return digest(self)
    
    
    @property_depends_on('_npoints_azi','_npoints_ele')
    def _get_size( self ):
        return (self._npoints_azi * self._npoints_ele)

    @property_depends_on('_npoints_azi','_npoints_ele')
    def _get_shape( self ):
        return (self._npoints_azi, self._npoints_ele)


#    @cached_property
    def _get_phi( self ):
        '''
        Gibt NPOINTS_AZI(=Anzahl), gleichmäßig verteilte Azimuth-Gitterpunkte
        zurück.
        Intervall: [-pi, +pi]
        '''
        indices = arange(0, self._npoints_azi, dtype=float) + 0.5
        phi = repeat((2*pi * (indices / self._npoints_azi)) - pi, self._npoints_ele)
        return phi

#    @cached_property
    def _get_theta( self ):
        indices = arange(0, self._npoints_ele, dtype=float)
        theta = []
        theta_tmp = pi * ((indices+0.5)/self._npoints_ele)
        for i in range (self._npoints_azi):
            theta = concatenate((theta, theta_tmp), axis=None)
        return theta

    def _get_gpos( self ):
        """
        Calculates grid co-ordinates.
        Returns
        -------
        array of floats of shape (3, :attr:`~Grid.size`)
            The grid points 
            oordinates in one array.
        """
        
        bpos = array([sin(self.theta) * cos(self.phi),
                      sin(self.theta) * sin(self.phi),
                      cos(self.theta)])
        return bpos



    
class SphericalGrid_EvenlyDist( Grid ):
    """
    Provides a spherical grid for beamforming.
    """
   
    #: how many grid points
    npoints = Int(100)
    
    theta = Property(depends_on=['npoints'])
    
    phi = Property(depends_on=['npoints'])
    
    @property_depends_on('npoints')
    def _get_size( self ):
       return self.npoints

    @property_depends_on('npoints')
    def _get_shape( self ):
        return self.npoints

    @cached_property
    def _get_phi( self ):
        indices = arange(0, self.npoints, dtype=float) + 0.5
        phi = 2 * pi * (1 + 5**0.5)/2 * indices
        print("phi=", phi, phi.shape)
        return phi

    @cached_property
    def _get_theta( self ):
        indices = arange(0, self.npoints, dtype=float) + 0.5
        theta = arccos(1 - 2*indices/(self.npoints))
        print("theta=", theta, theta.shape)
        return theta

    def _get_gpos( self ):
        """
        Calculates grid co-ordinates.
        Returns
        -------
        array of floats of shape (3, :attr:`~Grid.size`)
            The grid points 
            oordinates in one array.
        """
        
        bpos = array([sin(self.theta) * cos(self.phi),
                      sin(self.theta) * sin(self.phi),
                      cos(self.theta)])
        return bpos
    