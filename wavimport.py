#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 09:07:31 2020

@author: kujawski
"""

from scipy.io import wavfile
from acoular.tprocess import *
from acoular.signals import *
from acoular import TimeSamples,MaskedTimeSamples, Grid
from numpy import pi, array, cos, sin, arange, iinfo, arccos, arcsin, repeat, concatenate
from traits.api import property_depends_on,Property,cached_property
from pylab import deg2rad

class WavSamples( MaskedTimeSamples ):
    """
    Import wav file and use as signal generator. 
    Signal from wav is sound pressure at 1m distance from source.
    """
    
    #: Name of the file to import.
    name = File(filter = ['*.wav'],
                desc = "name of the *.wav file to import")

    #: amplitude multiplicator of source signal (for point source: in 1 m distance).
    amp = Float(1.0, desc="amplification of wav signal")
    
    #: The signal data
    data = CArray()

    # internal identifier
    digest = Property( 
        depends_on = ['amp', 'numsamples', \
        'sample_freq', 'name', '__class__'], 
        )
               
    @cached_property
    def _get_digest( self ):
        return digest(self)
    

    @on_trait_change('name')
    def load_data( self ):
        """ 
        Open the .wav file and set attributes.
        """
        self.sample_freq, data = wavfile.read(self.name)
        norm_factor = iinfo(self.data.dtype).max
        self.data = data/norm_factor
        self.numsamples_total = self.data.shape[0]
        self.numchannels_total = self.data.shape[1]
    
    def signal(self):
        """
        Deliver the signal.

        Returns
        -------
        Array of floats
            The resulting signal as an array of length :attr:`~SignalGenerator.numsamples`.
        """
        norm_factor = iinfo(self.data.dtype).max
        return self.data / norm_factor
    


class SphericalGrid_Equiangular( Grid ):
    """
    Provides a spherical grid for beamforming.
    Ratio of the Gridpoints should be Azimuth:Elevation = 2:1
    """
     
    def __init__(self,npoints_azi,npoints_ele):
        self.set_npoints_azi(npoints_azi)
        self.set_npoints_ele(npoints_ele)

    def set_npoints_azi(self,npoints_azi):
        self.__npoints_azi = npoints_azi
        
    def set_npoints_ele(self,npoints_ele):
        self.__npoints_ele = npoints_ele
    
    # def get_npoints_azi(self):
    #     return self.__npoints_azi
    
    # def get_npoints_ele(self):
    #     return self.__npoints_ele
        
#    npoints = Int(npoints_azi*npoints_ele)
    #: how many grid points
    
    phi = Property(depends_on=['__npoints_azi'])

    theta = Property(depends_on=['__npoints_ele'])
    

    
    @property_depends_on('__npoints_azi','__npoints_ele')
    def _get_size( self ):
        return (self.__npoints_azi * self.__npoints_ele)

    @property_depends_on('__npoints_azi','__npoints_ele')
    def _get_shape( self ):
        return (self.__npoints_azi * self.__npoints_ele)

    @cached_property
    def _get_phi( self ):
        indices = arange(0, self.__npoints_azi, dtype=float) + 0.5
        phi = repeat(2*pi * (indices / self.__npoints_azi), self.__npoints_ele)
        print("Phi=", phi)
        return phi

    @cached_property
    def _get_theta( self ):
        indices = arange(0, self.__npoints_ele, dtype=float)
        theta = []
        theta_tmp = pi * ((indices+0.5)/self.__npoints_ele)
        for i in range (self.__npoints_azi):
            theta = concatenate((theta, theta_tmp), axis=None)
        print("Theta=", theta)
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
    