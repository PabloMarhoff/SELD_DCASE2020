#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 09:07:31 2020

@author: kujawski
"""

from scipy.io import wavfile
from acoular.tprocess import File, Float, CArray, digest, on_trait_change
from acoular import MaskedTimeSamples
from numpy import iinfo
from traits.api import Property,cached_property


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
        depends_on = ['amp', 'numsamples', 'sample_freq', 'name',
                      '__class__', 'start','stop'] )
               
    @cached_property
    def _get_digest( self ):
        return digest(self)
    

    @on_trait_change('name')
    def load_data( self ):
        """ 
        Open the .wav file and set attributes.
        """
        self.sample_freq, data = wavfile.read(self.name)
        norm_factor = iinfo(data.dtype).max
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
    