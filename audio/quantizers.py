#!/usr/bin/env python
# coding: utf-8

"Quantizers"

# Python 2.7 Standard Library
from __future__ import division

# Third-Party Librairies
import numpy as np

#
# Metadata
# ------------------------------------------------------------------------------
#

from .about_quantizers import *


#
# Quantizer Base Class
# ------------------------------------------------------------------------------
#

class Quantizer(object):
    """
    Quantizers Base Class.
    """
    def encode(self, data):
        "Encode data"
        raise NotImplementedError("undefined forward quantizer")

    def decode(self, data):
        "Decode data"
        raise NotImplementedError("undefined inverse quantizer")
    
    def __call__(self, data):
        "Chain encoding and decoding of the data"
        return self.decode(self.encode(data))

#
# Linear (Uniform) Quantizer
# ------------------------------------------------------------------------------
#

class Uniform(Quantizer):
    "Uniform Quantizer"

    def __init__(self, low=-1.0, high=1.0, N=2**8):
        """
        Arguments
        ----------

          - `low, high`: quantizer range,
        
          - `N`: number of quantizer values.        
        """
        self._low  = low
        self._high = high
        # N=0 and N=1 are equivalent, either way the encoding returns no 
        # information about the data. It's simpler to assume that N>0
        # for encode/decode computations, so if N=0, we still pick N=1.
        self._N = max(1, N) 


        if N <= 2**8:
            self._integer_type = np.uint8
        elif N <= 2**16:
            self._integer_type = np.uint16
        elif N <= 2**32:
            self._integer_type = np.uint32
        elif N <= 2**53:
            # Above 2**53, integers representation as floats is incorrect.
            # The encode method cannot return 2**53 + 1 for example, but 
            # would produce 2**53 instead as float(2**53 + 1) == 2**53.
            self._integer_type = np.uint64
        else:
            error = "N={0} is too big: only N <= 2**53 is supported."
            raise ValueError(error.format(N))

    def encode(self, data):
        low, high = self._low, self._high
        delta = (high - low) / self._N
        data = np.clip(data, low + delta/2.0, high - delta/2)
        flints = np.round((data - low) / delta - 0.5)
        return np.array(flints, dtype=self._integer_type)

    def decode(self, i):
        delta = (self._high - self._low) / self._N
        return self._low + (i + 0.5) * delta

Linear = Uniform

#
# NonLinear Quantizer
# ------------------------------------------------------------------------------
#

class NonLinear(Quantizer):
    "Nonlinear quantizer based on a characteristic function"

    def __init__(self, f, f_inv, quantizer):
        """
        Arguments
        ---------
        `f:` 
          : characteristic function
        
        `f_inv:`
          : inverse of the characteristic function

        `quantizer:`
          : the internal quantizer
        """
        self.f = f
        self.f_inv = f_inv
        self.quantizer = quantizer

    def encode(self, data):
        return self.quantizer.encode(self.f(data))

    def decode(self, i):
        return self.f_inv(self.quantizer.decode(i))

#
# Mu-Law Quantizer
# ------------------------------------------------------------------------------
#

class MuLaw(Quantizer):
    """
    Mu-law quantizer
    """
    scale  = 32768
    iscale = 1.0 / scale
    bias   = 132
    clip   = 32635
    etab   = np.array([0, 132, 396, 924, 1980, 4092, 8316, 16764])
    
    @staticmethod
    def sign(data):
        """
        Sign function such that sign(+0) = 1 and sign(-0) = -1
        """
        data = np.array(data, dtype=float)
        settings = np.seterr(divide="ignore")
        try:
            inv_data = 1.0 / data
        finally:
            np.seterr(**settings)
        zero_p = (inv_data == + np.inf)
        zero_m = (inv_data == - np.inf)
        return np.sign(data) + zero_p - zero_m

    def encode(self, data):
        data = np.array(data)
        s = MuLaw.scale * data
        s = np.minimum(abs(s), MuLaw.clip)
        [f,e] = np.frexp(s + MuLaw.bias)

        step  = np.floor(32 * f) - 16   # 4 bits
        chord = e - 8                   # 3 bits
        sgn   = (MuLaw.sign(data) == 1) # 1 bit

        mu = 16 * chord + step # 7-bit coding
        mu = 127 - mu          # bits inversion
        mu = 128 * sgn + mu    # final 8-bit coding
	
        return np.array(mu, dtype=np.uint8)
    
    def decode(self, i):
        i = np.array(i)
        i = 255 - i
        sgn = i > 127
        e = np.array(np.floor(i / 16.0) - 8 * sgn + 1, dtype=np.uint8)
        f = i % 16
        data = np.ldexp(f, e + 2)
        e = MuLaw.etab[e-1]
        data = MuLaw.iscale * (1 - 2 * sgn) * (e + data)
	
        return data

mulaw = MuLaw()

#
# Scale Factor Vector Quantizer
# ------------------------------------------------------------------------------
#

class ScaleFactor(Quantizer):
    "Scale Factor Quantizer"
    def __init__(self, scale_factors, quantizer=None):
        """
        Arguments
        ---------

          - `scale_factors` : an increasing sequence of numbers,

          - `quantizer` : a `Quantizer` instance.

        The `index` method can still be used if the quantizer is not specified.
        The `quantizer` can be set after the construction stage, it is a public
        attribute.
        """
        if not all(np.diff(scale_factors) > 0):
            error = "the sequence of scale factors is not increasing"
            raise ValueError(error)
        else:
            self.scale_factors = np.array(scale_factors, dtype=float)
        self.quantizer = quantizer

    def index(self, data):
        """
        Return the scale factor index for a numeric sequence `data`.
        """
        max_ = np.max(np.abs(data))
        i = np.searchsorted(self.scale_factors, max_, side="left")
        return min(i, len(self.scale_factors) - 1)
      

# ... DEPRECATED ...............................................................
#    def index(self, data):
#        max_ = np.max(np.abs(data))
#        index = len(self.scale_factors) - 1
#        while index >= 1:
#            if self.scale_factors[index - 1] < max_:
#                break
#            else:
#                index = index - 1
#        return index
# ..............................................................................

    def encode(self, data):
        """
        Argument
        --------

          - `data`: a sequence of floating-point numbers.

        Returns
        -------

          - `i`: the scale factor index,

          - `codes`: sequence of integers, the quantizer codes.
        """
        i = self.index(data)
        sf = self.scale_factors[i]
        return (i, self.quantizer.encode(data / sf))

    def decode(self, data):
        """
        Argument
        --------

          - `data`: a 2-uple `(i, codes)` (see `encode` method).
        """

        i, encoded_data = data
        sf = self.scale_factors[i]
        return sf * self.quantizer.decode(encoded_data)


# TODO: CodeBook Approach (Shape-Gain VQ).

class CodeBook(object):
    def __init__(self, codebook, criteria):
        pass
    def encode(self, data):
        pass
    def decode(self, data):
        pass

#
# Unit Tests
# ------------------------------------------------------------------------------
#

def test_sign():
    """
    Sign function that works on IEEE 754 extended reals:

        >>> from numpy import inf
        >>> MuLaw.sign([-inf, -1.0, -0.0, 0.0, 1.0, +inf])
        array([-1., -1., -1.,  1.,  1.,  1.])
    """

def test_mulaw():
    """
    basic consistency check:

        >>> codes = range(0, 256)
        >>> all(mulaw.encode(mulaw.decode(codes)) == codes)
        True
    """

def test_scale_factors():
    """
    Scale factor quantizer:

    Index method:

        >>> sf = ScaleFactor(scale_factors=[1.0, 2.0, 4.0])
        >>> sf.index(0.0), sf.index(0.5), sf.index(1.0)
        (0, 0, 0)
        >>> sf.index(1.5), sf.index(2.0)
        (1, 1)
        >>> sf.index(3.0), sf.index(4.0), sf.index(5.0)
        (2, 2, 2)
        >>> sf.index([-1.0, 0.0, 1.0])
        0
        >>> sf.index([-2.0, -1.0, 0.0, 1.0, 2.0])
        1
        >>> sf.index([-1000.0, 1000.0])
        2

   Encoder:

        >>> sf.quantizer = Uniform(low=-1.0, high=1.0, N=3)
        >>> sf.encode([-1.0, 0.0, 1.0])
        (0, array([0, 1, 2], dtype=uint8))
        >>> sf.encode([-2.0, -1.0, 0.0, 1.0, 2.0])
        (1, array([0, 0, 1, 2, 2], dtype=uint8))
        >>> sf.encode([-1000.0, 1000.0])
        (2, array([0, 2], dtype=uint8))
    """

