import numpy as np
from scipy.special import erf


def make_F(loc, std, amp):
    """
    Make the cumulative distribution function of Y in terms of layers
    Inputs:
        loc - mean of CDF/location of the layer

        std - standard deviation/width of the layer

        amp - amplitude/relative size of the layer compared to the others

    Returns:
        F_Y = sum_n F_Y^n - a sum of the CDFs for every layer
    """
    def F_Y(y):
        Y = 0*y
        for n in range(len(loc)):
            Y += amp[n]*(1 + erf( (y-loc[n])/(np.sqrt(2)*std[n]) ))/2
        return Y
    return F_Y


def make_f(loc, std, amp):
    """
    Make the probability distribution function of Y in terms of layers
    Inputs:
        loc - mean of CDF/location of the layer

        std - standard deviation/width of the layer

        amp - amplitude/relative size of the layer compared to the others

        y - the grid on which the PDF is evaluated.
    Returns:
        f_Y = sum_n f_Y^n - a sum of the PDFs for every layer
    """

    def f_Y(y):
        Y = 0*y
        for n in range(len(loc)):
            Y += ( amp[n]/np.sqrt(2*np.pi*std[n]**2) )*np.exp( -((y-loc[n])**2)/(2*std[n]**2) )
        return Y
     
    return f_Y
