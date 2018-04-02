from scipy import stats as spst
import numpy as np
from numpy import timedelta64 as td64


def generateData(distCallable,param,kwargsx,nsamples,year='2017'):
    """distCallable is a distribution from scipy.stats,
    params is a parameter of the distribution,
    kwargsx is a set of keyword arguments for distCallable in scipy,
    nsamples is the number of samples"""

    #Todo: favour weekends, week 52, spring and summer on selecting starting dates.

    # make all dates in a year (365)
    allyear = np.arange('{}-01'.format(year), '{}-12'.format(year),dtype='datetime64[D]')
    startdates=np.random.choice(allyear,size=nsamples) #select whatever stating dates uniformly
    lengths=np.array(distCallable.rvs(param,size=nsamples,**kwargsx),dtype='timedelta64[s]') #lengths
    return {'startdates':startdates,
            'enddates':startdates+lengths,
            'lengths':lengths}

def radAngletoCoordinates(latcenter, longcenter, radius, angle, lengthDegree=111.69):
    """given a center of reference in DECIMAL degrees (latcenter, longcenter)
    and a point in polar coordinates, returns DECIMAL coordinates of
    polar point.
    Note 1: Trust this method  for less than 100km radius.
    Note 2: Default parameters are biased towards top of northern hemisphere.
    1 deg of Long = cos(lat in decimal degrees) * length of degree at that lat (km) """

    assert (radius <= 100.),'Radius {} too big. Try less than 100km.'.format(radius)
    #radius^2=(X-x0)^2+(Y-y0)^2
    #cos(angle)=x
    #km for x (longitude, meridian)


    #km for y (latitude, parallel)



def main():
    generateData(spst.genpareto, 1.,
                 {'loc':1800.,'scale':3.}, 10000)


if __name__ == "__main__":
    main()