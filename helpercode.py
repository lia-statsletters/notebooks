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


def haversine(p1, p2, earthrad=6371.):

    """
    long and lat in decimal degrees.
    The haversine formula determines the great-circle distance between two points on a
    sphere given their longitudes and latitudes. Gives as-the-crow-flies distance between points.
    Check: https://en.wikipedia.org/wiki/Haversine_formula
    Based on: https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas"""
    lon1, lat1, lon2, lat2 = map(np.radians,
                                 [p1['lon'], p1['lat'], p2['lon'], p2['lat']])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.power(np.sin(dlat / 2.0), 2) + np.cos(lat1) * np.cos(lat2) * np.power(np.sin(dlon / 2.0),2)

    c = 2. * np.arcsin(np.sqrt(a))
    return earthrad * c



def main():
    #generateData(spst.genpareto, 1.,
    #             {'loc':1800.,'scale':3.}, 10000)
    print(haversine({'lat':57.7090,'lon':11.9745},{'lat': 57.75,'lon':11.975}) )


if __name__ == "__main__":
    main()