from scipy import stats as spst
import numpy as np
from numpy import timedelta64 as td64


def generateData(distCallable,param,kwargsx,nsamples,year='2017'):
    """distCallable is a distribution from scipy.stats,
    params is a parameter of the distribution,
    kwargsx is a set of keyword arguments,
    nsamples is the number of samples"""

    #ToDo: favour weekends, week 52, spring and summer on selecting starting dates.

    # make all dates in a year (365)
    allyear = np.arange('{}-01'.format(year), '{}-12'.format(year),dtype='datetime64[D]')
    startdates=np.random.choice(allyear,size=nsamples) #select whatever stating dates uniformly
    lengths=np.array(distCallable.rvs(param,size=nsamples,**kwargsx),dtype='timedelta64[s]') #lengths
    return {'startdates':startdates,
            'enddates':startdates+lengths,
            'lengths':lengths}




def main():
    generateData(spst.genpareto, 1.,
                 {'loc':1800.,'scale':3.}, 10000)


if __name__ == "__main__":
    main()