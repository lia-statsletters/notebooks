from scipy import stats as spst
import numpy as np

def generateData(distCallable,param,kwargsx,nsamples):
    """distCallable is a distribution from scipy.stats,
    params is a parameter of the distribution,
    kwargsx is a set of keyword arguments,
    nsamples is the number of samples"""
    try:
        r = distCallable.rvs(param,size=nsamples,**kwargsx)
    except Exception as whathappened:
        print (whathappened.message)
        return
    startdays=np.random.random_integers(0,365,nsamples)
    for sample in enumerate(r):
        #generating dates in 2017.
        delta=np.datetime64(sample,'s')




def main():
    generateData(spst.genpareto, 1.,
                 {'loc':1800.,'scale':3.}, 10000)


if __name__ == "__main__":
    main()