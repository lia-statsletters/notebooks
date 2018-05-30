import numpy as np
from numpy import timedelta64 as td64


def generateTimeData(distCallable,param,kwargsx,nsamples,year='2017'):
    """distCallable is a distribution from scipy.stats,
    params is a parameter of the distribution,
    kwargsx is a set of keyword arguments for distCallable in scipy,
    nsamples is the number of samples"""

    #Todo: favour weekends, week 52, spring and summer on selecting starting dates.

    # make all dates in a year (365)
    allyear = np.arange('{}-01'.format(year), '{}-12'.format(year),dtype='datetime64[D]')
    startdates=np.random.choice(allyear,size=nsamples) #select whatever stating dates uniformly
    lengths=np.array(distCallable.rvs(param,size=nsamples,**kwargsx),dtype='timedelta64[s]') #
    return {'startdates':startdates,
            'enddates':startdates+lengths,
            'lengths':lengths}
			
def generateLocations(nlocations,rad,center):
    """
    Assumption warning: xtremely lazily treating lat and long as Y and X in cartesian.
    Use simple rules:
        - 1 degree lat = 110.5 kms
        - 1 degree long = 111.320* cos(latitude) kms
        center is given in {'lon': ,'lat':}
        consider: https://gis.stackexchange.com/questions/8650/measuring-accuracy-of-latitude-and-longitude
        and: https://stackoverflow.com/questions/1253499/simple-calculations-for-working-with-lat-lon-km-distance
    """
    # Rotation matrix x'= xcos(theta)+ysin(theta), y'=-xsin(theta)+ycos(theta)
    thetas = np.random.uniform(low=0.,high=2*np.pi,size=nlocations)  # radians
    sinthetas = np.sin(thetas)
    costhetas = np.cos(thetas)

    # Generate what to add on lat (Y), lon (X) in kms
    Y = spst.uniform.rvs(size=nlocations)*rad

    lats = center['lat'] + Y * costhetas/110.5
    lons = center['lon']+ Y * sinthetas/(111.320*np.cos(lats))

    return{'lat':lats,
           'lon':lons}

def handler(event, context):
    #relies on environ
    try:
        printable='Executing {} v.{}, weighting {} MB in memory.'.format(envx['AWS_LAMBDA_FUNCTION_NAME'],
                        envx['AWS_LAMBDA_FUNCTION_VERSION'],
                        envx['AWS_LAMBDA_FUNCTION_MEMORY_SIZE'])
        print printable
    except Exception as wtf:
        print(wtf.message)

