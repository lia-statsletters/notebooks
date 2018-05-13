from scipy import stats as spst
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
    thetas = spst.uniform.rvs(loc=0, scale=2*np.pi, size=nlocations)  # radians
    sinthetas = np.sin(thetas)
    costhetas = np.cos(thetas)

    # Generate what to add on lat (Y), lon (X) in kms
    Y = spst.uniform.rvs(size=nlocations)*rad

    lats = center['lat'] + Y * costhetas/110.5
    lons = center['lon']+ Y * sinthetas/(111.320*np.cos(lats))


    return{'lat':lats,
           'lon':lons}


def generateLocations3(nlocations,rad,center):
    """Use simple rules:
        - 1 degree lat = 110.5 kms
        - 1 degree long = 111.320* cos(latitude) kms
        center is given in {'lon': ,'lat':}
        consider: https://gis.stackexchange.com/questions/8650/measuring-accuracy-of-latitude-and-longitude
        and: https://stackoverflow.com/questions/1253499/simple-calculations-for-working-with-lat-lon-km-distance
    """
    clat=np.radians(center['lat'])

    #get a lazy percentage of the radius we want to capture in latitudes
    lazyp=spst.uniform.rvs(size=nlocations)*rad
    #use a quadrant mask to decide the position of the point with respect to the center
    quatrantmask=np.random.choice([1,-1],nlocations)
    lats=center['lat']+quatrantmask*lazyp/110.5
    #make another draw in the quadrant mask and for the lazy radius
    quatrantmask = np.random.choice([1, -1], nlocations)
    lazyp = spst.uniform.rvs(size=nlocations) * rad
    return {'lat': lats, 'lon': center['lon']+quatrantmask*lazyp/111.320*np.cos(lats)}


def generateLocations2(nlocations,rad,center,earthrad=6371.):
    #TODO: clean this up
    """Generate nlocations at max rad radius (in km) of a center.
    center is given in {'lon': ,'lat':}.
    """
    #Generate radios and angles
    radx = spst.uniform.rvs(size=nlocations) * rad
    angles = np.pi*spst.uniform.rvs(size=nlocations) # in radians

    #TODO:Here is the problem: the reference frame of the polars is different.
    #I should not have used this expression here.
    #place points in cartesian
    x=radx*np.cos(angles)
    y=radx*np.sin(angles)

    #centerpoint from coordinates to radians.
    clat=np.radians(center['lat'])
    #get center in cartesian
    projlon=earthrad*np.sin(np.radians(center['lon']))
    deltax=x+projlon*np.cos(clat)
    deltay=y+projlon*np.sin(clat)

    earthrad2,x2,y2=map(lambda squarables: np.power(squarables,2),
                        [earthrad, deltax, deltay])

    #return absolute long and lat of points
    longs=np.arccos(np.sqrt( 1. - ( (x2+y2)/earthrad2) ) )
    return {'lon': np.degrees(longs),
            'lat': np.degrees(np.arccos( deltax/earthrad*np.sin(longs) ))}

def haversine(p1, p2, earthrad=6371.):

    """
    long and lat in decimal degrees, returns distance in km between points.
    points given as dictionaries with keys 'lon' and 'lat'.
    The haversine determines the great-circle distance between two points on a
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
    #print(haversine({'lat':57.7090,'lon':11.9745},{'lat': 57.75,'lon':11.975}) )
    locations=generateLocations(100,10,{'lat':57.7090,'lon':11.9745})
    print ("yay")

if __name__ == "__main__":
    main()