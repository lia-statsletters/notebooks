from scipy import stats as spst
import numpy as np
from os import environ as envx
import boto3
from boto3.dynamodb.conditions import Key, Attr
from decimal import *

def generateLengths(distCallable,param,kwargsx,nsamples):
    #generates lengths in seconds
    returnable=np.array(distCallable.rvs(param, size=nsamples, **kwargsx), dtype='timedelta64[s]')
    if nsamples>1:
        return returnable
    return returnable[0]


def generateTimeData(distCallable,param,kwargsx,nsamples,year='2017'):
    """distCallable is a distribution from scipy.stats,
    params is a parameter of the distribution,
    kwargsx is a set of keyword arguments for distCallable in scipy,
    nsamples is the number of samples"""

    #Todo: favour weekends, week 52, spring and summer on selecting starting dates.

    # make all dates in a year (365)
    allyear = np.arange('{}-01'.format(year), '{}-12'.format(year),dtype='datetime64[m]')
    startdates=np.random.choice(allyear,size=nsamples) #select whatever stating dates uniformly
    lengths=generateLengths(distCallable,param,kwargsx,nsamples)
    ends=startdates+lengths
    return {'startdates':(startdates.astype(str)).tolist(),
            'enddates':(ends.astype(str)).tolist(),
            'lengths':(lengths.astype(int)).tolist()}

def generateLocations(nlocations,rad,center,prefix='start'):
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
    Y = np.random.uniform(size=nlocations)*rad

    lats = center['lat'] + Y * costhetas/110.5
    lons = center['lon']+ Y * sinthetas/(111.320*np.cos(lats))


    returnable={'{}lat'.format(prefix):list(map(Decimal,lats.astype(str))),
               '{}lon'.format(prefix):list(map(Decimal,lons.astype(str)))}
    if nlocations > 1:
        return returnable
    #otherwise, return as scalar
    return {'{}lat'.format(prefix): returnable['{}lat'.format(prefix)][0],
            '{}lon'.format(prefix): returnable['{}lon'.format(prefix)][0]}

def generator_simple(samples,radio,init_lat,init_lon):
    #relies on environ
    returnable={}
    returnable.update(generateLocations(samples,radio,
                                  {'lat':init_lat,'lon':init_lon},
                                  prefix='start'))
    returnable.update(generateLocations(samples, radio,
                                {'lat':init_lat, 'lon':init_lon},
                                prefix='end'))
    #perhaps later pick this distribution from elsewhere as a parameter.
    returnable.update(generateTimeData(spst.genpareto, 1.,
                                {'loc':1800.,'scale':3.}, samples))

    return returnable

def write_to_Dynamo(dict_of_lists_to_write,aKey):
    try: #attempt to write "outdata" into "parrotbot" Dynamo table.
        dynamodb = boto3.resource('dynamodb')
        parrot_table_name=envx['parrotbot_dynamo_buffer_name']
        table=dynamodb.Table(parrot_table_name)

        for booking_ID,item in enumerate(dict_of_lists_to_write[aKey]):
            itemx={key:item for key in dict_of_lists_to_write}
            table.put_item(Item=itemx)
    except Exception as wtf:
        print(wtf)

def parrotgen_instant_handler(event,context):
    try:
        init_lat = float(envx['parrotbot_init_lat'])
        init_lon = float(envx['parrotbot_init_lon'])
        radio = float(envx['parrotbot_radio'])
        outdata={}
        outdata.update(generateLocations(1, radio,
                                            {'lat': init_lat, 'lon': init_lon},
                                            prefix='start'))
        outdata.update(generateLocations(1, radio,
                                            {'lat': init_lat, 'lon': init_lon},
                                            prefix='end'))
        lengths = generateLengths(spst.genpareto, 1.,
                                {'loc':1800.,'scale':3.}, 1)
        now=np.datetime64('now')
        outdata.update({'startdates': now.astype(str)})
        outdata.update({'enddates':(now+lengths).astype(str)})
        outdata.update({'lenghts':int(lengths.astype(int))})

        dynamodb = boto3.resource('dynamodb')
        parrot_table_name = envx['parrotbot_dynamo_buffer_name']
        table = dynamodb.Table(parrot_table_name)
        table.put_item(Item=outdata)
        #TODO: MAKE SURE WE CAN DO A PUT OPERATION HERE.

        print(outdata)

    except Exception as wtf:
        print(wtf)


def parrotgen_historic_handler(event, context):
    try:
        samples = int(envx['parrotbot_number_of_samples'])
        init_lat = float(envx['parrotbot_init_lat'])
        init_lon = float(envx['parrotbot_init_lon'])
        radio = float(envx['parrotbot_radio'])
        outdata=generator_simple(samples,radio,init_lat,init_lon)
        write_data_to_Dynamo(outdata, 'startdates')
    except Exception as wtf:
        print(wtf)

def main():
    #parrotgen_historic_handler(0,0)
    parrotgen_instant_handler(0,0)

if __name__ == "__main__":
    main()