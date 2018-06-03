from scipy import stats as spst
import numpy as np
from os import environ as envx
import boto3
from boto3.dynamodb.conditions import Key, Attr
from decimal import *


def generateTimeData(distCallable,param,kwargsx,nsamples,year='2017'):
    """distCallable is a distribution from scipy.stats,
    params is a parameter of the distribution,
    kwargsx is a set of keyword arguments for distCallable in scipy,
    nsamples is the number of samples"""

    #Todo: favour weekends, week 52, spring and summer on selecting starting dates.

    # make all dates in a year (365)
    allyear = np.arange('{}-01'.format(year), '{}-12'.format(year),dtype='datetime64[m]')
    startdates=np.random.choice(allyear,size=nsamples) #select whatever stating dates uniformly
    lengths=np.array(distCallable.rvs(param,size=nsamples,**kwargsx),dtype='timedelta64[s]') #
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

    return{'{}lat'.format(prefix):list(map(Decimal,lats.astype(str))),
           '{}lon'.format(prefix):list(map(Decimal,lons.astype(str)))}

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

def write_data_to_Dynamo(dict_of_lists_to_write,aKey,base=0):
    try: #attempt to write "outdata" into "parrotbot" Dynamo table.
        dynamodb = boto3.resource('dynamodb')
        parrot_table_name=envx['parrotbot_dynamo_buffer_name']
        table=dynamodb.Table(parrot_table_name)
        # get biggest booking_ID in buffer to maintain counters
        answer=table.query(Limit=1,
                           #ScanIndexForward=False,
                           KeyConditionExpression=Key('booking_ID').gt(0)
                           )
        answer=answer[0]

        for booking_ID,item in enumerate(dict_of_lists_to_write[aKey]):
            itemx={key:item for key in dict_of_lists_to_write}
            itemx['booking_ID']=booking_ID+answer
            table.put_item(Item=itemx)
    except Exception as wtf:
        print(wtf)
        return "error!"


def parrotgen_handler(event, context):
    try:
        samples = int(envx['parrotbot_number_of_samples'])
        init_lat = float(envx['parrotbot_init_lat'])
        init_lon = float(envx['parrotbot_init_lon'])
        radio = float(envx['parrotbot_radio'])
        printable='Executing {} v.{}, weighting {} MB in memory. ' \
                  'Build {} samples at (lat:{},lon:{}) in a radio of {} ' \
                  'km'.format(envx['AWS_LAMBDA_FUNCTION_NAME'],
                        envx['AWS_LAMBDA_FUNCTION_VERSION'],
                        envx['AWS_LAMBDA_FUNCTION_MEMORY_SIZE'],
                        samples,init_lat,init_lon,radio)
        print(printable)
        outdata=generator_simple(samples,radio,init_lat,init_lon)
        write_data_to_Dynamo(outdata, 'startdates')
    except Exception as wtf:
        print(wtf)
        return "error!"

def main():
    parrotgen_handler(0,0)

if __name__ == "__main__":
    main()