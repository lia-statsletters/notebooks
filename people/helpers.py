from scipy import stats as spst
import numpy as np
import boto3

from decimal import *

def generatePurchaseData():
    """Generates data matching the following conditions:
            - Set N has "n" users, each user has "x" attributes.
            - Set K has "k" purchases, each purchase has "y" attributes.
            - To connect any "n_i" user with a list "L" of purchases, we use
             a "purchasing" function. Note that "L" is a List instead of a set
             due to the fact that is possible to have more than one instance
             of each purchase.

        The "purchasing" function can be as arbitrary or informed as
        we want it to be. I want to use it to model different relations between
        purchasing and user models.

        Populations of purchases and users can be generated separately with
        other functions.
    """
    #To-Do: This is a stub. Assorted patterns for generators below.
    returnable=np.array(distCallable.rvs(param, size=nsamples, **kwargsx), dtype='timedelta64[s]')

    # make all dates in a year (365)
    allyear = np.arange('{}-01'.format(year), '{}-12'.format(year),dtype='datetime64[m]')
    startdates=np.random.choice(allyear,size=nsamples) #select whatever stating dates uniformly

    return {'startdates':(startdates.astype(str)).tolist()}
