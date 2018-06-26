import numpy as np

""" 
    In this file, all methods:
     * Are intended to generate populations of users with features.
     * Returnin a column vector per user feature, optionally taking kwargs.
     
    See each method for more info on HOW was each population generated.
"""

def pop_easy(howmany,feature_labels,**kwargs):
    """
    Few features per user. Configurable yet well segmented clusters.

    :param howmany: length of column vectors to generate (samples).
    :param feature_labels: labels for user features (variables).
    :return: dictionary with feature_labels as keys and column
    vectors as values of length "howmany".
    """
    returnable={}

    return returnable