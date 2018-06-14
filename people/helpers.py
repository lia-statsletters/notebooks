def generatePurchaseData(n,k,user_labels,spec_labels,
                         userGenFunc,specGenFunc,
                         purchasingFunc):
    """Generates data matching the following conditions:
            - Set N has "n" users, each user has "x" attributes.
            - Set K has "k" purchases, each purchase has "y" attributes.
            - To connect any "n_i" user with a list "L" of purchases, we use
             a "purchasing" function. Note that "L" is a List instead of a set
             due to the fact that is possible to have more than one instance
             of each purchase.

        The "purchasing" function can be as arbitrary or informed as
        we want it to be. I want to use it to model different relations between
        specs and user models.

        Populations of specs and users can be generated separately with
        other functions.

        To generate dependent populations (spec populations depending on user
        populations and vice-versa) the *intended* use pattern is placing the
        independent population data as part of the kwargs.

    """

    #To-Do: This is a stub. Assorted patterns for generators below.
    N = makePopulation(n,user_labels,userGenFunc)
    K = makePopulation(k,spec_labels,
                       specGenFunc,
                       users=N)
    return purchasingFunc(N,K)

def makePopulation(howmany,feature_labels,genFunc,**kwargs):
    """
    :param howmany: number of rows to generate
    :param feature_labels: labels for user features
    :param genFunc: a function returning rows and user features,
    optionally taking kwargs.
    :return: dictionary with feature_labels as keys and an element per row.
    """
    assert len(feature_labels)>0, "At least one feature please."
    return genFunc(howmany,feature_labels,**kwargs)







