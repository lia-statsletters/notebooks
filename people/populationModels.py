import numpy as np

""" 
    In this file, all methods:
     * Are intended to generate populations of users with features.
     * Return a column vector per user feature, optionally taking kwargs.
     
    See each method for more info on HOW was each population generated.
"""

def pop_crude(howmany,feature_labels,**kwargs):
    """
    Crude population generation function.

    Few features per user. Configurable yet well segmented clusters.

    Use rectangular cuboid to bound sampling of clusters.

    Actual clusters are samples taken uniformly inside that cuboid.

    kwargs contains bounds for each cluster in the format
    {
    cluster_1: {"feature_label1": (min,max),
                ...
                "feature_labelx": (min,max),
                "percentage": % of "howmany".
               },
    clusterx: {...}
    }

    :param howmany: length of column vectors to generate (samples).
    :param feature_labels: labels for user features (variables).
    :return: dictionary with feature_labels as keys and column
    vectors as values of length "howmany". A column "cluster_label"
    is added to the output.
    """
    assert len(feature_labels) <= 3, "at most 3 features"
    samples={feature_label:[] for feature_label in feature_labels}
    samples["cluster_label"]=[]
    for cluster_label in kwargs:
        n = np.ceil(howmany * kwargs[cluster_label]["percentage"]).astype(int)
        samples["cluster_label"].extend(np.full(n,cluster_label))
        for feature_label in feature_labels:
            minl,maxl=kwargs[cluster_label][feature_label]
            samples[feature_label].extend(np.random.uniform(minl,maxl,size=n))
    return samples

def main():
    from sklearn.cluster import AgglomerativeClustering
    c1={
            "something": (1, 1),
            "age": (20, 35),
            "postcode": (41760,
                         41799),
            "percentage": 0.25
        }

    c2={
            "something": (0, 5),
            "age": (30, 55),
            "postcode": (41760,
                         41799),
            "percentage": 0.5
        }

    c3={
            "something": (0, 20),
            "age": (23, 55),
            "postcode": (41260,
                         41299),
            "percentage": 0.25
        }

    population=pop_crude(100,["something","age","postcode"],
              c1=c1,
              c2=c2,
              c3=c3)

    n_clusters = len(set(population['cluster_label']))
    fittable=np.array([population['something'],
                           population['age']]).T
    alg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    out = alg.fit(fittable)
    print(alg.labels_)


if __name__ == "__main__":
    main()