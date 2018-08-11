import numpy as np

SEED = [137247, 472483, 701983]

def generate_data(n_samples, n_dim, seed, C=None, R=None, r=None):
    """
    Generates randomly sampled points in donut shape.
    :param n_samples: size of the dataset
    :param n_dim: number of dimensions > 0
    :param seed: random seed
    :param C: n_dim-dimensional center, optional, if None generated randomly
    :param R: >0, radius, optional, if None generated randomly
    :param r: >0, <<R, 'noise' radius, optional, if None generated randomly
    :return: sampled points, C, R, and r
    """
    np.random.seed(seed)   
    if C is None:
        C = np.random.uniform(-50, 50, n_dim).astype(int)
    if R is None:
        R = int(np.random.uniform(5, 50))
    if r is None:
        r = np.random.uniform(0.1, 2)
      
    # reset the seed   
    np.random.seed(seed)
    # pick a random point (n standard gaussians)    
    v = np.random.randn(n_samples, n_dim)
    # rescale to unit vector
    v = v / np.linalg.norm(v, axis=1)[:,None]
    x = np.random.normal(C + v*R, r)
    return x, C, R, r


def generate_datasets(n_samples, n_dim, seeds=SEED):
    """
    Generates three donut datasets.
    :param n_samples: size of the dataset
    :param n_dim: number of dimensions > 0
    :param seeds: random seeds (list of three)
    :return: list of [sampled points, C, R, r]
    """
    X = []
    C = []
    R = []
    r = []

    for seed in seeds:
        x, CC, RR, rr = generate_data(n_samples, n_dim, seed)
        X.append(x)
        C.append(CC)
        R.append(RR)
        r.append(rr)
      
    return X, C, R, r
    

def find_limits(x, threshold = 5):
    """
    Find limits for integration in KL divergence estimation. 
    Most of the probability mass is concentrated in some limited interval, 
    Even though theoretically the integral has infinite limits, 
    the computation is expensive and furthermore,
    in practice it is enough to integrate over this smaller interval.
    :param x: points from the true distribution for which KL is being calculated
    :param threshold: defualt 5, widens the scope outside of limits of x
    :return: list of lower and upper bounds for each dimension of x
    """
    x = np.array(x)
    return [x.min(axis=0) - 5, x.max(axis=0) + 5]
    
    
# for Stan NUTS    
def count_divergences(fit):
    divergent_per_chain = np.column_stack([y['divergent__'] for y in fit.get_sampler_params(inc_warmup=False)]).sum(axis=0)
    return int(divergent_per_chain.sum())

## References

# [1] Marsaglia, G. "Choosing a Point from the Surface of a Sphere." Ann. Math. Stat. 43, 645-646, 1972.
# [2] Muller, M. E. "A Note on a Method for Generating Points Uniformly on N-Dimensional Spheres." Comm. Assoc. Comput. Mach. 2, 19-20, Apr. 1959.
        