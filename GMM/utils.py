import numpy as np

SEED = [183043, 236724, 394782]

def generate_datasets(n_gaussians, n_samples, seeds=SEED, means=None, limit=20, sd=0.1):
    """
    Generates a GMM of univariate gaussians with different means but same variance and same weights
    :param n_gaussians, int:
        number of components
    :param: n_samples, int: 
        number of points to sample
    :param: means, array of arrays, optional: 
        means for the gaussians
        if not given, chosen at random
        if given, should be the same length as seeds
    :param: limit, float, optional:
        needed if means are not given, the range in which to sample the means
    :param: sd, float:
        variance for each of the Gaussians
    param: seeds, optional:
        random seeds
    :return: 
        list of datasets and means
    """
    Y = []
    M = []
    for seed in seeds:
        y, means = mixture(n_gaussians, n_samples, seed=seed)
        Y.append(y)
        M.append(means)
    return Y, M

def mixture(n_gaussians, n_samples, means=None, limit=20, cov=[[1/100, 0], [0, 1/100]], seed=42):
    """
    Generates a GMM of bivariate gaussians with different means but same variance and same weights
    :param n_gaussians, int:
        number of components
    :param: n_samples, int: 
        number of points to sample
    :param: means, array, optional: 
        means for the gaussians
        if not given, chosen at random
    :param: limit, float, optional:
        needed if means are not given, the range in which to sample the means
    :param: sd, float:
        variance for each of the gaussians
    param: seed, optional:
        random seed
    :return: 
        y (sampled points), means
    """
    np.random.seed(seed)
    # generate random means in range [-limit, limit]
    if means is None:
        means = np.stack((np.random.rand(n_gaussians)*2*limit - limit, np.random.rand(n_gaussians)*2*limit - limit)).T
    # Equal weights
    w = np.ones(n_gaussians)
    w = w / np.sum(w)
    n_samples_k = (w * n_samples).astype(int)
    n_samples_k[-1] += 1 if n_samples_k.sum() < n_samples else 0
    # Equal covariance matrices
    result = np.random.multivariate_normal(means[0], cov, n_samples_k[0])
    for k in range(1, n_gaussians):
         result = np.concatenate((result, np.random.multivariate_normal(means[k], cov, n_samples_k[k])))
    
    return result, means


def mixture_biased(n_gaussians, n_samples, means=None, limit=20, center=[5,5], seed=42):
    """
    Generates a GMM of bivariate gaussians with different means but same variance and same weights
    :param n_gaussians, int:
        number of components
    :param: n_samples, int: 
        number of points to sample
    :param: means, array, optional: 
        means for the gaussians
        if not given, chosen at random
    :param: limit, float, optional:
        needed if means are not given, the range in which to sample the means
    :param: center, array:
        two-dimensional point towards which the Gaussians are biased
    param: seed, optional:
        random seed
    :return: 
        y (sampled points), means
    """
    np.random.seed(seed)    
    # generate random means
    if means is None:
        means = np.stack((np.random.rand(n_gaussians) * limit, np.random.rand(n_gaussians) * limit)).T        
    # calculate weights assigned to each component - biased towards the center
    w = 1 / np.linalg.norm(means - center, axis=1)
    w = w / w.sum()
    n_samples_k = (w * n_samples).astype(int)
    n_samples_k[-1] += n_samples - n_samples_k.sum()
    # calculate the deviation for each component
    tau = np.linalg.norm(means - center, axis=1) / n_gaussians
    cov = [[tau[0], 0], [0, tau[0]]]
    
    result = np.random.multivariate_normal(means[0], cov, n_samples_k[0])
    for k in range(1, n_gaussians):
        cov = [[tau[k], 0], [0, tau[k]]]
        result = np.concatenate((result, np.random.multivariate_normal(means[k], cov, n_samples_k[k])))
    return result, means, cov 


def univarate(n_gaussians, n_samples, seed=42, means=None, limit=20, sd=0.1):
    """
    Generates a GMM of univariate gaussians with different means but same variance and same weights
    :param n_gaussians, int:
        number of components
    :param: n_samples, int: 
        number of points to sample
    :param: means, array, optional: 
        means for the gaussians
        if not given, chosen at random
    :param: limit, float, optional:
        needed if means are not given, the range in which to sample the means
    :param: sd, float:
        variance for each of the gaussians
    param: seed, optional:
        random seed
    :return: 
        y (sampled points), means
    """
    np.random.seed(seed)
    # Equal weights
    w = np.ones(n_gaussians) / n_gaussians
    w = (w * n_samples).astype(int)
    w[n_gaussians-1] = n_samples - (np.sum(w[:-1]))
    # If not given, choose random means in range [-limit, limit]
    if means is None:
        means = np.random.rand(n_gaussians)*2*limit - limit
        
    y = np.random.normal(means[0], sd, w[0])
    for i in range(1,n_gaussians):
         y = np.concatenate((y, np.random.normal(means[i], sd, w[i])))
    
    return y, means  
    
# for Stan and NUTS    
def count_divergences(fit):
    """
    parameters:
    -----------
    fit, Stan fit object:
        obtained by calling pystan samling method
    return: 
        Total number of divergences that appeared in all the chains of MCMC algorithm
    """
    divergent_per_chain = np.column_stack([y['divergent__'] for y in fit.get_sampler_params(inc_warmup=False)]).sum(axis=0)
    return int(divergent_per_chain.sum())
    
def stan_nuts_summary(fit, params, time):
    """
    parameters:
    -----------
    fit, Stan fit object:
        obtained by calling pystan samling method
    params, array of strings:
        parameters for which to query the fit object
    time, float:
        time (in seconds) that sampling of fit took
    """
    s = fit.summary(params)
    print('{:^9s}'.format('param'), ' '.join('{:^7s}'.format(name) for name in s['summary_colnames']),
          '{:^9s}'.format('div.'), '{:^9s}'.format('time'))
    print('--------------------------------------------------------------------------------------------------------')
    for par in params:
        print( '{:^9s}'.format(par), 
               ' '.join('{:^7.2f}'.format(float(val)) for val in s['summary'][list(s['summary_rownames']).index(par),:]),
               '{:^9d}'.format(count_divergences(fit)), '{:^7.2f}'.format(time) )
        
def stan_nuts_csv(config, fit, params, time, file, append=False):
    """
    parameters:
    -----------
    config: string
        any additional information used during MCMC (e.g. nb of iterations, acceptance rate...)
    fit, Stan fit object:
        obtained by calling pystan samling method
    params, array of strings:
        parameters for which to query the fit object
    time, float:
        time (in seconds) that sampling of fit took
    file, string:
        path to the (csv) file in which to store the results
    append, boolean:
        indicates if the file should be rewritten or the results appended
    """
    
    mode = 'w+'
    if append:
        mode = 'a+'
    
    s = fit.summary(params)
    
    with open(file, mode, newline='') as csvfile:
        fieldnames = ['param']
        fieldnames += s['summary_colnames']
        fieldnames.append('divergences')
        fieldnames.append('time')
        fieldnames.append('method')
        fieldnames.append('config')
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, dialect='excel')

        writer.writeheader()
        
        writer.writerow({'method':'Stan NUTS', 'divergences':count_divergences(fit), 'time':time, 'config':config}) 
        
        for par in params:
            dict = {}
            dict['param'] = par
            for col in s['summary_colnames']:
                dict[col] = s['summary'][list(s['summary_rownames']).index(par), list(s['summary_colnames']).index(col)]
            writer.writerow(dict)
            
# for Stan and VI
def stan_vi_summary(fit, params, time):
    """
    parameters:
    -----------
    fit, Stan fit object:
        obtained by calling pystan vb method
    params, array of strings:
        parameters for which to query the fit object
    time, float:
        time (in seconds) that sampling of fit took
    """
    print('{:^9s}'.format('param'), ' '.join('{:^7s}'.format(name) for name in s['summary_colnames']),
          '{:^9s}'.format('time'))
    print('--------------------------------------------------------------------------------------------------------')
    for par in params:
        print( '{:^9s}'.format(par), 
               ' '.join('{:^7.2f}'.format(float(val)) for val in s['summary'][list(s['summary_rownames']).index(par),:]),
               '{:^7.2f}'.format(time) )