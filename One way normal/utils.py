import numpy as np
import os
dir = os.path.dirname(__file__)

SEED = [183043, 236724, 394782]

I = 800
MU = 8
TAU = 3
SIGMA = 10

# GENERATING
def generate_datasets(I=I, mu=MU, tau=TAU, sigma=SIGMA, seeds = SEED):
    Y = []
    theta = []
    for seed in seeds:
        y, t = generate_data(I, mu, tau, sigma, seed)
        Y.append(y)
        theta.append(t)
    return Y, theta

def generate_data(I, mu, tau, sigma, seed):
    np.random.seed(seed)

    theta = np.random.normal(mu, tau, size=I)
    y = np.random.normal(theta, sigma)
    
    return y, theta


# STAN NUTS
def count_divergences(fit):
    divergent_per_chain = np.column_stack([y['divergent__'] for y in fit.get_sampler_params(inc_warmup=False)]).sum(axis=0)
    return divergent_per_chain.sum()

# filename example: nuts_nominal
def dump_stan_nuts(filename, fit, iters, warmup, time, divergences=True):
    with open(os.path.join(dir, 'results/stan/{}_fit.pkl'.format(filename)), 'wb') as f:
        pickle.dump(fit, f)

    results = {'iters': iters if iters is not None else '-', 'warmup': warmup if warmup is not None else '-',
               'time': time if time is not None else '-', 'divergences': count_divergences(fit) if divergences else '-'}
    
    with open(os.path.join(dir, 'results/stan/{}_statistics.pkl'.format(filename)), 'wb') as f:
        pickle.dump(results, f)
        
    e = fit.extract()
    with open(os.path.join(dir, 'results/stan/{}_extracted.pkl'.format(filename)), 'wb') as f:
        pickle.dump(e, f)
              