import pystan
import pickle

def recompile_model(name = 'gmm_1d.pkl'):
    code = """
        data {
         int<lower = 0> N;   // number of points
         int<lower = 0> K;   // number of components
         vector[N] y;        // observed data
         vector[K] alpha;    // probabilities for the Dirichlet distribution
        }

        parameters {
          ordered[K] mu;
          real<lower=0> sigma[K];
          simplex[K] w;
        }

        model {
         real ps[K];

         sigma ~ inv_gamma(1, 1);
         mu ~ normal(0, 10);
         w ~ dirichlet(alpha);

         for (n in 1:N) {   
           for (k in 1:K) {
             ps[k] = log(w[k]) + 
                 normal_lpdf(y[n] | mu[k], sigma[k]); //increment log probability of the Gaussian
                 } 
            target += log_sum_exp(ps);
           }
        }
    """

    sm = pystan.StanModel(model_code=code)
    with open(name, 'wb') as f:
        pickle.dump(sm, f)
    return sm