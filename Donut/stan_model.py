import pystan
import pickle

def recompile_model(name='sm_donut.pkl'):
    code = """
    data {
     int<lower=0> N;    // Number of datapoints
     int<lower=0> D;    // Number of dimensions
     vector[D] y[N];    // Observed data
    }

    parameters {
     real<lower=0> R;    // Big radius
     real<lower=0> r;    // Small radius
     vector[D] C;        // Center
     vector[D] v[N];     // random vector
    }

    transformed parameters {
     vector[D] mu[N];
     for (n in 1:N)
       mu[n] = v[n] / sqrt(dot_self(v[n]));
    }

    model {  
     R ~ normal(0, 10);
     r ~ normal(0, 10);
     C ~ normal(0, 10);

     for (n in 1:N) 
       v[n] ~ normal(0, 1); 

     for (n in 1:N)
       y[n] ~ normal(C + mu[n] * R, r);
     }"""

    print('Compiling model...', flush=True)
    sm = ps.StanModel(model_code=code)
    print('Saving the model to file...')
    with open(name, 'wb') as f:
        pickle.dump(sm, f)
    print('Done', flush=True)