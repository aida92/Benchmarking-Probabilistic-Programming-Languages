import pystan
import pickle

def recompile_centered_model(name='1wayN_centered.pkl'):
    code = """
    data {
     int<lower=0> I;          // Number of datapoints
     real y[I];               // Observed data
     real<lower=0> sigma[I];  // Parameter for the Gaussian
    }

    parameters {
     real mu;    
     real<lower=0> tau;
     real theta[I];
    }

    model {  
     mu ~ normal(0, 5);
     tau ~ cauchy(0, 2.5);
     theta ~ normal(mu, tau);

     y ~ normal(theta, sigma);
     }"""

    print('Compiling model...')
    sm = pystan.StanModel(model_code=code)
    print('Saving the model to file...')
    with open(name, 'wb') as f:
        pickle.dump(sm, f)
    print('Done')
    
def recompile_non_centered_model(name='1wayN_noncentered.pkl'):
    code = """
    data {
     int<lower=0> I;          // Number of datapoints
     real y[I];               // Observed data
     real<lower=0> sigma[I];  // Parameter for the Gaussian
    }

    parameters {
     real mu;    
     real<lower=0> tau;
     real var_theta[I];
    }
    
    transformed parameters {
      real theta[I];
      for (i in 1:I)
        theta[i] = tau*var_theta[i] + mu;
    }

    model {  
     mu ~ normal(0, 5);
     tau ~ cauchy(0, 2.5);
     var_theta ~ normal(0, 1);

     y ~ normal(theta, sigma);
     }"""

    print('Compiling model...')
    sm = pystan.StanModel(model_code=code)
    print('Saving the model to file...')
    with open(name, 'wb') as f:
        pickle.dump(sm, f)
    print('Done')