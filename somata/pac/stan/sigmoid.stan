data {
    int<lower=0> N;
    vector[N] swa;
    vector[N] ce;
}

parameters {
    real<lower=0> s;
    real<lower=0> r;
    real<lower=0, upper=4> t;
    real<lower=0, upper=1> u;
    real log_sigma;
}

transformed parameters {
    real sigma = exp(log_sigma);  // Jeffreys prior is uniform on log(sigma)
}

model {
    vector[N] logistic_error = swa - (s / (1 + exp(-(ce - t) / u))) - r;
    logistic_error ~ normal(0, sigma);
}