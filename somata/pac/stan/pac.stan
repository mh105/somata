data {
    int<lower=0> n;
    vector[n] phase;
    vector[n] amplitude;
}

parameters {
    real beta1;
    real beta2;
    real<lower=sqrt(beta1^2 + beta2^2)> beta0;
    real<lower=0> sigma;
}

model {
    target += normal_lpdf(amplitude | beta0 + beta1 * cos(phase) + beta2 * sin(phase), sigma);
}