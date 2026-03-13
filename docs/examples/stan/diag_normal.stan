data {
    int n;
    vector[n] scales;
}
parameters {
    vector[n] x;
}
model {
    x ~ normal(0, scales);
}