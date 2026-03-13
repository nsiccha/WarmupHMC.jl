data {
    int n;
    vector[n] loc;
    vector[n] scales;
}
parameters {
    vector[n] x;
}
model {
    x ~ normal(loc, scales);
}