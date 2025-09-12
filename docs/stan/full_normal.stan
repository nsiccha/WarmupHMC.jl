data {
    int n;
    matrix[n,n] cov;
}
parameters {
    vector[n] x;
}
model {
    x ~ multi_normal(rep_vector(0., n), cov);
}