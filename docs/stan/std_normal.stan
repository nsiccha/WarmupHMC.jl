data {
    int n;
}
parameters {
    vector[n] x;
}
model {
    x ~ std_normal();
}