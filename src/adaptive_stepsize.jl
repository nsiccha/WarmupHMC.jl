dual_averaging!!!(
    current_acceptance_rate;
    target_acceptance_rate=.8, regularization_scale=.05, relaxation_exponent=.75, offset=10,
    current_log_stepsize=0., shrinkage_target=log(10.)+current_log_stepsize, counter=0, Hbar=0., final_log_stepsize=0.

) = begin
    counter += 1
    Hbar += (target_acceptance_rate - current_acceptance_rate - Hbar) / (counter+offset) 
    current_log_stepsize = shrinkage_target - sqrt(counter) / regularization_scale * Hbar
    final_log_stepsize += counter ^ (-relaxation_exponent) * (current_log_stepsize - final_log_stepsize)
end