module SerializationExt

import Serialization, WarmupHMC

WarmupHMC.restore(path) = isfile(path) ? Serialization.deserialize(path) : nothing
WarmupHMC.store(path, state) = Serialization.serialize(path, state)
# WarmupHMC.initialize_state(::Nothing, state_path::AbstractString) = WarmupHMC.initialize_state(isfile(state_path) ? Serialization.deserialize(state_path) : (;), state_path)
# WarmupHMC.save_state(state::NamedTuple) = if hasproperty(state, :state_path)
#     @info "Writing to $(state.state_path)!" 
#     Serialization.serialize(state.state_path, state)
# end
end