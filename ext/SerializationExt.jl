module SerializationExt
import WarmupHMC
import Serialization: serialize, deserialize
WarmupHMC.initialize_state(::Nothing, state_path::AbstractString) = WarmupHMC.initialize_state(isfile(state_path) ? deserialize(state_path) : (;), state_path)
WarmupHMC.save_state(state::NamedTuple) = if hasproperty(state, :state_path)
    @info "Writing to $(state.state_path)!" 
    serialize(state.state_path, state)
end
end