module SerializationExt

import Serialization, WarmupHMC

WarmupHMC.restore(path) = isfile(path) ? Serialization.deserialize(path) : nothing
WarmupHMC.store(path, state) = Serialization.serialize(path, state)
end