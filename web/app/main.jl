using Revise
using WarmupHMCWeb

begin
    WarmupHMCWeb.terminate()
    port = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8090
    WarmupHMCWeb.serve(; host="0.0.0.0", revise=:lazy, port, async=true)
end
