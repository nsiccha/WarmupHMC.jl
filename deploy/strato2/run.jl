using WarmupHMCWeb

length(ARGS) <= 1 || error("usage: run.jl [port]")
port = isempty(ARGS) ? 8128 : parse(Int, only(ARGS))

# The release checkout moves only while the service is stopped.
WarmupHMCWeb.terminate()
WarmupHMCWeb.serve(; host="127.0.0.1", revise=nothing, port, async=true)

println("WarmupHMCWeb listening on 127.0.0.1:$port")
flush(stdout)

while true
    sleep(3600)
end
