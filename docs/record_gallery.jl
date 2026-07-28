ENV["WHMC_STATIC_RECORDING"] = "1"

using HTMXObjects
using WarmupHMCWeb

appdata = WarmupHMCWeb.APPDATA
paths = ["/", "/gallery"]

isempty(paths) && error("WarmupHMCWeb has no recording paths")
all(in(appdata.recording_paths), paths) ||
    error("docs recording contains a route not registered by WarmupHMCWeb")

HTMXObjects.record!(WarmupHMCWeb.AppContext();
    record_dir = appdata.recording_dir,
    record_base = appdata.recording_base,
    paths,
    full = true,
    hx = true,
    markdown = false,
)

for path in paths
    stem = path == "/" ? "index" : path[2:end]
    for shape in ("full", "hx")
        output = shape == "full" ?
            joinpath(appdata.recording_dir, "$stem.html") :
            joinpath(appdata.recording_dir, "hx", "$stem.html")
        isfile(output) || error("recording did not produce $shape output for $path: $output")
    end
end

println("Recorded $(length(paths)) WarmupHMCWeb routes to $(appdata.recording_dir)")
