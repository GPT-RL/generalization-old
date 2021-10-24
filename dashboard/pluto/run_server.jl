using Pluto

notebook_name = length(ARGS) > 0 ? ARGS[1] : "dashboard.jl"

Pluto.run(;
    host="0.0.0.0",
    port=7777,
    launch_browser=false,
)
