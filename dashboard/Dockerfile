FROM julia:1.6.1-buster AS main
WORKDIR /workspace
RUN julia --project=. -e "using Pkg; Pkg.add(name=\"Pluto\", rev=\"main\")"

COPY ./pluto/run_server.jl .

EXPOSE 7777
ENTRYPOINT ["julia", "--project=.", "/workspace/run_server.jl"]
