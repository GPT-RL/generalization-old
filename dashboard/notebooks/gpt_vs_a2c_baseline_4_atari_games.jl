### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ 6e3ef066-d115-11eb-2338-013a707dfe8a
begin
	using PlutoUI
	using DataFrames
	using Statistics
	using RollingFunctions
	using HTTP
	using JSON
	using Chain
	using Tables
	using PerceptualColourMaps
	using Gadfly
	Gadfly.set_default_plot_size(16cm, 14cm)
end

# ╔═╡ 9b04a5ce-318e-453a-b8ee-0b218c949b85
md"""
# Approach to experiments
The three new Atari games were chosen because they were featured in the [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/pdf/1602.01783.pdf) paper.

For each architecture I ran hyperparameter search across I broad, fixed set of hyperparameters for a fixed duration of time, forcing each run to terminate at 30,000,000 time-steps. It is worth noting that this gave some advantage to the baseline, since GPT runs slower and therefore searched fewer hyperparameters.

The results use four seeds for the the top performing hyperparameter set for each architecture.
"""

# ╔═╡ 1abe3478-49a7-443f-ab04-b1a97dea3b6b
md"""
## Changes from previous experiments
In order to run the GPT variant on smaller machines, I used the medium variant, which has 355M parameters as opposed to the XL's 1558 parameters. 

I also searched several several additional deep learning mechanics:
- Adam optimizer vs. RMSProp. Adam generally outperforms RMSProp.
- tanh vs. ReLU nonlinearity. ReLU generally outperforms tanh
- discount factor $\gamma$, searching among values recommended by [this paper](https://arxiv.org/pdf/2006.05990.pdf):
  - 0.95
  - 0.97
  - 0.99
  - 0.999
- adding layer normalization to the observation
- adding layer normalization to the torso output

Finally I restricted the search on the action hidden size (the extra layer after the GPT layers) to 
 - 32
 - 64
 - 128
"""

# ╔═╡ a078faa8-f18b-44d2-9732-96326a1fc644
md"""
## Assessment
The GPT architecture appears to learn Atari tasks with performance comparable to the A2C baseline. However it does not outperform the baseline. This indicates that gradients are flowing correctly through the frozen weights of GPT but that the current GPT architecture is not leveraging the information stored in the GPT weights to improve performance. 
"""

# ╔═╡ 654ad80a-a802-49d5-8373-b0b4056fd8f1
md"""
I accidentally left a `max_timestep` parameter set for the GPT architecture when running the Pong experiment which accounts for the repeated lines and the relatively short experiment duration. The mulitple lines do not each correspond to distinct seeds. In fact this chart demonstrates that there is non-determinism in our code somewhere, probably a lack of proper seeding. Since we are contemplating a shift to a new code-base, I suggest troubleshooting this issue after the shift.
"""

# ╔═╡ aac261d7-c785-4997-9cb2-850e0148ea3a
md"""
Note that the divergence in performance starts after 30,000,000 time-steps so a good hyperparameter search would have allowed the runs to go for longer, probably out to 1e8.
"""

# ╔═╡ 0515a101-ebfa-443c-a541-e7d8bb681fe8
md"""
These are the results that I presented last time and I included them for the sake of completeness
"""

# ╔═╡ 8798b46c-1b0b-4d34-93ef-1d5062f8a632
md"""
## Architectures
This report compares the performance of a standard A2C baseline against a "GPT-in-the-middle" variant. The architectures are as follows:

#### A2C

- Convolution with 
  - output-size $32$
  - kernel-shape $8\times 8$
  - stride $4\times 4$
- ReLU
- Convolution with 
  - output-size $64$
  - kernel-shape $4\times 4$
  - stride $2\times 2$
- ReLU
- Convolution with 
  - output-size $64$
  - kernel-shape $3\times 3$
  - stride $1\times 1$
- ReLU
- Linear layer with output-size $512$
- Parallel value head and action head

#### GPT
(differences in **bold**)

- Convolution with 
  - output-size $32$
  - kernel-shape $8\times 8$
  - stride $4\times 4$
- ReLU
- Convolution with 
  - output-size $64$
  - kernel-shape $4\times 4$
  - stride $2\times 2$
- ReLU
- Convolution with 
  - output-size $64$
  - kernel-shape $3\times 3$
  - stride $1\times 1$
- ReLU
- **Linear layer with output-size $1600$**
- **Medium-size GPT torso**
- **one `action_hidden_size` layer**
- one layer projecting to action and value
"""

# ╔═╡ f0e364db-c646-4448-9774-f6c66a3589d5
md"""
### To do
- try medium and small to compare with XL
- try deeper action head
- try double GPT
"""

# ╔═╡ 5d3bfc57-6a5a-41e5-8f63-6902e4958936
begin
	struct HTMLDocument
		embedded
	end
	function Base.show(io::IO, mime::MIME"text/html", doc::HTMLDocument)
		println(io, "<html>")
		show(io, mime, doc.embedded)
		println(io, "</html>")
	end
end;

# ╔═╡ 41bd06a2-e7bd-46c6-9249-6e69223b0e11
begin
	HASRUA_ENDPOINT_URL = "http://rldl12.eecs.umich.edu:1200/v1/graphql"
	HASURA_ADMIN_SECRET = "hNuxFSTJMk28GMiFDpZDbviKtelFQamcb20UJiYZSIZ4LYsyLCCwQFYA6Y8HRSUo"
end;

# ╔═╡ 94185c13-b6d2-4337-b5ce-336f5e128032
function gql_query(query:: String; variables:: Dict = nothing)
	r = HTTP.request(
		"POST",
		HASRUA_ENDPOINT_URL;
		verbose=3,
		body= JSON.json(Dict("query" => query, "variables" => variables)) ,
		headers=Dict("x-hasura-admin-secret" => HASURA_ADMIN_SECRET),
	)
	@chain r begin
		_.body
		String
		JSON.parse(_, null=missing)
		_["data"]
	end
end;

# ╔═╡ 8f417ce0-fa72-4b12-8d3b-8f88b2e5a2bf
function latest_runs(n::Integer)::Vector{AbstractDict}
	query = """
	  query latestRuns(\$limit: Int!){
		run(order_by: {id: desc}, limit: \$limit) {
		  id
		  metadata
		  sweep{
	        id
		  }
		}
	  }
	  """
	gql_query(query; variables=Dict("limit" => n))["run"]
end;

# ╔═╡ 48be0808-f22a-4b72-a078-b2f2552db327
function format_logs(run::AbstractDict)
	sweep_id = ismissing(run["sweep"]) ? nothing : run["sweep"]["id"]
	@chain run["run_logs"] begin
		map(x -> Dict((Symbol(replace(k, " " => "_")) => v for (k,v) in x["log"])), _)
		DataFrame
		insertcols!(:run_id => run["id"], :sweep_id => sweep_id)
	end
end;

# ╔═╡ f4d4785e-0df1-4882-8eb9-005ff2a50614
function format_run(run::AbstractDict)
	sweep_id = something(get(run, "sweep", nothing), Dict("id" => nothing))["id"]
	data = run["metadata"]
	push!(data, pop!(data, "parameters")...)
	items = [Symbol(replace(k, " " => "_")) => v for (k, v) in data]
	push!(items, :run_id => run["id"], :sweep_id => sweep_id)
	Dict(items)
end;

# ╔═╡ 0ca92cdc-102c-45e7-a5b0-17087b9ba634
function format_runs(runs::Vector)::DataFrame
	rows = map(format_run, runs)
	cols = map(keys, rows) |> Iterators.flatten |> Set
	for r in rows
		for c in setdiff(cols, keys(r))
			r[c] = nothing
		end
	end
	df = DataFrame(rows)
	leading = [:run_id, :sweep_id]
	cols = filter(x -> !(x in leading), propertynames(df))
	df[!, vcat(leading, cols)]
end;

# ╔═╡ 68e80154-126b-403c-a593-d896cdf96ef2
function run_data(ids::AbstractVector{Int}, runs::AbstractDataFrame)
	query = """
		query getRuns(\$ids: [Int!]) {
		  run(where: {id: {_in: \$ids}}) {
			id
			run_logs{
			  log
			}
			metadata
			charts{
			  spec
			}
			sweep {
				id
			}
		  }
		}
  	"""
	@chain gql_query(query; variables=Dict("ids" => ids)) begin
			_["run"]
			format_logs.(_)
			DataFrame.(_)
			vcat(_...;cols=:union)
			leftjoin(runs[!, Not(:sweep_id)]; on=:run_id)
	end
end;

# ╔═╡ 0f54629c-e045-47bf-90a5-36e44c05b8f0
function sweep_runs(sweep_ids::AbstractVector{Int}, max_step::Int)
	query = """
		query getSweepRuns(\$ids: [Int!], \$max_step: Int!) {
		  logs_less_than_step(args: {max_step: \$max_step}, where: {run: {sweep_id: {_in: \$ids}}}) {
			log
			run_id
			run {
			  metadata
			  sweep_id
			}
		  }
		}
  	"""
	rows = @chain gql_query(query; variables=Dict("ids" => sweep_ids, "max_step" => max_step)) begin
		_["logs_less_than_step"]		
		map(d -> Dict(
				"run_id" => d["run_id"],
				"sweep_id" => d["run"]["sweep_id"],
				d["log"]...,
				d["run"]["metadata"]["parameters"]...
				), _)
		map(d -> Dict(
				"action_hidden_size" => get(d, "action_hidden_size", nothing),
				"gpt" => get(d, "gpt", nothing),
				"time" => get(d, "time", nothing),
				d..., 
				), _)
		collect
	end
	vcat(DataFrame.(rows)...)
	
end;

# ╔═╡ b4f3d3e0-ef36-4674-9691-9a09de3d35e9
begin
	plot(
        sweep_runs([75, 67], 200000000),
		x=:step, y="episode return",
		group=:run_id,
		color=:gpt,
		Guide.xlabel("Step"),
		Guide.ylabel("Episode Return"),
		Geom.line,
		Scale.color_discrete(),
		Guide.colorkey(title="GPT size"),
		Guide.title("BeamRider"),
		alpha=[0.5]
	) |> HTMLDocument
end

# ╔═╡ ed1718d0-e1ed-4eb4-899d-f6d74ce40501
begin
	plot(
        sweep_runs([66, 71], 30000000),
		x=:step, y="episode return",
		group=:run_id,
		color=:gpt,
		Guide.xlabel("Step"),
		Guide.ylabel("Episode Return"),
		Geom.line,
		Scale.color_discrete(),
		Guide.colorkey(title="GPT size"),
		Guide.title("Pong"),
		alpha=[0.5]
	) |> HTMLDocument
end

# ╔═╡ 655e7222-69dc-4d39-b344-cd9c128d7b58
begin
	plot(
        sweep_runs([73, 69], 300000000),
		x=:step, y="episode return",
		group=:run_id,
		color=:gpt,
		Guide.xlabel("Step"),
		Guide.ylabel("Episode Return"),
		Geom.line,
		Scale.color_discrete(),
		Guide.colorkey(title="GPT size"),
		Guide.title("Q*Bert"),
		alpha=[0.5]
	) |> HTMLDocument
end

# ╔═╡ 67944dbb-0ebb-44d9-b6f3-79e8d5610f61
begin
	plot(
        sweep_runs([30, 32], 400000000),
		x=:step, y="episode return",
		group=:run_id,
		color=:gpt,
		Guide.xlabel("Step"),
		Guide.ylabel("Episode Return"),
		Geom.line,
		Scale.color_discrete(),
		Guide.colorkey(title="GPT size"),
		Guide.title("Breakout"),
		alpha=[0.5]
	) |> HTMLDocument
end

# ╔═╡ bbed8889-d401-4167-9370-9927aabbc83b
# @bind window_size Select(string.(collapse_runs ? (100:100:500) : (1:10)), default=collapse_runs ? "200" : "5")
window_size = "11";

# ╔═╡ Cell order:
# ╟─9b04a5ce-318e-453a-b8ee-0b218c949b85
# ╟─1abe3478-49a7-443f-ab04-b1a97dea3b6b
# ╟─a078faa8-f18b-44d2-9732-96326a1fc644
# ╟─b4f3d3e0-ef36-4674-9691-9a09de3d35e9
# ╟─ed1718d0-e1ed-4eb4-899d-f6d74ce40501
# ╟─654ad80a-a802-49d5-8373-b0b4056fd8f1
# ╟─655e7222-69dc-4d39-b344-cd9c128d7b58
# ╟─aac261d7-c785-4997-9cb2-850e0148ea3a
# ╟─67944dbb-0ebb-44d9-b6f3-79e8d5610f61
# ╟─0515a101-ebfa-443c-a541-e7d8bb681fe8
# ╟─8798b46c-1b0b-4d34-93ef-1d5062f8a632
# ╠═f0e364db-c646-4448-9774-f6c66a3589d5
# ╠═5d3bfc57-6a5a-41e5-8f63-6902e4958936
# ╠═6e3ef066-d115-11eb-2338-013a707dfe8a
# ╟─41bd06a2-e7bd-46c6-9249-6e69223b0e11
# ╟─94185c13-b6d2-4337-b5ce-336f5e128032
# ╟─8f417ce0-fa72-4b12-8d3b-8f88b2e5a2bf
# ╟─48be0808-f22a-4b72-a078-b2f2552db327
# ╟─f4d4785e-0df1-4882-8eb9-005ff2a50614
# ╟─0ca92cdc-102c-45e7-a5b0-17087b9ba634
# ╟─68e80154-126b-403c-a593-d896cdf96ef2
# ╟─0f54629c-e045-47bf-90a5-36e44c05b8f0
# ╟─bbed8889-d401-4167-9370-9927aabbc83b
