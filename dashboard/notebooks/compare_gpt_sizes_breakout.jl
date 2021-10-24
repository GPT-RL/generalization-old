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
	using Match
	Gadfly.set_default_plot_size(16cm, 14cm)
end

# ╔═╡ 654ad80a-a802-49d5-8373-b0b4056fd8f1
md"""
## Description
These graphs present a comparison on the Breakout environment of GPT architectures varying by size on the Breakout environment. We ran four distinct seeds per architecture. 
"""

# ╔═╡ 00305455-57cd-496b-9158-df7624f71c35
md"""
Note that the GPT architecture continues to exhibit instability and that many of the seeds diverge late in training. For this reason we present the first 30,000,000 million steps of training. It is noteworthy that some of the GPT architectures have an initial advantage over the baseline, but the baseline overtakes them.
"""

# ╔═╡ 9b848612-298c-4893-b8d5-15e7bf84ff24
md"""
## Analysis
Contrary to our expectations, size does not seem to consistently correlate with performance. We note that all four seeds of the 774M model diverge but the same is not true of the 1558M model.
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
- **Linear layer with output-size depending on GPT size**
- **GPT torso**
- **one `action_hidden_size` layer**
- one layer projecting to action and value
"""

# ╔═╡ 5362cb43-4d6a-41d1-b3c9-978ca15dc534
md"""
## To do
- Search gradient clipping on failed seeds.
- run forward and backward pass performance analysis for PyTorch vs. Jax
"""

# ╔═╡ 623ed8e1-7f9e-4092-bf39-b9a3d90a7d23
colors(n) = cmap("I2"; N=n);

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
				d...,
				[name => get(d, name, nothing) for name in [
							"action_hidden_size",
							"gpt",
							"time",
							"gae",
							"gradient_clip", 
							"nonlinearity", 
							"normalize_observation",
							"normalize_torso_output",
							"optimizer"
						]]... 
				), _)
		collect
	end
	vcat(DataFrame.(rows)...)
	
end;

# ╔═╡ 67944dbb-0ebb-44d9-b6f3-79e8d5610f61
begin
	gpt_comparison_data = sweep_runs([32, 30, 83, 84, 85], 300000000)
	gpt_comparison_data.gpt = replace(
		gpt_comparison_data.gpt,
		nothing => "Baseline"
	)
	plot(
        gpt_comparison_data,
		x=:step, y="episode return",
		group=:run_id,
		color=:gpt,
		Guide.xlabel("Step"),
		Guide.ylabel("Episode Return"),
		Geom.line,
		Scale.color_discrete(colors),
		Guide.colorkey(title="GPT size"),
		Guide.title("Breakout"),
		alpha=[0.5]
	) |> HTMLDocument
end

# ╔═╡ 3468f23d-f3e4-4366-b908-f01a86f385a2
begin
	gpt_early_comparison_data = sweep_runs([32, 30, 83, 84, 85], 50000000)
	gpt_early_comparison_data.gpt = replace(
		gpt_early_comparison_data.gpt,
		nothing => "Baseline"
	)
	plot(
        gpt_early_comparison_data,
		x=:step, y="episode return",
		group=:run_id,
		color=:gpt,
		Guide.xlabel("Step"),
		Guide.ylabel("Episode Return"),
		Geom.line,
		Scale.color_discrete(colors),
		Guide.colorkey(title="GPT size"),
		Guide.title("Breakout"),
		alpha=[0.5]
	) |> HTMLDocument
end

# ╔═╡ bbed8889-d401-4167-9370-9927aabbc83b
# @bind window_size Select(string.(collapse_runs ? (100:100:500) : (1:10)), default=collapse_runs ? "200" : "5")
window_size = "11";

# ╔═╡ Cell order:
# ╟─654ad80a-a802-49d5-8373-b0b4056fd8f1
# ╠═67944dbb-0ebb-44d9-b6f3-79e8d5610f61
# ╟─00305455-57cd-496b-9158-df7624f71c35
# ╟─3468f23d-f3e4-4366-b908-f01a86f385a2
# ╟─9b848612-298c-4893-b8d5-15e7bf84ff24
# ╟─8798b46c-1b0b-4d34-93ef-1d5062f8a632
# ╠═5362cb43-4d6a-41d1-b3c9-978ca15dc534
# ╟─623ed8e1-7f9e-4092-bf39-b9a3d90a7d23
# ╟─5d3bfc57-6a5a-41e5-8f63-6902e4958936
# ╟─6e3ef066-d115-11eb-2338-013a707dfe8a
# ╟─41bd06a2-e7bd-46c6-9249-6e69223b0e11
# ╟─94185c13-b6d2-4337-b5ce-336f5e128032
# ╟─0f54629c-e045-47bf-90a5-36e44c05b8f0
# ╟─bbed8889-d401-4167-9370-9927aabbc83b
