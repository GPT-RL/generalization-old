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

# ╔═╡ 0be25163-f9c1-41bc-82cf-9f63ff8e609a
md"# Stabilizing the GPT architecture"

# ╔═╡ 654ad80a-a802-49d5-8373-b0b4056fd8f1
md"""
For this report we focused on stabilizing GPT's learning. We discovered that GPT was using an unusually low discount value for that large but unstable run that we showcased last time and found that changing this value improved stability but did not entirely fix the issueas the following graph indicates:
"""

# ╔═╡ ab745b36-164c-4534-be1f-a703a2010f3e
md"""
Note that all but one GPT seed diverges, though one does so only after nearly 1.5e8 steps. We are currently searching other hyperparameters with the updated discount value to see if these improve stability. Unfortunately all but `num_embeddings=1` and `num_embeddings=2` crashed due to memory issues on the smaller machines. Experiments with larger embedding numbers are currently running on rldl18 but have not yet converged:
"""

# ╔═╡ 9b848612-298c-4893-b8d5-15e7bf84ff24
md"""
## Status on other tasks
- Moving to Pytorch repository (almost done)
- Analyzing perception outputs (Logan)
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
							"optimizer",
							"num_embeddings", 
							"save_interval",
							"save_path"
						]]... 
				), _)
		collect
	end
	vcat(DataFrame.(rows)...)
	
end;

# ╔═╡ 67944dbb-0ebb-44d9-b6f3-79e8d5610f61
begin
	gpt_comparison_data = sweep_runs([98, 32], 300000000)
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

# ╔═╡ f4368e0e-dd79-4a80-bae0-84a09de65e97
begin
	plot(
        sweep_runs([99, 103], 300000000),
		x=:step, y="episode return",
		group=:run_id,
		color="num_embeddings",
		Guide.xlabel("Step"),
		Guide.ylabel("Episode Return"),
		Geom.line,
		Scale.color_discrete(colors),
		Guide.colorkey(title="#embeddings"),
		Guide.title("Breakout"),
		alpha=[0.5]
	) |> HTMLDocument
end

# ╔═╡ bbed8889-d401-4167-9370-9927aabbc83b
# @bind window_size Select(string.(collapse_runs ? (100:100:500) : (1:10)), default=collapse_runs ? "200" : "5")
window_size = "11";

# ╔═╡ Cell order:
# ╟─0be25163-f9c1-41bc-82cf-9f63ff8e609a
# ╟─654ad80a-a802-49d5-8373-b0b4056fd8f1
# ╟─67944dbb-0ebb-44d9-b6f3-79e8d5610f61
# ╟─ab745b36-164c-4534-be1f-a703a2010f3e
# ╠═f4368e0e-dd79-4a80-bae0-84a09de65e97
# ╟─9b848612-298c-4893-b8d5-15e7bf84ff24
# ╟─8798b46c-1b0b-4d34-93ef-1d5062f8a632
# ╟─623ed8e1-7f9e-4092-bf39-b9a3d90a7d23
# ╟─5d3bfc57-6a5a-41e5-8f63-6902e4958936
# ╟─6e3ef066-d115-11eb-2338-013a707dfe8a
# ╟─41bd06a2-e7bd-46c6-9249-6e69223b0e11
# ╟─94185c13-b6d2-4337-b5ce-336f5e128032
# ╟─0f54629c-e045-47bf-90a5-36e44c05b8f0
# ╟─bbed8889-d401-4167-9370-9927aabbc83b
