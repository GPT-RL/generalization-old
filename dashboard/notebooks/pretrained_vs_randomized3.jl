### A Pluto.jl notebook ###
# v0.15.1

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
	using MLStyle
	Gadfly.set_default_plot_size(16cm, 14cm)
end

# ╔═╡ 0da5a4ba-6b23-4f23-8165-1bd6183abbc1
html"<button onclick='present()'>Present</button>"

# ╔═╡ ba990146-058e-440e-b172-4d35b2b63985
md"""
## Looking Back

### Updates from last time
  - Compared performance on current source code to previous source code
  - Showed graphs colored by `num_embeddings` parameter but results were inconclusive because parameters were randomly sampled.

### Goals set last time
- Compare pretrained GPT to GPT with randomized parameters
- Choose best hyperparameters (one set across all games for each `num_embeddings`).
- Compare `num_embeddings` runs using single best parameter-set per `num_embeddings` value.
- Lingustic Analysis: We need to re-run to record the necessary data. Any particular runs we should prioritize?
- Generalization-Focused Tasks ([BabyAI](https://github.com/mila-iqia/babyai), [ProcGen](https://openai.com/blog/procgen-benchmark/))
  - Finish implementation of few-shot learning algorithm
  - Implement learning-from-tokens algorithm
"""

# ╔═╡ 087ee9a9-e5e6-4607-a547-824d2fea470e
md"""
However, if we extend both lines, we see that the actual performance curves are nearly identcal, measured against time.

"""

# ╔═╡ 07b1a576-f7a1-404d-9043-2827da1f4d07
md"""
Furthermore, PPO learns much more quickly than A2C when performance is measured against time steps.
"""

# ╔═╡ 0be25163-f9c1-41bc-82cf-9f63ff8e609a
md"## Compare GPT with randomly initialized GPT"

# ╔═╡ 654ad80a-a802-49d5-8373-b0b4056fd8f1
md"""
The following graph compares the performance of our GPT architecture with and without randomized (but still frozen) GPT parameters. The randomized parameter architecture is still running, so results are currently inconclusive.
"""

# ╔═╡ 0f54629c-e045-47bf-90a5-36e44c05b8f0
function sweep_runs(sweep_ids::AbstractVector{Int}, max_step::Int)
	set_default_plot_size(18cm, 15cm)

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
				[k => v for (k1, v1, k2, v2) in [
							(
								"hours", get(d, "time-delta", 0) / 3600, 
								"time-delta", get(d, "hours", 0) * 3600,
							),
							(
								"env", get(d, "env_name", nothing),
								"env_name", get(d, "env", nothing),
							)
						] 
						for (k, v) in [
								(k1, get(d, k1, v1)), 
								(k2, get(d, k2, v2)),
								]]...,				
				[name => get(d, name, false) for name in [
							"randomize_parameters"
						]]...,
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
							"save_path",
							"config",

							"graphql_endpoint", 
							"linguistic_analysis_path",
							"hidden_size", 
							"host_machine", 
							"kernel",
							"stride"
						]]... 
				), _)
		collect
	end
	vcat(DataFrame.(rows)...)
	
end;

# ╔═╡ 03f1dfc7-970e-4920-9df9-79dd9f048e65
function a2c_vs_ppo(a2c_steps, ppo_steps)
	a2c = sweep_runs([85], a2c_steps * 4)
	time_min = minimum(a2c[!, :time])
	seconds =  (a2c[!, :time] .- time_min) ./ 1000000
	hours =  seconds ./ 3600
	a2c[!, :hours] = hours
	a2c[!, :step] /= 4
	insertcols!(a2c, :algorithm=>"A2C")
	ppo = sweep_runs([672], ppo_steps)
	insertcols!(ppo, :algorithm=>"PPO")
	vcat(a2c, ppo)
end;

# ╔═╡ 942d8c47-bb2f-410c-aeea-cb2e0190cfcf
md"""
| ` num_embeddings` | pretrained | random |
|-------------------|------------|--------|
| Breakout          | 8          | 2      |
| Beamrider         | 4          | 4      |
"""

# ╔═╡ ab745b36-164c-4534-be1f-a703a2010f3e
md"""
## Choose best hyperparameters (one set across all games for each num_embeddings)
Currently the parameter sweep is still in progress. I am performing grid-search on the following:
-  `env`:
    - Seaquest-v0
    - BeamRider-v0
    - BreakoutNoFrameskip-v0
-  `gpt_size`: medium
-  `ppo_epoch`  (number of PPO updates per rollout): 3
-  `clip_param` (PPO clip parameter):
    - 0.1
    - 0.2
    - 0.3
-  `num_env_steps` (cut-off for training in steps): 10000000
-  `num_processes` (number of parallel processes):
    - 8
    - 16
-  `lr` (learning-rate):
    - 0.00025
    - 0.0003
-  `num_embeddings` (number of input embeddings to GPT):
    - 1
    - 2
    - 4
    - 8
Grid search is not complete, but there does seem to already be some correlation between performance and `num_embeddings`.
"""

# ╔═╡ 43695fbb-f8f2-4c05-a149-7ae545b2c738
md"##"

# ╔═╡ a0b13a5c-0337-43ba-89b1-e277954d4d94
md"##"

# ╔═╡ 7d96dcb9-3d63-4d14-b092-411109a7fac0
md"##"

# ╔═╡ 4e37c894-cdd4-48b2-a25a-cf7cda2d6d7b
sweeps = sweep_runs([784], 10000000) ;

# ╔═╡ ce675911-0344-43b6-8a8d-6750ddbb1924
max_returns = Dict(
	"BeamRider-v0" => 1590, 
	"PongNoFrameskip-v0" => 20.7, 
	"BreakoutNoFrameskip-v0" => 274.8, 
	"Seaquest-v0" => 1204.5,
	"Qbert-v0" => 14293.3
) ;# from PPO paper

# ╔═╡ 42b94309-0cf6-4220-9c91-1fa9ba1b37af
EPISODE_RETURN = "episode return";

# ╔═╡ df070706-0482-4804-84cd-03f9655dac88
min_returns = @chain sweeps begin
	dropmissing(_, EPISODE_RETURN)
	groupby(_, [:env])
	combine(_, EPISODE_RETURN => minimum)
	Dict(k=>v for (k,v) in eachrow(_))
end;

# ╔═╡ 132deac0-130a-4bab-8ae9-460f5a66a776
dframe = @chain sweeps begin
	filter(:step => >=(8000000), _)
	groupby(_, [:env])
	transform(_, ["env", EPISODE_RETURN] =>
		function (envs, ret) 
			@match [Set(envs)...] (
				[env] => (ret .- min_returns[env]) ./ max_returns[env]
			)
		end => [EPISODE_RETURN])
	_[!, filter(names(_)) do name
			!(name in [
				"action loss",
				"config",
				"cuda",
				"entropy",
				"env",
				"eval_interval",
				"fps",
				"gradient norm",
				"hours",
				"log_interval",
				"log_level",
				"num_env_steps",
				"recurrent_policy",
				"run_id",
				"save_interval",
				"save_path",
				"step",
				"subcommand",
				"sweep_id",
				"time", 
				"time-delta",
				"value loss",
			])
		end]
end;

# ╔═╡ cadd8b9d-791d-41d6-9f6c-b549d3bbd45f
function filter_by_type(ty) 
	filter(name -> eltype(dframe[:, name]) == ty, names(dframe))
end;

# ╔═╡ b0d1c3b7-3cd6-4cd6-81b5-ad09f3d5cd10
df = @chain dframe begin
	groupby(_, "run ID")
	combine(_, 
		filter_by_type(Float64) .=> first,
		filter_by_type(Int64) .=> first,
		filter_by_type(Bool) .=> first,
		EPISODE_RETURN .=> mean .=> :episode_return_mean,
		"env_name" .=> first .=> :env,
		)
	_[!, filter(n -> !(n in  ["run ID", "episode return_first"]), names(_))]
	sort!(_, [:episode_return_mean], rev=true)
	rename(name -> replace(name, "_first" => ""), _)
end;

# ╔═╡ 2a1fcf4d-19bb-40d9-90d1-62155843b853
begin
	bool_df = DataFrame()
	for name in names(df)
		if !occursin("episode_return", name)
			for value in df[:, name]
				new_name = string(replace(name, "_first"=>""), " = ", value)
				bool_df[:, :($new_name)] = df[:, name] .== value
			end
		end
	end
    bool_df.episode_return_mean = df.episode_return_mean
	bool_df
end;

# ╔═╡ 98c0fea9-f348-4e30-81c5-4e2f1e5e239e
function get_cor_df(env)
	
	filtered = @chain df begin
		filter(row -> row.env == env, _)
		_[!, filter(name -> name != "env", names(_))]
	end
	
	bool_df = DataFrame()
	for name in names(df)
		if !occursin("episode_return", name)
			for value in df[:, name]
				new_name = string(replace(name, "_first"=>""), " = ", value)
				bool_df[:, :($new_name)] = df[:, name] .== value
			end
		end
	end
    bool_df.episode_return_mean = df.episode_return_mean
	
	cor_mat = cor(Matrix(bool_df))
	return_index = findfirst(name -> name == "episode_return_mean", names(bool_df))
	
	correlation = cor_mat[:, return_index]
	
	name = names(bool_df)
	mapping = Dict(zip(name, correlation))
	[
		mapping["num_embeddings = 1"], 
		mapping["num_embeddings = 2"],
		mapping["num_embeddings = 4"],
		mapping["num_embeddings = 8"],
	]
end;

# ╔═╡ ab24c6bd-2721-4162-a8c2-fd7aabab1155
md"##"

# ╔═╡ 8bf7862f-9be4-44de-9d14-b8056d9d27c2
cor_df = vcat([
		DataFrame(
			num_embeddings=[1, 2, 4, 8],
			env=repeat([env], 4),
			correlation=get_cor_df(env)
			) for env in [
				"Seaquest-v0",
				"BreakoutNoFrameskip-v4", 
				"BeamRider-v0",
			]
		]...)

# ╔═╡ 5a968482-35cc-4afa-b658-0ab130a47508
plot(cor_df, x=:num_embeddings, y=:correlation, color=:env, Geom.point, Geom.line)

# ╔═╡ 3838787a-f138-45ec-9a58-7879aeab9b99
md"""
## Linguistic Analysis
We performed analysis on the best BeamRider run using GPT-2 `medium`, which can be viewed [here](https://colab.research.google.com/drive/1heCWKd8oyOSMLtaVxpezIaSQZ9ODnyeP#scrollTo=nSrYCATp_oKa).

A high level summary of that we found:
 - a
"""

# ╔═╡ 8798b46c-1b0b-4d34-93ef-1d5062f8a632
md"""
## Architectures
This report compares the performance of a standard PPO baseline against a "GPT-in-the-middle" variant. The architectures are as follows:

#### PPO baseline

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
- **Linear layer with output-size = $n \times e$ where $n$ is the number of embeddings (a hyperparameter) and $e$ is the GPT embedding size (see table below).**
- **GPT torso**
- **one `hidden_size` layer**
- one layer projecting to action and value

| Architecture | parameters | embedding size ($e$) |
|--------------|------------|----------------|
| small        | 117M       | 768            |
| medium       | 345M       | 1024           |
| large        | 774M       | 1280           |
| xl           | 1558M      | 1600           |
"""

# ╔═╡ 33909132-16c2-4eb3-97d3-607010a82e52
md"""
## Next Steps
- How to choose best hyperparameters (one set across all games for each `num_embeddings`).
- Graph comparing the speed of different GPT sizes (e.g. run 1 million samples for each size and see how long it takes).
- "Time Steps" -> "Steps"
- Display interesting hyperparameters for the runs below each graph (e.g. how many embs for the first few graphs in this notebook).
"""

# ╔═╡ 623ed8e1-7f9e-4092-bf39-b9a3d90a7d23
# Colormap Gallery: https://colorcet.com/gallery.html
colors(n) = cmap("I1"; N=n);

# ╔═╡ 16ce1ff4-3a0e-48e5-ae84-30afb618804c
begin
	num_embeddings = [1,2,4,8]
	color_scale = Scale.color_discrete_hue(colors, levels=reverse(num_embeddings))
end;

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

# ╔═╡ 9af550be-4ef4-4938-a710-41f96392b05a


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

# ╔═╡ ce12c840-ece3-48bd-8696-40c1f9802695
for (x,y) in [1=>2] 
	x
end

# ╔═╡ c5b77c36-436f-4682-8d52-945ae25fb47c
gadfly_theme = :default;

# ╔═╡ 67944dbb-0ebb-44d9-b6f3-79e8d5610f61
begin
	Gadfly.with_theme(gadfly_theme) do
		plot(
			sweep_runs([817], 50000000),
			x=:step, y="episode return",
			yintercept=[274.8],
			group=:run_id,
			color=:randomize_parameters,
			Guide.xlabel("Step"),
			Guide.ylabel("Episode Return"),
			Geom.line,
			Geom.hline(style=:dot,color="#b88fff"),
			Scale.color_discrete(colors),
			Guide.colorkey(title="Random Parameters"),
			Guide.title("Breakout"),
			alpha=[0.5]
		) |> HTMLDocument
	end
end

# ╔═╡ de9db738-822c-4bd6-8516-a35bd7369929
begin
	Gadfly.with_theme(gadfly_theme) do
		plot(
			sweep_runs([799, 690, 810], 50000000),
			x=:step, y="episode return",
			yintercept=[274.8],
			group=:run_id,
			color=:sweep_id,
			Guide.xlabel("Step"),
			Guide.ylabel("Episode Return"),
			Geom.line,
			Geom.hline(style=:dot,color="#b88fff"),
			Scale.color_discrete(colors),
			Guide.colorkey(title="Random Parameters"),
			Guide.title("BeamRider"),
			alpha=[0.5]
		) |> HTMLDocument
	end
end

# ╔═╡ c85805a8-725d-4a84-8e8a-e1f3f1ddeae5
begin
	Gadfly.with_theme(gadfly_theme) do
		plot(
			filter(sweep_runs([784], 10000000)) do row
				row.env == "Seaquest-v0"
			end,
			x=:step, y="episode return",
			yintercept=[1204.5],
			group=:run_id,
			color="num_embeddings",
			Guide.xlabel("Step"),
			Guide.ylabel("Episode Return"),
			Geom.line,
			Geom.hline(style=:dot,color="purple"),
			color_scale,
			Guide.colorkey(title="#embeddings"),
			Guide.title("Seaquest"),
			alpha=[0.5]
		) |> HTMLDocument
	end
end

# ╔═╡ e8572b0c-6556-4c22-aa0f-cddbbda8b954
begin
	Gadfly.with_theme(gadfly_theme) do
		plot(
			filter(sweep_runs([784], 10000000)) do row
				row.env == "BeamRider-v0"
			end,
			x=:step, y="episode return",
			yintercept=[1590],
			group=:run_id,
			color="num_embeddings",
			Guide.xlabel("Step"),
			Guide.ylabel("Episode Return"),
			Geom.line,
			Geom.hline(style=:dot,color="purple"),
			color_scale,
			Guide.colorkey(title="#embeddings"),
			Guide.title("BeamRider"),
			alpha=[0.5]
		) |> HTMLDocument
	end
end

# ╔═╡ acdc2e20-8bb2-41f7-95db-973fd969490b
begin
	Gadfly.with_theme(gadfly_theme) do
		plot(
			filter(sweep_runs([784], 10000000)) do row
				row.env == "BreakoutNoFrameskip-v0"
			end,
			x=:step, y="episode return",
			yintercept=[274.8],
			group=:run_id,
			color="num_embeddings",
			Guide.xlabel("Step"),
			Guide.ylabel("Episode Return"),
			Geom.line,
			Geom.hline(style=:dot,color="purple"),
			color_scale,
			Guide.colorkey(title="#embeddings"),
			Guide.title("Breakout"),
			alpha=[0.5]
		) |> HTMLDocument
	end
end

# ╔═╡ bbed8889-d401-4167-9370-9927aabbc83b
# @bind window_size Select(string.(collapse_runs ? (100:100:500) : (1:10)), default=collapse_runs ? "200" : "5")
window_size = "11";

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Chain = "8be319e6-bccf-4806-a6f7-6fae938471bc"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Gadfly = "c91e804a-d5a3-530f-b6f0-dfbca275c004"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
MLStyle = "d8e11817-5142-5d16-987a-aa16d5891078"
PerceptualColourMaps = "54e51dfa-9dd7-5231-aa84-a4037b83483a"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
RollingFunctions = "b0e4dd01-7b14-53d8-9b45-175a3e362653"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[compat]
Chain = "~0.4.7"
DataFrames = "~1.2.2"
Gadfly = "~1.3.3"
HTTP = "~0.9.13"
JSON = "~0.21.2"
MLStyle = "~0.4.10"
PerceptualColourMaps = "~0.3.5"
PlutoUI = "~0.7.9"
RollingFunctions = "~0.6.2"
Tables = "~1.5.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "84918055d15b3114ede17ac6a7182f68870c16f7"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.1"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArrayInterface]]
deps = ["IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "2e004e61f76874d153979effc832ae53b56c20ee"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.1.22"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "a4d07a1c313392a77042855df46c5f534076fab9"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.0"

[[AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "d127d5e4d86c7680b20c35d40b503c74b9a39b5e"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.4"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[CEnum]]
git-tree-sha1 = "215a9aa4a1f23fbd05b92769fdd62559488d70e9"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.4.1"

[[CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[CategoricalArrays]]
deps = ["DataAPI", "Future", "JSON", "Missings", "Printf", "RecipesBase", "Statistics", "StructTypes", "Unicode"]
git-tree-sha1 = "1562002780515d2573a4fb0c3715e4e57481075e"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.0"

[[Chain]]
git-tree-sha1 = "c72673739e02d65990e5e068264df5afaa0b3273"
uuid = "8be319e6-bccf-4806-a6f7-6fae938471bc"
version = "0.4.7"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f53ca8d41e4753c41cdafa6ec5f7ce914b34be54"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "0.10.13"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "024fe24d83e4a5bf5fc80501a314ce0d1aa35597"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.0"

[[ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "42a9b08d3f2f951c9b283ea427d96ed9f1f30343"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.5"

[[Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "417b0ed7b8b838aa6ca0a87aadf1bb9eb111ce40"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.8"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "344f143fa0ec67e47917848795ab19c6a455f32c"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.32.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[Compose]]
deps = ["Base64", "Colors", "DataStructures", "Dates", "IterTools", "JSON", "LinearAlgebra", "Measures", "Printf", "Random", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "c6461fc7c35a4bb8d00905df7adafcff1fe3a6bc"
uuid = "a81c6b42-2e10-5240-aca2-a61377ecd94b"
version = "0.9.2"

[[ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[Conda]]
deps = ["JSON", "VersionParsing"]
git-tree-sha1 = "299304989a5e6473d985212c28928899c74e9421"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.5.2"

[[Contour]]
deps = ["StaticArrays"]
git-tree-sha1 = "9f02045d934dc030edad45944ea80dbd1f0ebea7"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.5.7"

[[CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "6d1c23e740a586955645500bbec662476204a52c"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.1"

[[CoupledFields]]
deps = ["LinearAlgebra", "Statistics", "StatsBase"]
git-tree-sha1 = "6c9671364c68c1158ac2524ac881536195b7e7bc"
uuid = "7ad07ef1-bdf2-5661-9d2b-286fd4296dac"
version = "0.2.0"

[[Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[CustomUnitRanges]]
git-tree-sha1 = "537c988076d001469093945f3bd0b300b8d3a7f3"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.1"

[[DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d785f42445b63fc86caa08bb9a9351008be9b765"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.2.2"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4437b64df1e0adccc3e5d1adbc3ac741095e4677"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.9"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "abe4ad222b26af3337262b8afb28fab8d215e9f8"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.3"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "a837fdf80f333415b69684ba8e8ae6ba76de6aaa"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.24.18"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "8041575f021cba5a099a456b4163c9a08b566a02"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.1.0"

[[FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "70a0cfd9b1c86b0209e38fbfe6d8231fd606eeaf"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.1"

[[FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "f985af3b9f4e278b1d24434cbb546d6092fca661"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.4.3"

[[FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3676abafff7e4ff07bbd2c42b3d8201f31653dcc"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.9+8"

[[FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "256d8e6188f3f1ebfa1a5d17e072a0efafa8c5bf"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.10.1"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "693210145367e7685d8604aee33d9bfb85db8b31"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.11.9"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[Gadfly]]
deps = ["Base64", "CategoricalArrays", "Colors", "Compose", "Contour", "CoupledFields", "DataAPI", "DataStructures", "Dates", "Distributions", "DocStringExtensions", "Hexagons", "IndirectArrays", "IterTools", "JSON", "Juno", "KernelDensity", "LinearAlgebra", "Loess", "Measures", "Printf", "REPL", "Random", "Requires", "Showoff", "Statistics"]
git-tree-sha1 = "96da4818e4d481a29aa7d66aac1eb778432fb89a"
uuid = "c91e804a-d5a3-530f-b6f0-dfbca275c004"
version = "1.3.3"

[[Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "2c1cf4df419938ece72de17f368a021ee162762e"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.0"

[[Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "44e3b40da000eab4ccb1aecdc4801c040026aeb5"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.13"

[[Hexagons]]
deps = ["Test"]
git-tree-sha1 = "de4a6f9e7c4710ced6838ca906f81905f7385fd6"
uuid = "a1b4810d-1bce-5fbd-ac56-80944d57a21f"
version = "0.2.0"

[[IdentityRanges]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be8fcd695c4da16a1d6d0cd213cb88090a150e3b"
uuid = "bbac6d45-d8f3-5730-bfe4-7a449cd117ca"
version = "0.3.1"

[[IfElse]]
git-tree-sha1 = "28e837ff3e7a6c3cdb252ce49fb412c8eb3caeef"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.0"

[[ImageAxes]]
deps = ["AxisArrays", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "794ad1d922c432082bc1aaa9fa8ffbd1fe74e621"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.9"

[[ImageContrastAdjustment]]
deps = ["ColorVectorSpace", "ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "2e6084db6cccab11fe0bc3e4130bd3d117092ed9"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.7"

[[ImageCore]]
deps = ["AbstractFFTs", "Colors", "FixedPointNumbers", "Graphics", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "Reexport"]
git-tree-sha1 = "db645f20b59f060d8cfae696bc9538d13fd86416"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.8.22"

[[ImageDistances]]
deps = ["ColorVectorSpace", "Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "6378c34a3c3a216235210d19b9f495ecfff2f85f"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.13"

[[ImageFiltering]]
deps = ["CatIndices", "ColorVectorSpace", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageCore", "LinearAlgebra", "OffsetArrays", "Requires", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "bf96839133212d3eff4a1c3a80c57abc7cfbf0ce"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.6.21"

[[ImageIO]]
deps = ["FileIO", "Netpbm", "PNGFiles", "TiffImages", "UUIDs"]
git-tree-sha1 = "d067570b4d4870a942b19d9ceacaea4fb39b69a1"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.5.6"

[[ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils", "Libdl", "Pkg", "Random"]
git-tree-sha1 = "5bc1cb62e0c5f1005868358db0692c994c3a13c6"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.2.1"

[[ImageMagick_jll]]
deps = ["JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1c0a2295cca535fabaf2029062912591e9b61987"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "6.9.10-12+3"

[[ImageMetadata]]
deps = ["AxisArrays", "ColorVectorSpace", "ImageAxes", "ImageCore", "IndirectArrays"]
git-tree-sha1 = "ae76038347dc4edcdb06b541595268fca65b6a42"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.5"

[[ImageMorphology]]
deps = ["ColorVectorSpace", "ImageCore", "LinearAlgebra", "TiledIteration"]
git-tree-sha1 = "68e7cbcd7dfaa3c2f74b0a8ab3066f5de8f2b71d"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.2.11"

[[ImageQualityIndexes]]
deps = ["ColorVectorSpace", "ImageCore", "ImageDistances", "ImageFiltering", "OffsetArrays", "Statistics"]
git-tree-sha1 = "1198f85fa2481a3bb94bf937495ba1916f12b533"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.2.2"

[[ImageShow]]
deps = ["Base64", "FileIO", "ImageCore", "OffsetArrays", "Requires", "StackViews"]
git-tree-sha1 = "832abfd709fa436a562db47fd8e81377f72b01f9"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.1"

[[ImageTransformations]]
deps = ["AxisAlgorithms", "ColorVectorSpace", "CoordinateTransformations", "IdentityRanges", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "e4cc551e4295a5c96545bb3083058c24b78d4cf0"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.8.13"

[[Images]]
deps = ["AxisArrays", "Base64", "ColorVectorSpace", "FileIO", "Graphics", "ImageAxes", "ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageShow", "ImageTransformations", "IndirectArrays", "OffsetArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "8b714d5e11c91a0d945717430ec20f9251af4bd2"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.24.1"

[[IndirectArrays]]
git-tree-sha1 = "c2a145a145dc03a7620af1444e0264ef907bd44f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "0.5.1"

[[Inflate]]
git-tree-sha1 = "f5fc07d4e706b84f72d54eedcc1c13d92fb0871c"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.2"

[[IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d979e54b71da82f3a65b62553da4fc3d18c9004c"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2018.0.3+2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[Interpolations]]
deps = ["AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "1470c80592cf1f0a35566ee5e93c5f8221ebc33a"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.13.3"

[[IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "3cc368af3f110a767ac786560045dceddfc16758"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.3"

[[InvertedIndices]]
deps = ["Test"]
git-tree-sha1 = "15732c475062348b0165684ffe28e85ea8396afc"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.0.0"

[[IterTools]]
git-tree-sha1 = "05110a2ab1fc5f932622ffea2a003221f4782c18"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.3.0"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "d735490ac75c5cb9f1b00d8b5509c11984dc6943"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "2.1.0+0"

[[Juno]]
deps = ["Base64", "Logging", "Media", "Profile"]
git-tree-sha1 = "07cb43290a840908a771552911a6274bc6c072c7"
uuid = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
version = "0.8.4"

[[KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "591e8dc09ad18386189610acafb970032c519707"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.3"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "340e257aada13f95f98ee352d316c3bed37c8ab9"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.3.0+0"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[Loess]]
deps = ["Distances", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "b5254a86cf65944c68ed938e575f5c81d5dfe4cb"
uuid = "4345ca2d-374a-55d4-8d30-97f9976e7612"
version = "0.5.3"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "LinearAlgebra"]
git-tree-sha1 = "7bd5f6565d80b6bf753738d2bc40a5dfea072070"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.2.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg"]
git-tree-sha1 = "c253236b0ed414624b083e6b72bfe891fbd2c7af"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2021.1.1+1"

[[MLStyle]]
git-tree-sha1 = "594e189325f66e23a8818e5beb11c43bb0141bcd"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.10"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "0fb723cd8c45858c22169b2e42269e53271a6df7"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.7"

[[MappedArrays]]
git-tree-sha1 = "18d3584eebc861e311a552cbb67723af8edff5de"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.0"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Measures]]
git-tree-sha1 = "e498ddeee6f9fdb4551ce855a46f54dbd900245f"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.1"

[[Media]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "75a54abd10709c01f1b86b84ec225d26e840ed58"
uuid = "e89f7d12-3494-54d1-8411-f7d8b9ae1f27"
version = "0.5.0"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "4ea90bd5d3985ae1f9a908bd4500ae88921c5ce7"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.0"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "b34e3bc3ca7c94914418637cb10cc4d1d80d877d"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.3"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[Netpbm]]
deps = ["ColorVectorSpace", "FileIO", "ImageCore"]
git-tree-sha1 = "09589171688f0039f13ebe0fdcc7288f50228b52"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.0.1"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "5cc97a6f806ba1b36bac7078b866d4297ae8c463"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.4"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "520e28d4026d16dcf7b8c8140a3041f0e20a9ca8"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.3.7"

[[PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "59925f4ae6861cddc2313a47514b93b6740f9b6f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.9"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "2276ac65f1e236e0a6ea70baff3f62ad4c625345"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.2"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "477bf42b4d1496b454c10cce46645bb5b8a0cf2c"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.0.2"

[[PerceptualColourMaps]]
deps = ["Colors", "Images", "Interpolations", "LinearAlgebra", "Printf", "PyPlot", "Statistics"]
git-tree-sha1 = "1a590b17c9a2fc3933c23587330fb39b85e82e10"
uuid = "54e51dfa-9dd7-5231-aa84-a4037b83483a"
version = "0.3.5"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "a7a7e1a88853564e551e4eba8650f8c38df79b37"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.1.1"

[[PlutoUI]]
deps = ["Base64", "Dates", "InteractiveUtils", "JSON", "Logging", "Markdown", "Random", "Reexport", "Suppressor"]
git-tree-sha1 = "44e225d5837e2a2345e69a1d1e01ac2443ff9fcb"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.9"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "cde4ce9d6f33219465b55162811d8de8139c0414"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.2.1"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "0d1245a357cc61c8cd61934c07447aa569ff22e6"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.1.0"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "169bb8ea6b1b143c5cf57df6d34d022a7b60c6db"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.92.3"

[[PyPlot]]
deps = ["Colors", "LaTeXStrings", "PyCall", "Sockets", "Test", "VersionParsing"]
git-tree-sha1 = "67dde2482fe1a72ef62ed93f8c239f947638e5a2"
uuid = "d330b81b-6aea-500a-939a-2ce795aea3ee"
version = "2.9.0"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "12fbe86da16df6679be7521dfb39fbc861e1dc7b"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.1"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[Ratios]]
git-tree-sha1 = "37d210f612d70f3f7d57d488cb3b6eff56ad4e41"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.0"

[[RecipesBase]]
git-tree-sha1 = "b3fb709f3c97bfc6e948be68beeecb55a0b340ae"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.1"

[[Reexport]]
git-tree-sha1 = "5f6c21241f0f655da3952fd60aa18477cf96c220"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.1.0"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[RollingFunctions]]
deps = ["LinearAlgebra", "Statistics", "StatsBase", "Test"]
git-tree-sha1 = "cdf9158377f81470b1b73c630d0853a3ec0c7445"
uuid = "b0e4dd01-7b14-53d8-9b45-175a3e362653"
version = "0.6.2"

[[Rotations]]
deps = ["LinearAlgebra", "StaticArrays", "Statistics"]
git-tree-sha1 = "2ed8d8a16d703f900168822d83699b8c3c1a5cd8"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.0.2"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "508822dca004bf62e210609148511ad03ce8f1d8"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.6.0"

[[StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "62701892d172a2fa41a1f829f66d2b0db94a9a63"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.3.0"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "3fedeffc02e47d6e3eb479150c8e5cd8f15a77a0"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.10"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "fed1ec1e65749c4d96fc20dd13bea72b55457e62"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.9"

[[StatsFuns]]
deps = ["LogExpFunctions", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "30cd8c360c54081f806b1ee14d2eecbef3c04c49"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.8"

[[StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "e36adc471280e8b346ea24c5c87ba0571204be7a"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.7.2"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[Suppressor]]
git-tree-sha1 = "a819d77f31f83e5792a76081eee1ea6342ab8787"
uuid = "fd094767-a336-5f1f-9728-57cf17d0bbfb"
version = "0.2.0"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "d0c690d37c73aeb5ca063056283fde5585a41710"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.5.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TiffImages]]
deps = ["ColorTypes", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "OffsetArrays", "OrderedCollections", "PkgVersion", "ProgressMeter"]
git-tree-sha1 = "03fb246ac6e6b7cb7abac3b3302447d55b43270e"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.4.1"

[[TiledIteration]]
deps = ["OffsetArrays"]
git-tree-sha1 = "52c5f816857bfb3291c7d25420b1f4aca0a74d18"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.3.0"

[[URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[VersionParsing]]
git-tree-sha1 = "80229be1f670524750d905f8fc8148e5a8c4537f"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.2.0"

[[WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "59e2ad8fd1591ea019a5259bd012d7aee15f995c"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.3"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "94d180a6d2b5e55e447e2d27a29ed04fe79eb30c"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.38+0"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─0da5a4ba-6b23-4f23-8165-1bd6183abbc1
# ╟─ba990146-058e-440e-b172-4d35b2b63985
# ╟─087ee9a9-e5e6-4607-a547-824d2fea470e
# ╟─07b1a576-f7a1-404d-9043-2827da1f4d07
# ╟─03f1dfc7-970e-4920-9df9-79dd9f048e65
# ╟─0be25163-f9c1-41bc-82cf-9f63ff8e609a
# ╟─654ad80a-a802-49d5-8373-b0b4056fd8f1
# ╠═67944dbb-0ebb-44d9-b6f3-79e8d5610f61
# ╠═0f54629c-e045-47bf-90a5-36e44c05b8f0
# ╠═de9db738-822c-4bd6-8516-a35bd7369929
# ╟─942d8c47-bb2f-410c-aeea-cb2e0190cfcf
# ╟─ab745b36-164c-4534-be1f-a703a2010f3e
# ╟─43695fbb-f8f2-4c05-a149-7ae545b2c738
# ╠═c85805a8-725d-4a84-8e8a-e1f3f1ddeae5
# ╟─a0b13a5c-0337-43ba-89b1-e277954d4d94
# ╟─e8572b0c-6556-4c22-aa0f-cddbbda8b954
# ╟─7d96dcb9-3d63-4d14-b092-411109a7fac0
# ╟─acdc2e20-8bb2-41f7-95db-973fd969490b
# ╟─4e37c894-cdd4-48b2-a25a-cf7cda2d6d7b
# ╟─df070706-0482-4804-84cd-03f9655dac88
# ╟─ce675911-0344-43b6-8a8d-6750ddbb1924
# ╟─cadd8b9d-791d-41d6-9f6c-b549d3bbd45f
# ╟─42b94309-0cf6-4220-9c91-1fa9ba1b37af
# ╟─132deac0-130a-4bab-8ae9-460f5a66a776
# ╟─b0d1c3b7-3cd6-4cd6-81b5-ad09f3d5cd10
# ╟─2a1fcf4d-19bb-40d9-90d1-62155843b853
# ╟─98c0fea9-f348-4e30-81c5-4e2f1e5e239e
# ╟─ab24c6bd-2721-4162-a8c2-fd7aabab1155
# ╠═8bf7862f-9be4-44de-9d14-b8056d9d27c2
# ╠═5a968482-35cc-4afa-b658-0ab130a47508
# ╠═3838787a-f138-45ec-9a58-7879aeab9b99
# ╠═8798b46c-1b0b-4d34-93ef-1d5062f8a632
# ╠═33909132-16c2-4eb3-97d3-607010a82e52
# ╠═16ce1ff4-3a0e-48e5-ae84-30afb618804c
# ╠═623ed8e1-7f9e-4092-bf39-b9a3d90a7d23
# ╠═5d3bfc57-6a5a-41e5-8f63-6902e4958936
# ╠═6e3ef066-d115-11eb-2338-013a707dfe8a
# ╠═9af550be-4ef4-4938-a710-41f96392b05a
# ╠═41bd06a2-e7bd-46c6-9249-6e69223b0e11
# ╠═94185c13-b6d2-4337-b5ce-336f5e128032
# ╠═ce12c840-ece3-48bd-8696-40c1f9802695
# ╠═c5b77c36-436f-4682-8d52-945ae25fb47c
# ╠═bbed8889-d401-4167-9370-9927aabbc83b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
