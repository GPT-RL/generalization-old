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

# ╔═╡ ab24c6bd-2721-4162-a8c2-fd7aabab1155
md"##"

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
							# This is where the compatibility with old gpt stuff starts:
							"action loss",
							"alpha",
							"clip_param",
							"cuda",
							"linguistic_analysis_path",
							"entropy",
							"eps",
							"eval_interval",
							"gae_lambda", 
							"gpt_size", 
							"linear_lr_decay", 
							"lr", 
							"max_grad_norm",
							"num_env_steps",
							"num_mini_batch",
							"num_processes",
							"ppo_epoch",
							"recurrent_policy",
							"use_proper_time_limits",
							"value loss",
							"value_coef",
							"async_envs",
							"charts_path",
							"episode length",
							"epsilon",
							"jit",
							"lambda_",
							"learning_rate",
							"logger",
							"loss",
							"num_envs",
							"resnet",
							"rnn",
							"training_steps",
							"graphql_endpoint", 
							"host_machine",
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

# ╔═╡ 67944dbb-0ebb-44d9-b6f3-79e8d5610f61
begin
	Gadfly.with_theme(gadfly_theme) do
		plot(
			sweep_runs([672, 687], 50000000),
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
			sweep_runs([690, 787], 50000000),
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

# ╔═╡ 4e37c894-cdd4-48b2-a25a-cf7cda2d6d7b
sweeps = sweep_runs([784], 10000000) ;

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

# ╔═╡ bbed8889-d401-4167-9370-9927aabbc83b
# @bind window_size Select(string.(collapse_runs ? (100:100:500) : (1:10)), default=collapse_runs ? "200" : "5")
window_size = "11";

# ╔═╡ Cell order:
# ╟─0da5a4ba-6b23-4f23-8165-1bd6183abbc1
# ╟─ba990146-058e-440e-b172-4d35b2b63985
# ╟─087ee9a9-e5e6-4607-a547-824d2fea470e
# ╟─07b1a576-f7a1-404d-9043-2827da1f4d07
# ╟─03f1dfc7-970e-4920-9df9-79dd9f048e65
# ╟─0be25163-f9c1-41bc-82cf-9f63ff8e609a
# ╟─654ad80a-a802-49d5-8373-b0b4056fd8f1
# ╟─67944dbb-0ebb-44d9-b6f3-79e8d5610f61
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
# ╟─8798b46c-1b0b-4d34-93ef-1d5062f8a632
# ╠═33909132-16c2-4eb3-97d3-607010a82e52
# ╟─16ce1ff4-3a0e-48e5-ae84-30afb618804c
# ╟─623ed8e1-7f9e-4092-bf39-b9a3d90a7d23
# ╟─5d3bfc57-6a5a-41e5-8f63-6902e4958936
# ╟─6e3ef066-d115-11eb-2338-013a707dfe8a
# ╟─9af550be-4ef4-4938-a710-41f96392b05a
# ╟─41bd06a2-e7bd-46c6-9249-6e69223b0e11
# ╟─94185c13-b6d2-4337-b5ce-336f5e128032
# ╟─ce12c840-ece3-48bd-8696-40c1f9802695
# ╟─c5b77c36-436f-4682-8d52-945ae25fb47c
# ╟─0f54629c-e045-47bf-90a5-36e44c05b8f0
# ╟─bbed8889-d401-4167-9370-9927aabbc83b
