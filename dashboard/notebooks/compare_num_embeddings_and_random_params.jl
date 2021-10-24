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

# ╔═╡ 0da5a4ba-6b23-4f23-8165-1bd6183abbc1
html"<button onclick='present()'>Present</button>"

# ╔═╡ ba990146-058e-440e-b172-4d35b2b63985
md"""
## Looking Back

### Updates from last time
- Switched codebase to PyTorch + Huggingface
  - No more stability issues
  - Smaller memory footprint (can now use more machines)
  - Comes with helper functions for linguistic analysis
- Showed comparison between PPO with/without GPT on Breakout.
  - Both versions were comparable.

### Goals set last time
- How fast is new code?
  - How long to run a standard amount of steps for Atari?
- What is the published PPO benchmark on each Atari game?
- How is performance on other Atari games?
  - How do we compare to the baseline across Atari library?
- How does performance compare when using different numbers of embeddings for perception -> GPT input.
- How does model with pretrained GPT compare to model with randomly initialized GPT?
- Linguistic analysis
  - How similar are perception module's outputs to GPT's vocabulary of token embeddings?
  - For `n_embeddings > 1`, is the sequence of embeddings grammatical according to GPT?
    - (Does the sequence of embeddings have high probability under GPT's language model?)
"""

# ╔═╡ 731054d8-e5df-45d0-bd4f-91083884f0c8
md"""## What is the published PPO benchmark on each Atari game?
We have added a purple horizontal line to our graphs indicating the 10-million-step performance reported in the [PPO paper](https://arxiv.org/pdf/1707.06347.pdf).
"""

# ╔═╡ a15380dc-fb70-42ed-8025-77ba49dfb073
md"""
## How fast is the new code?
**Assessment:** The new code (PPO) runs at about the same speed as the old code (A2C). 

The FPS for PPO is much lower (probably due to the high number of gradient updates performed per frame) but the PPO algorithm is more sample efficient.
"""

# ╔═╡ 5084f38e-b69a-4ee5-90c3-f23d1e6abad4
md"""
Here we compare the time to reach 10 million frames (the number of frames used in the PPO paper). As the graph demonstrates, A2C reaches 10 million frames in much less time.
"""

# ╔═╡ 76f3a217-3303-4952-ab6e-d546c8c05ffe
md"""
## Next Steps
- Lingustic Analysis
  - We need to re-run to record the necessary data. Any particular runs we should prioritize?
  - Correlation between particular state attributes and perception embeddings?
- Generalization-Focused Tasks ([BabyAI](https://github.com/mila-iqia/babyai), [ProcGen](https://openai.com/blog/procgen-benchmark/))
  - Finish implementation of few-shot learning algorithm
  - Implement learning-from-tokens algorithm
- Comparison with randomized parameters:
  - Search for good hyperparameters with randomized parameters
  - Compare with non-randomized on 5 Atari games
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

# ╔═╡ ab745b36-164c-4534-be1f-a703a2010f3e
md"""
## Graph comparing GPT architectures by number of embeddings
The current architecture passes each observation through a series of CNNs and then reshapes the results into $n$ input embeddings to be fed into GPT, where $n$ is a hyperparameter.

The following graphs depict hyperparameter searches on the medium (355M parameter) GPT architecture with lines colored by $n$.

**Assessment:** There is not a clear linear correlation between the number of embeddings and performance, but it is with noting that 8-embedding architectures are among the top performers for all games except Seaquest.

Note that the first graph only runs to 5 million frames which explains the poor relative performance of the GPT architectures.
"""

# ╔═╡ 9b367184-ecfa-40b1-b439-d6f517d022eb
md"""
## Other Atari games
As the preceeding graphs demonstrate, hyperparameter search is underway on four Atari games besides Breakout. Once this search is complete, we will select a single, best-performing set of parameters to run on the complete Atari suite. 
"""

# ╔═╡ 3838787a-f138-45ec-9a58-7879aeab9b99
md"""
## Linguistic Analysis
We are currently rerunning the best BeamRider run, saving
- the input observation
- the perception output
- the action logits
at set intervals.

We chose BeamRider because we significantly outperformed the baseline on this task. When the run is complete we will run linguistic analysis on the saved data.
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

# ╔═╡ 5027f577-6604-40e0-a40c-17110a1bc3fa
md"""
## TODO:
"""

# ╔═╡ 623ed8e1-7f9e-4092-bf39-b9a3d90a7d23
# Colormap Gallery: https://colorcet.com/gallery.html
colors(n) = cmap("I1"; N=n);

# ╔═╡ 16ce1ff4-3a0e-48e5-ae84-30afb618804c
begin
	num_embeddings = [1,2,4,6,8]
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

# ╔═╡ 2ab4c326-8ae0-40a3-9dc9-e5fd1cf17dfb
begin

	Gadfly.with_theme(gadfly_theme) do
		plot(
			a2c_vs_ppo(10000000, 10000000),
			x=:hours, y="episode return",
			yintercept=[274.8],
			group=:run_id,
			color="algorithm",
			Guide.xlabel("Hours"),
			Guide.ylabel("Episode Return"),
			Geom.line,
			Geom.hline(style=:dot,color="#b88fff"),
			Scale.color_discrete(colors),
			Guide.colorkey(title="Algorithm"),
			Guide.title("Breakout"),
			alpha=[0.5]
		) |> HTMLDocument
	end
end

# ╔═╡ d3352f83-5228-489a-9db6-b944fb2c1f72
begin

	Gadfly.with_theme(gadfly_theme) do
		plot(
			a2c_vs_ppo(80000000, 80000000),
			x=:hours, y="episode return",
			group=:run_id,
			color="algorithm",
			yintercept=[274.8],
			Geom.line,
			Geom.hline(style=:dot, color="#b88fff"),
			Guide.xlabel("Hours"),
			Guide.ylabel("Episode Return"),
			Scale.color_discrete(colors),
			Guide.colorkey(title="Algorithm"),
			Guide.title("Breakout"),
			alpha=[0.5]
		) |> HTMLDocument
	end
end

# ╔═╡ f05ae4d2-9b03-4013-a4d2-cb5ff3e4b61e
begin

	Gadfly.with_theme(gadfly_theme) do
		plot(
			a2c_vs_ppo(10000000, 10000000),
			x=:step, y="episode return",
			yintercept=[274.8],
			group=:run_id,
			color="algorithm",
			Guide.xlabel("Time Steps"),
			Guide.ylabel("Episode Return"),
			Geom.line,
			Geom.hline(style=:dot,color="#b88fff"),
			Scale.color_discrete(colors),
			Guide.colorkey(title="Algorithm"),
			Guide.title("Breakout"),
			alpha=[0.5]
		) |> HTMLDocument
	end
end

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

# ╔═╡ f4368e0e-dd79-4a80-bae0-84a09de65e97
begin
	Gadfly.with_theme(gadfly_theme) do
		plot(
			sweep_runs([646], 100000000),
			x=:step, y="episode return",
			yintercept=[274.8],
			group=:run_id,
			color="num_embeddings",
			Guide.xlabel("Time Step"),
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

# ╔═╡ 9000d2d6-353d-499f-ae30-0338dc85bfe9
begin
	Gadfly.with_theme(gadfly_theme) do	
		plot(
			sweep_runs([677], 50000000),
			x=:step, y="episode return",
			yintercept=[1204.5],
			group=:run_id,
			color="num_embeddings",
			Guide.xlabel("Time Step"),
			Guide.ylabel("Episode Return"),
			Geom.line,
			Geom.hline(style=:dot,color="#b88fff"),
			color_scale,
			Guide.colorkey(title="#embeddings"),
			Guide.title("Seaquest"),
			alpha=[0.5]
		) |> HTMLDocument
	end
end

# ╔═╡ 0a801028-935d-463a-b09c-01fca6eca913
begin
	Gadfly.with_theme(gadfly_theme) do
		plot(
			sweep_runs([676], 50000000),
			x=:step, y="episode return",
			yintercept=[14293.3],
			group=:run_id,
			color="num_embeddings",
			Guide.xlabel("Time Step"),
			Guide.ylabel("Episode Return"),
			Geom.line,
			Geom.hline(style=:dot,color="#b88fff"),
			color_scale,
			Guide.colorkey(title="#embeddings"),
			Guide.title("Qbert"),
			alpha=[0.5]
		) |> HTMLDocument
	end
end

# ╔═╡ 8420320c-222d-4144-9669-5f4f2a3b8b11
begin
	Gadfly.with_theme(gadfly_theme) do
	plot(
        sweep_runs([675], 50000000),
		x=:step, y="episode return",
		yintercept=[20.7],
		group=:run_id,
		color="num_embeddings",
		Guide.xlabel("Time Step"),
		Guide.ylabel("Episode Return"),
		Geom.line,
		Geom.hline(style=:dot,color="#b88fff"),
		color_scale,
		Guide.colorkey(title="#embeddings"),
		Guide.title("Pong"),
		alpha=[0.5]
	) |> HTMLDocument
	end
end

# ╔═╡ 3fd9c768-3dcc-4aa5-8671-c83ab39bd887
begin
	Gadfly.with_theme(gadfly_theme) do
		plot(
			sweep_runs([674], 50000000),
			x=:step, y="episode return",
			yintercept=[1590.0],
			group=:run_id,
			color="num_embeddings",
			Guide.xlabel("Time Step"),
			Guide.ylabel("Episode Return"),
			Geom.line,
			Geom.hline(style=:dot,color="#b88fff"),
			color_scale,
			Guide.colorkey(title="#embeddings"),
			Guide.title("BeamRider"),
			alpha=[0.5]
		) |> HTMLDocument
	end
end

# ╔═╡ bbed8889-d401-4167-9370-9927aabbc83b
# @bind window_size Select(string.(collapse_runs ? (100:100:500) : (1:10)), default=collapse_runs ? "200" : "5")
window_size = "11";

# ╔═╡ Cell order:
# ╟─0da5a4ba-6b23-4f23-8165-1bd6183abbc1
# ╟─ba990146-058e-440e-b172-4d35b2b63985
# ╠═731054d8-e5df-45d0-bd4f-91083884f0c8
# ╟─a15380dc-fb70-42ed-8025-77ba49dfb073
# ╟─5084f38e-b69a-4ee5-90c3-f23d1e6abad4
# ╟─76f3a217-3303-4952-ab6e-d546c8c05ffe
# ╟─2ab4c326-8ae0-40a3-9dc9-e5fd1cf17dfb
# ╟─087ee9a9-e5e6-4607-a547-824d2fea470e
# ╟─d3352f83-5228-489a-9db6-b944fb2c1f72
# ╟─07b1a576-f7a1-404d-9043-2827da1f4d07
# ╟─f05ae4d2-9b03-4013-a4d2-cb5ff3e4b61e
# ╟─03f1dfc7-970e-4920-9df9-79dd9f048e65
# ╟─0be25163-f9c1-41bc-82cf-9f63ff8e609a
# ╟─654ad80a-a802-49d5-8373-b0b4056fd8f1
# ╠═67944dbb-0ebb-44d9-b6f3-79e8d5610f61
# ╟─ab745b36-164c-4534-be1f-a703a2010f3e
# ╠═f4368e0e-dd79-4a80-bae0-84a09de65e97
# ╠═9000d2d6-353d-499f-ae30-0338dc85bfe9
# ╠═0a801028-935d-463a-b09c-01fca6eca913
# ╠═8420320c-222d-4144-9669-5f4f2a3b8b11
# ╠═3fd9c768-3dcc-4aa5-8671-c83ab39bd887
# ╟─9b367184-ecfa-40b1-b439-d6f517d022eb
# ╟─3838787a-f138-45ec-9a58-7879aeab9b99
# ╟─8798b46c-1b0b-4d34-93ef-1d5062f8a632
# ╠═33909132-16c2-4eb3-97d3-607010a82e52
# ╠═5027f577-6604-40e0-a40c-17110a1bc3fa
# ╠═16ce1ff4-3a0e-48e5-ae84-30afb618804c
# ╠═623ed8e1-7f9e-4092-bf39-b9a3d90a7d23
# ╠═5d3bfc57-6a5a-41e5-8f63-6902e4958936
# ╠═6e3ef066-d115-11eb-2338-013a707dfe8a
# ╟─9af550be-4ef4-4938-a710-41f96392b05a
# ╠═41bd06a2-e7bd-46c6-9249-6e69223b0e11
# ╠═94185c13-b6d2-4337-b5ce-336f5e128032
# ╠═ce12c840-ece3-48bd-8696-40c1f9802695
# ╠═c5b77c36-436f-4682-8d52-945ae25fb47c
# ╠═0f54629c-e045-47bf-90a5-36e44c05b8f0
# ╟─bbed8889-d401-4167-9370-9927aabbc83b
