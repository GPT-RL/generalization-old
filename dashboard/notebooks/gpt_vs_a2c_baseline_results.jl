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

# ╔═╡ 8798b46c-1b0b-4d34-93ef-1d5062f8a632
md"""
## Summary
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
- **XL-size GPT torso**
- Parallel value head and action head
"""

# ╔═╡ 2954c622-ec01-4b1f-8f44-a48aa5413a16
md"""
## Results
The following graphs compare the performance of these two architectures on the Atari game Breakout.

The baseline uses 8 seeds using the best parameters from a random hyperparameter search.

The GPT uses the top 4 hyperparameter sets from our hyperparameter search.
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

# ╔═╡ 5cba55d9-c7fe-4918-88b3-abbabe08c56b
function smooth_curves(df::AbstractDataFrame, y::Symbol, group_col::Symbol, window_size::Int)::DataFrame
	bounded_window(d) = max(1, min(window_size, nrow(d) ÷ 5))
	trim_window(d) = first(d, nrow(d) - mod(nrow(d), bounded_window(d)))
	@chain df begin
		dropmissing!([y, group_col])
		groupby(group_col)
		pairs
		[@chain g begin
				DataFrame
				sort!(:step)
				trim_window
				DataFrame(
					group_col => first(k),
					:step => round.(Int, rollmedian(_.step, bounded_window(_))),
					:mean => rollmean(_[:, y],  bounded_window(_)),
					:std => rollstd(_[:, y],  bounded_window(_)),
				)
				transform(
					[:mean, :std] => ((x, y) -> x - y) => [:lower],
					[:mean, :std] => ((x, y) -> x + y) => [:upper]
				)
				leftjoin(g; on=:step, makeunique=true)
			end for (k, g) in _]
		vcat(_...)
	end
end;

# ╔═╡ f7306428-2a04-43da-822e-85bed96216d8
md"""
## Analysis
The Abbeel paper ([Pretrained Transformers as Universal Computation Engines](https://arxiv.org/abs/2103.05247)) does not present state of the art performance after convergence. Instead it presents faster learning. It achieves this result by learning a very small number of parameters (relative to the baselines), forcing the networks to take advantage of information stored in the pretrained GPT network. 

In contrast, our GPT model learns just as many parameters as the baseline and therefore it is not forced to take advantage of the GPT network. Instead the GPT network simply adds noise to the learning process.
"""

# ╔═╡ 5d99bf99-8973-4c4d-b88a-1e478b1678fb
md"""
### Analysis of Action Hidden Size

We explored a wide range of parameters including action hidden sizes (by this we mean then additional perceptron layer in the action head). We found that in general, layers of size $256$ or greater seemed to performed poorly on the first 30M steps of Breakout as the following graph indicates.
"""

# ╔═╡ f35a94e5-7568-4b89-af94-6167aebd3dab
md"(Group runs with error bands?)"

# ╔═╡ 3b614ff9-663e-459e-9621-98edd8922fdc
@bind collapse_runs CheckBox()

# ╔═╡ b721ca18-e5cd-4762-9ff0-9d94294b6969
md"""
###### Lit review for Deep Learning tricks

- [STABILIZING TRANSFORMERS FOR REINFORCEMENT LEARNING](https://arxiv.org/pdf/1910.06764.pdf) offers two tricks:
  - Replacing residual layers with gates. The reason this helps is not explained.
  - Moving the layer norm before the inputs (instead of after). This is supposed to help the agent learn a Markov policy before learning a history-dependent policy. Not relevant to what we are doing.

- [Pretrained Transformers As Universal Computation Engines](https://arxiv.org/pdf/2103.05247.pdf) had some additional tricks that we should try:
  - Fine-tuning the layer norm.
  - breaking up the input into patches and feeding in sequentially.
- [WHAT MATTERS FOR ON-POLICY DEEP ACTORCRITIC METHODS? A LARGE-SCALE STUDY](https://arxiv.org/pdf/2006.05990.pdf) suggests some good general techniques to use with actor-critic algorithms:
  - Use PPO loss instead of generic actor-critic
  - Use separate networks for value and policy. Try using the GPT only for value or only for policy.
  - Try Tanh instead of ReLU
  - Try normalizing observations and GPT outputs
  - Try using GAE
  - Tune gamma
  - Switch to Adam
"""

# ╔═╡ 731ba6f9-fa51-4fa3-a2dd-7748c9ecbea3
md"""
###### To do:
- Deep learning tricks
- Try minimal parameter learning
- Try additional Atari games
- Debug memory issues (get to 8)
  - https://github.com/pytorch/pytorch/issues/3022
  - Vivek
  - Zeyu
  - Laura
- Look into performance issues
- time forward and backward pass and compare against huggingface implementation
"""

# ╔═╡ 939f57a0-cb34-471b-aa24-10e5a95f410d
md"""
###### Open questions:
- [Do Frameskip versions work?](http://rldl12.eecs.umich.edu:8081/#sweeps/49)
- [Does normalize observation work?](http://rldl12.eecs.umich.edu:8081/#sweeps/50)
- [What are the best GPT parameters for Pong?](http://rldl12.eecs.umich.edu:8081/#sweeps/52)
"""

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
function sweep_runs(sweep_ids::AbstractVector{Int})
	query = """
		query getSweepRuns(\$ids: [Int!]) {
		  sweep(where: {id: {_in: \$ids}}) {
			runs {
           		id
			    metadata
			    sweep{
				 id
			    }
        	}
		  }
		}
  	"""
	rows = @chain gql_query(query; variables=Dict("ids" => sweep_ids)) begin
		_["sweep"]
		map(values, _)
		Iterators.flatten
		Iterators.flatten
		collect
	end
	format_runs(rows)
end;

# ╔═╡ 18147da2-1945-42e4-b140-c1375f46962b
function sweep_data(sweep_runs::AbstractDataFrame; ids = nothing)::DataFrame
	s_run_ids = sweep_runs.run_id |> unique |> collect
	if !isnothing(ids)
		s_run_ids = filter(in(ids), s_run_ids)
	end
	s_df = [run_data(collect(i), sweep_runs) for i in Iterators.partition(s_run_ids, 10)]
	s_df = vcat(s_df...; cols=:union)
	dropmissing!(s_df, :step)
end;

# ╔═╡ 1f0c13a6-ae73-4aec-bee3-745610c23175
begin
	gpt_plot_group = collapse_runs ? :sweep_id : :run_id
	plot(
		# smooth_curves(
		# 	sweep_data(sweep_runs([30, 31])),
		# 	:episode_return,
		# 	gpt_plot_group,
		# 	parse(Int, window_size)
		# ),
		sweep_data(sweep_runs([30, 32])),
		x=:step, y=:episode_return,
		# ymin=:lower, ymax=:upper,
		group=gpt_plot_group,
		color=:sweep_id,
		Guide.xlabel("Step"),
		Guide.ylabel("Episode Return"),
		Geom.line,
		# Geom.ribbon,
		Scale.color_discrete(),
		Guide.colorkey(labels=["Yes", "No"], title="GPT?"),
		Guide.title("GPT vs. No GPT"),
		alpha=[0.5]
	) |> HTMLDocument
end

# ╔═╡ 0135af1a-d8c8-4027-af38-7f57bbfef282
gpt_runs = sweep_runs([26, 23, 22, 19, 17]);

# ╔═╡ 3ba73b7c-f5dc-4639-ab9c-c43cc992634f
gpt_df = sweep_data(gpt_runs);

# ╔═╡ bbed8889-d401-4167-9370-9927aabbc83b
# @bind window_size Select(string.(collapse_runs ? (100:100:500) : (1:10)), default=collapse_runs ? "200" : "5")
window_size = collapse_runs ? "500" : "11";

# ╔═╡ fe6f09ca-dc39-4d77-aae1-757c87ccd3e7
begin
	action_plot_group = collapse_runs ? :action_hidden_size : :run_id
	action_sizes = gpt_df.action_hidden_size |> unique |> sort
	action_colors(n) = cmap("L08"; N=n)
	plot(
		smooth_curves(
			gpt_df,
			:episode_return,
			action_plot_group,
			parse(Int, window_size)
		),
		x=:step, y=:mean,
		ymin=:lower, ymax=:upper,
		group=action_plot_group,
		color=:action_hidden_size,
		Guide.xlabel("Step"),
		Guide.ylabel("Episode Return"),
		Geom.line,
		Geom.ribbon,
		Scale.color_discrete(action_colors; levels=action_sizes),
		Guide.colorkey(title="Action Hidden Size"),
		Guide.title("Performance Across Action Sizes"),
		alpha=[0.5]
	) |> HTMLDocument
end

# ╔═╡ Cell order:
# ╟─8798b46c-1b0b-4d34-93ef-1d5062f8a632
# ╟─2954c622-ec01-4b1f-8f44-a48aa5413a16
# ╟─5d3bfc57-6a5a-41e5-8f63-6902e4958936
# ╟─5cba55d9-c7fe-4918-88b3-abbabe08c56b
# ╟─1f0c13a6-ae73-4aec-bee3-745610c23175
# ╟─f7306428-2a04-43da-822e-85bed96216d8
# ╟─5d99bf99-8973-4c4d-b88a-1e478b1678fb
# ╟─fe6f09ca-dc39-4d77-aae1-757c87ccd3e7
# ╟─f35a94e5-7568-4b89-af94-6167aebd3dab
# ╟─3b614ff9-663e-459e-9621-98edd8922fdc
# ╟─b721ca18-e5cd-4762-9ff0-9d94294b6969
# ╠═731ba6f9-fa51-4fa3-a2dd-7748c9ecbea3
# ╠═939f57a0-cb34-471b-aa24-10e5a95f410d
# ╟─6e3ef066-d115-11eb-2338-013a707dfe8a
# ╟─41bd06a2-e7bd-46c6-9249-6e69223b0e11
# ╟─94185c13-b6d2-4337-b5ce-336f5e128032
# ╟─8f417ce0-fa72-4b12-8d3b-8f88b2e5a2bf
# ╟─48be0808-f22a-4b72-a078-b2f2552db327
# ╟─f4d4785e-0df1-4882-8eb9-005ff2a50614
# ╟─0ca92cdc-102c-45e7-a5b0-17087b9ba634
# ╟─68e80154-126b-403c-a593-d896cdf96ef2
# ╟─0f54629c-e045-47bf-90a5-36e44c05b8f0
# ╟─18147da2-1945-42e4-b140-c1375f46962b
# ╟─0135af1a-d8c8-4027-af38-7f57bbfef282
# ╟─3ba73b7c-f5dc-4639-ab9c-c43cc992634f
# ╟─bbed8889-d401-4167-9370-9927aabbc83b
