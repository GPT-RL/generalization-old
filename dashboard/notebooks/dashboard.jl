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
end

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
end

# ╔═╡ 48be0808-f22a-4b72-a078-b2f2552db327
function format_logs(run::AbstractDict)
	sweep_id = ismissing(run["sweep"]) ? nothing : run["sweep"]["id"]
	@chain run["run_logs"] begin
		map(x -> Dict((Symbol(replace(k, " " => "_")) => v for (k,v) in x["log"])), _)
		DataFrame
		insertcols!(:run_id => run["id"], :sweep_id => sweep_id)
	end
end

# ╔═╡ f4d4785e-0df1-4882-8eb9-005ff2a50614
function format_run(run::AbstractDict)
	sweep_id = something(get(run, "sweep", nothing), Dict("id" => nothing))["id"]
	data = run["metadata"]
	push!(data, pop!(data, "parameters")...)
	items = [Symbol(replace(k, " " => "_")) => v for (k, v) in data]
	push!(items, :run_id => run["id"], :sweep_id => sweep_id)
	Dict(items)
end

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
end

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
end

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
end

# ╔═╡ 18147da2-1945-42e4-b140-c1375f46962b
function sweep_data(sweep_runs::AbstractDataFrame; ids = nothing)::DataFrame
	s_run_ids = sweep_runs.run_id |> unique |> collect
	if !isnothing(ids)
		s_run_ids = filter(in(ids), s_run_ids)
	end
	s_df = [run_data(collect(i), sweep_runs) for i in Iterators.partition(s_run_ids, 10)]
	s_df = vcat(s_df...; cols=:union)
	dropmissing!(s_df, :step)
end

# ╔═╡ 0135af1a-d8c8-4027-af38-7f57bbfef282
gpt_runs = sweep_runs([26, 23, 22, 19, 17])

# ╔═╡ 3ba73b7c-f5dc-4639-ab9c-c43cc992634f
gpt_df = sweep_data(gpt_runs);

# ╔═╡ 8798b46c-1b0b-4d34-93ef-1d5062f8a632
md"##### Smoothing"

# ╔═╡ bbed8889-d401-4167-9370-9927aabbc83b
@bind window_size Slider(1:200, default=100)

# ╔═╡ 5cba55d9-c7fe-4918-88b3-abbabe08c56b
function smooth_curves(df::AbstractDataFrame, y::Symbol, group_col::Symbol, window_size::Int)::DataFrame
	@chain df begin
		dropmissing!([y, group_col])
		groupby(group_col)
		pairs
		[@chain g begin
				DataFrame
				sort!(:step)
				DataFrame(
					group_col => first(k),
					:step => round.(Int, rollmedian(_.step, min(window_size, nrow(_)))),
					:mean => rollmean(_[:, y], min(window_size, nrow(_))),
					:std => rollstd(_[:, y], min(window_size, nrow(_))),
				)
				transform(
					[:mean, :std] => ((x, y) -> x - y) => [:lower],
					[:mean, :std] => ((x, y) -> x + y) => [:upper]
				)
			end for (k, g) in _]
		vcat(_...)
	end
end

# ╔═╡ 70e759a8-2f8d-46c8-aa02-c581dd17c1c4


# ╔═╡ 23d3bad8-099c-4794-806a-aaacc86b9e1a
gpt_df

# ╔═╡ 92de44db-86be-4fe5-9eb1-1ac657676b95


# ╔═╡ c252df75-fb25-470d-956d-0aa532701565
md"### Runs"

# ╔═╡ 8b4a7da4-562e-4ad7-aa8b-8a458f302508
runs = format_runs(latest_runs(10))

# ╔═╡ 163fc23c-952e-4002-be1c-0baafbccd8aa
md"#### Select runs to visualize"

# ╔═╡ fe1f61d3-2b34-4a54-9fdc-9e4504248994
begin
	checkbox_values = [
		string(r.run_id) => "$(r.run_id)"
		for r in eachrow(runs)
	]
	@bind selected_run_id_strs MultiCheckBox(
		checkbox_values;
		orientation=:column,
		select_all=true
	)
end

# ╔═╡ 26b7f618-1576-4299-b8d5-5bfef31401f1
begin
	selected_run_ids = parse.(Int, something(selected_run_id_strs, []))
	selected_runs = filter(:run_id => in(selected_run_ids), runs, view=true)
end;

# ╔═╡ 352700d7-8f94-4aa6-9aff-4de783ae1b4d
md"#### Group by"

# ╔═╡ 83e33b0e-68ac-4039-a41c-3273b0317657
@bind group_col Radio(
	names(
		selected_runs[!, [!any(isnothing, c) for c in eachcol(selected_runs)]]
		),
	default="run_id")

# ╔═╡ d4729079-765c-4c54-b8c1-252e7b3fa4a9
df = isempty(selected_run_ids) ? DataFrame() : run_data(selected_run_ids, runs);

# ╔═╡ 20995b98-9a5b-41a4-be86-72be1a35e698
function metric_stats(df::AbstractDataFrame, metric_col::Symbol, group_col::Symbol)::DataFrame
	result = []
	df = dropmissing(df, [metric_col, group_col], view=true)
	for (g_key, g) in pairs(groupby(df, group_col))
		group_data = @chain groupby(g, :step) begin
				combine(
					metric_col => mean => metric_col,
					metric_col => std => :std
				)
				_ .= ifelse.(isnan.(_), 0, _)  # Replace NAN with 0
				transform(
					[metric_col, :std] => ((x, y) -> x - y) => :lower,
					[metric_col, :std] => ((x, y) -> x + y) => :upper;
				)
				select(:step, metric_col, :lower, :upper)
		end
		insertcols!(group_data, group_col => first(g_key))
		push!(result, group_data)
	end
	vcat(result...)
end

# ╔═╡ 19186224-91a0-421b-bdca-0689567c701f
md"### Plots"

# ╔═╡ 7acd7dfb-7edb-4904-8877-ac66a498eb9c
begin
	struct HTMLDocument
		embedded
	end
	function Base.show(io::IO, mime::MIME"text/html", doc::HTMLDocument)
		println(io, "<html>")
		show(io, mime, doc.embedded)
		println(io, "</html>")
	end
end

# ╔═╡ fe6f09ca-dc39-4d77-aae1-757c87ccd3e7
begin
	action_colors(n) = cmap("L08"; N=n)
	plot(
		smooth_curves(gpt_df, :episode_return, :action_hidden_size, window_size),
		x=:step, y=:mean,
		ymin=:lower, ymax=:upper,
		color=:action_hidden_size,
		Guide.xlabel("Step"),
		Guide.ylabel("Episode Return"),
		Geom.line,
		Geom.ribbon,
		Scale.color_discrete(action_colors),
		Guide.colorkey(title="Action Hidden Size"),
		Guide.title("Performance Across Action Sizes"),
		alpha=[0.5]
	) |> HTMLDocument
end

# ╔═╡ 1f0c13a6-ae73-4aec-bee3-745610c23175
plot(
		smooth_curves(
			sweep_data(sweep_runs([26, 9])),
			:episode_return,
			:sweep_id,
			window_size
		),
		x=:step, y=:mean,
		ymin=:lower, ymax=:upper,
		color=:sweep_id,
		Guide.xlabel("Step"),
		Guide.ylabel("Episode Return"),
		Geom.line,
		Geom.ribbon,
		Scale.color_discrete(),
		Guide.colorkey(labels=["No", "Yes"], title="GPT?"),
		Guide.title("GPT vs. No GPT Performance"),
		alpha=[0.5]
	) |> HTMLDocument

# ╔═╡ 24091393-3afb-4a1c-9a9a-8ab2627285a3
function multiline_chart(df::AbstractDataFrame, y::Symbol, group_col::Symbol)
	if !isempty(df)
		plt = (
			data(metric_stats(df, y, group_col)) *
			mapping(
				:step,
				y;
				color=group_col => nonnumeric,
				lower=:lower,
				upper=:upper
			) *
			visual(LinesFill))
		set_title!(draw(plt), replace(string(y), "_" => " ") |> titlecase)
	end
end

# ╔═╡ 9c6b8bc6-94ad-4fdf-a9b6-6df962227ead
multiline_chart(df, :loss, Symbol(group_col))

# ╔═╡ a346a9f5-490a-477c-8979-160be8198073
multiline_chart(df, :episode_return, Symbol(group_col))

# ╔═╡ 5811c404-e244-4fae-a442-db5e4946039b
md"### Data"

# ╔═╡ 233b2be1-ec9b-4b3e-b0cb-fe90ddaf4a1e
df

# ╔═╡ Cell order:
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
# ╟─8798b46c-1b0b-4d34-93ef-1d5062f8a632
# ╟─bbed8889-d401-4167-9370-9927aabbc83b
# ╟─5cba55d9-c7fe-4918-88b3-abbabe08c56b
# ╠═fe6f09ca-dc39-4d77-aae1-757c87ccd3e7
# ╟─70e759a8-2f8d-46c8-aa02-c581dd17c1c4
# ╟─1f0c13a6-ae73-4aec-bee3-745610c23175
# ╠═23d3bad8-099c-4794-806a-aaacc86b9e1a
# ╠═92de44db-86be-4fe5-9eb1-1ac657676b95
# ╟─c252df75-fb25-470d-956d-0aa532701565
# ╟─8b4a7da4-562e-4ad7-aa8b-8a458f302508
# ╟─163fc23c-952e-4002-be1c-0baafbccd8aa
# ╟─fe1f61d3-2b34-4a54-9fdc-9e4504248994
# ╟─26b7f618-1576-4299-b8d5-5bfef31401f1
# ╟─352700d7-8f94-4aa6-9aff-4de783ae1b4d
# ╟─83e33b0e-68ac-4039-a41c-3273b0317657
# ╟─d4729079-765c-4c54-b8c1-252e7b3fa4a9
# ╟─20995b98-9a5b-41a4-be86-72be1a35e698
# ╟─19186224-91a0-421b-bdca-0689567c701f
# ╟─7acd7dfb-7edb-4904-8877-ac66a498eb9c
# ╟─24091393-3afb-4a1c-9a9a-8ab2627285a3
# ╟─9c6b8bc6-94ad-4fdf-a9b6-6df962227ead
# ╟─a346a9f5-490a-477c-8979-160be8198073
# ╟─5811c404-e244-4fae-a442-db5e4946039b
# ╠═233b2be1-ec9b-4b3e-b0cb-fe90ddaf4a1e
