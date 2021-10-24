### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 6e3ef066-d115-11eb-2338-013a707dfe8a
begin
	using PlutoUI
	using DataFrames
	using Statistics
	using HTTP
	using JSON
	using Chain
	using Tables
	using MLStyle
end

# ╔═╡ 41bd06a2-e7bd-46c6-9249-6e69223b0e11
HASRUA_ENDPOINT_URL = "http://rldl12.eecs.umich.edu:1200/v1/graphql"

# ╔═╡ 94185c13-b6d2-4337-b5ce-336f5e128032
function gql_query(query:: String; variables:: Dict = nothing)
	r = HTTP.request(
		"POST",
		HASRUA_ENDPOINT_URL;
		verbose=3,
		body= JSON.json(Dict("query" => query, "variables" => variables)) ,
		# headers=Dict("x-hasura-admin-secret" => HASURA_ADMIN_SECRET),
	)
	@chain r begin
		_.body
		String
		JSON.parse(_, null=missing)
	    _["data"]
	end
end

# ╔═╡ ed56319c-fd7e-478e-a673-f799debbf7b3
function get_sweep(sweep_ids::AbstractVector{Int}, max_step::Int)
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
	@chain query begin
		gql_query(_; variables=Dict("ids" => sweep_ids, "max_step" => max_step))
		_["logs_less_than_step"]		
		map(_) do d
			Dict(
				"run_id" => d["run_id"],
				"sweep_id" => d["run"]["sweep_id"],
				d["log"]...,
				d["run"]["metadata"]["parameters"]...
			)
		end
		map(_) do d
			Dict(
				d...,
				[k => v for (k1, v1, k2, v2) in [
							(
								"hours", get(d, "time-delta", 0) / 3600, 
								"time-delta", get(d, "hours", 0) * 3600,
							),
						] 
						for (k, v) in [
								(k1, get(d, k1, v1)), 
								(k2, get(d, k2, v2)),
								]]...,				
				[name => get(d, name, false) for name in [
							"randomize_parameters"
						]]...,
				[name => get(d, name, nothing) for name in [
							"config",
						]]... 
			)
		end
		collect
		vcat(DataFrame.(_)...)
	end
end;

# ╔═╡ da091e45-0d08-490a-ba85-ee9acb6700bc
sweeps = filter(get_sweep([784], 10000000)) do row
				row.env == "Seaquest-v0"
			end

# ╔═╡ 1a6b7810-977d-4a82-ab14-b24e154ad491
EPISODE_RETURN = "episode return"

# ╔═╡ 7e2ac3d5-9b72-4859-96a7-b6b8711e14b3
min_returns = @chain sweeps begin
	dropmissing(_, EPISODE_RETURN)
	groupby(_, [:env])
	combine(_, EPISODE_RETURN => minimum)
	Dict(k=>v for (k,v) in eachrow(_))
end

# ╔═╡ 5a29110f-deca-4812-9240-ab445ed665c8
max_returns = Dict(
	"BeamRider-v0" => 1590, 
	"PongNoFrameskip-v0" => 20.7, 
	"Seaquest-v0" => 1204.5,
	"Qbert-v0" => 14293.3
) # from PPO paper

# ╔═╡ c838c44c-4bd8-4eb0-a18b-ceaaa5bdce93
dframe = @chain sweeps begin
	filter(:step => >=(8000000), _)
	groupby(_, [:env])
	transform(_, ["env", EPISODE_RETURN] =>
		function (envs, ret) 
			@match [Set(envs)...] (
				[env] => (ret .- min_returns[env]) ./ max_returns[env] 							)
		end => [EPISODE_RETURN])
	_[!, filter(names(_)) do name
			!(name in [
				"action loss",
				"config",
				"cuda",
				"entropy",
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
end

# ╔═╡ 2735fbc7-a318-4b4e-8ceb-0e65084cf972
function filter_by_type(ty) 
	filter(name -> eltype(dframe[:, name]) == ty, names(dframe))
end

# ╔═╡ 1fe4e26a-2786-4094-a7e5-0154b461e874
df = @chain dframe begin
	groupby(_, "run ID")
	combine(_, 
		filter_by_type(Float64) .=> first,
		filter_by_type(Int64) .=> first,
		filter_by_type(Bool) .=> first,
		EPISODE_RETURN .=> mean .=> :episode_return_mean,
		)
	_[!, filter(n -> !(n in  ["run ID", "episode return_first"]), names(_))]
	sort!(_, [:episode_return_mean], rev=true)
	rename(name -> replace(name, "_first" => ""), _)
end

# ╔═╡ 490d6756-f769-4a13-9438-ad724aa013ee
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
end

# ╔═╡ c3e89733-4a83-48d4-bf34-9f4f7895b798
function get_cor_df(df)

	cor_df = DataFrame()
	
	cor_df.name = names(df)
	
	cor_mat = cor(Matrix(df))
	return_index = findfirst(name -> name == "episode_return_mean", names(df))
	cor_df.correlation = cor_mat[:, return_index]
	
	sort!(cor_df, [:correlation], )
	cor_df
end

# ╔═╡ f43f3cf4-b9e4-4c21-bf95-96a0c494c99b
md"# Continuous Correlation"

# ╔═╡ a4cb4dbe-b880-4a4a-9b91-ccf6eaff1b07
get_cor_df(df)

# ╔═╡ 5fb3a66b-49a2-41f6-aea5-65a2bd4d7240
md"# Discrete Correlation"

# ╔═╡ 6c27eed0-e73d-4a44-b286-dca6ac404dbd
get_cor_df(bool_df)

# ╔═╡ 8b15faf7-0d54-481f-a513-311e963dd8c1
Dict(k => unique(df, k)[!, k] for k in names(df))

# ╔═╡ Cell order:
# ╠═6e3ef066-d115-11eb-2338-013a707dfe8a
# ╠═41bd06a2-e7bd-46c6-9249-6e69223b0e11
# ╠═94185c13-b6d2-4337-b5ce-336f5e128032
# ╠═ed56319c-fd7e-478e-a673-f799debbf7b3
# ╠═da091e45-0d08-490a-ba85-ee9acb6700bc
# ╠═1a6b7810-977d-4a82-ab14-b24e154ad491
# ╠═7e2ac3d5-9b72-4859-96a7-b6b8711e14b3
# ╠═5a29110f-deca-4812-9240-ab445ed665c8
# ╠═2735fbc7-a318-4b4e-8ceb-0e65084cf972
# ╠═c838c44c-4bd8-4eb0-a18b-ceaaa5bdce93
# ╠═1fe4e26a-2786-4094-a7e5-0154b461e874
# ╠═490d6756-f769-4a13-9438-ad724aa013ee
# ╟─c3e89733-4a83-48d4-bf34-9f4f7895b798
# ╟─f43f3cf4-b9e4-4c21-bf95-96a0c494c99b
# ╠═a4cb4dbe-b880-4a4a-9b91-ccf6eaff1b07
# ╟─5fb3a66b-49a2-41f6-aea5-65a2bd4d7240
# ╟─6c27eed0-e73d-4a44-b286-dca6ac404dbd
# ╠═8b15faf7-0d54-481f-a513-311e963dd8c1
