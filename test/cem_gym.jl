module cemgym

# this is a port of the example found at
#	http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/lab1.html#starter

using Reinforce
# using OpenAIGym
using Distributions
using Transformations

# ENV["GKS_DOUBLE_BUF"] = "true"
using Plots; gr(size=(500,200))

# ----------------------------------------------------------------

"""
Wraps a LearnBase.Transformation which converts an input vector to action values.
For discrete actions, it chooses the action which produces the highest value.
For continuous (interval) actions, it squashes actions to [0,1].
"""
immutable TransformPolicy{T} <: AbstractPolicy
	trans::T
end

# discrete: our action is the action which maximizes the affine transform
function Reinforce.action(π::TransformPolicy, r, s, A::DiscreteSet)
	A[indmax(transform(π.trans, s))]
end

# continuous: return the transform value, squashed to [0,1]
# TODO: remove the squashing when a "Affine + Sigmoid" transformation is available
function Reinforce.action(π::TransformPolicy, r, s, A::IntervalSet)
	Transformations.sigmoid(transform(π.trans, s)[1]) * (A.amax-A.amin) + A.amin
end

# given a (n*m)-length vector θ, split into a matrix and vector
function split_θ(θ, n, m)
	@assert length(θ) == n*m
	cutoff = (n-1) * m
	w = reshape(view(θ, 1:cutoff), m, (n-1))
	b = view(θ, cutoff+1:n*m)
	w, b
end


# construct an appropriate policy given the environment state and action space
function cem_policy(env, θ)
	model_type = Affine
	nS = length(state(env))
	nA = length(actions(env))
	w, b = split_θ(θ, nS+1, nA)

	# create a Transformations.Transformation, and wrap it in the policy
	trans = transformation(model_type(w, b))
	TransformPolicy(trans)
end

# ----------------------------------------------------------------

# """
# The Cross Entropy Method is a simple but useful optimization method without a need
# for gradients or differentiability.
# """
# type CrossEntropyMethod
# 	μ::Vector{Float64}
# 	σ::Vector{Float64}
# 	Z::Vector{Float64}  # extra variance
# 	noise_func::Function # additional deviation at each timestep
# 	options::KW
# end

# function CrossEntropyMethod(n::Integer, noise_func = t->0.0; kw...)
# 	options = merge(default_options(CrossEntropyMethod), KW(kw))
# 	CrossEntropyMethod(zeros(n), ones(n), zeros(n), noise_func, options)
# end

# default_options(::Type{CrossEntropyMethod}) = KW(
# 		:maxiter => 200,
# 		:cem_iter => 100,
# 		:cem_batch_size => 20,
# 		:cem_elite_frac => 0.2,
# 		:stopping_norm => 1e-2,
# 	)

# ----------------------------------------------------------------

# function LearnBase.learn!(mgr::CrossEntropyMethod, env::AbstractEnvironment, doanim = false) 
	
	# !!! INIT:

	# # this is a mappable function of θ to reward
	# cem_episode = θ -> begin
	# 	π = cem_policy(env, θ)
	# 	R, T = episode!(env, π; maxiter = solver.options[:maxiter])
	# 	R
	# end

	# anim = doanim ? Animation() : nothing
	# n_elite = round(Int, solver.options[:cem_batch_size] * solver.options[:cem_elite_frac])
	# # last_μ = copy(solver.μ)
	# last_μ = similar(solver.μ)
	# hist_min, hist_mean, hist_max = zeros(0),zeros(0),zeros(0)


	# for t=1:solver.options[:cem_iter]
	# 	# !!! UPDATE:

	# 	last_μ = copy(solver.μ)

	# 	# sample thetas from a multivariate normal distribution
	# 	N = MultivariateNormal(solver.μ, solver.σ)
	# 	θs = [rand(N) for k=1:solver.options[:cem_batch_size]]

	# 	# compute rewards and pick out an elite set
	# 	Rs = map(cem_episode, θs)
	# 	elite_indices = sortperm(Rs, rev=true)[1:n_elite]
	# 	elite_θs = θs[elite_indices]
	# 	info("Iteration $t. mean(R): $(mean(Rs)) max(R): $(maximum(Rs))")


	# 	# update the policy from the elite set
	# 	for j=1:length(solver.μ)
	# 		θj = [θ[j] for θ in elite_θs]
	# 		solver.μ[j] = mean(θj)
	# 		solver.Z[j] = solver.noise_func(t)
	# 		solver.σ[j] = sqrt(var(θj) + solver.Z[j])
	# 	end
	# 	@show solver.μ solver.σ solver.Z

	# 	# !!! TRACE:

	# 	push!(hist_min, minimum(Rs))
	# 	push!(hist_mean, mean(Rs))
	# 	push!(hist_max, maximum(Rs))

	# 	# finish the iteration by evaluating an episode with θ = μ
	# 	R, T = episode!(
	# 		env,
	# 		cem_policy(env, solver.μ),
	# 		maxiter = solver.options[:maxiter],
	# 		stepfunc = myplot(t, hist_min, hist_mean, hist_max, anim)
	# 	)
	# 	info("Iteration $t finished. Total reward: $R")

	# 	# !!! CONVERGENCE:

	# 	normdiff = norm(solver.μ - last_μ)
	# 	@show normdiff
	# 	if normdiff < solver.options[:stopping_norm]
	# 		info("Converged after $(t*solver.options[:cem_batch_size]) episodes.")
	# 		break
	# 	end
	# end

	# doanim && gif(anim)
# end

# ----------------------------------------------------------------

function myplot(t, hist_min, hist_mean, hist_max, anim=nothing)
	(env,i,sars) -> if mod1(t,3)==1 && mod1(i,10)==1
		plot(
			plot(hist_mean, c=:black, fill=((hist_min,hist_max), 0.2), title="Progress", leg=false),
			plot(env, title = "Episode: $t  Iter: $i")
		)
		if anim == nothing
			gui()
		else
			frame(anim)
		end
	else
		return
	end
end

# ----------------------------------------------------------------


# initialize a policy, do the learning, then return the policy
function do_cem_test(; env = GymEnv("CartPole-v0"),
                       maxiter = 200,
					   # noise_max = 1.0,
					   # noise_steps = 20,
					   # noise_func = t -> max(noise_max - t/noise_steps, 0.0),
					   kw...)
    function f(θ)
        π = cem_policy(env, θ)
        R, T = episode!(env, π; maxiter = maxiter)
        R
    end
    mgr = CrossEntropyMethod(f; kw...)
    istate = CrossEntropyMethodState(n = (length(state(env)) + 1) * length(actions(env)))
    learn!(mgr, istate)
    @show istate
	# solver = CrossEntropyMethod(n, noise_func; kw...)
	# learn!(solver, env)
	# solver
end


end #module
