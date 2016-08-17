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

type CrossEntropyMethodPolicy <: AbstractPolicy
	μ::Vector{Float64}
	σ::Vector{Float64}
	noise_func::Function # additional deviation at each timestep
	trans  # the transformation, updated at each iteration
end

function CrossEntropyMethodPolicy(env::AbstractEnvironment, noise_func = t->0.0)
	n = (length(state(env)) + 1) * length(actions(env))
	CrossEntropyMethodPolicy(zeros(n), ones(n), noise_func, cem_transformation(env, zeros(n)))
end

# discrete: our action is the action which maximizes the affine transform
function Reinforce.action(π::CrossEntropyMethodPolicy, r, s, A::DiscreteSet)
	A[indmax(transform(π.trans, s))]
end

# continuous: return the transform value
function Reinforce.action(π::CrossEntropyMethodPolicy, r, s, A::IntervalSet)
	Transformations.sigmoid(transform(π.trans, s)[1]) * (A.amax-A.amin) + A.amin
end

# update μ and σ with the sample mean/std of the elite set
function LearnBase.learn!(π::CrossEntropyMethodPolicy, elite_θs, t)
	for j=1:length(π.μ)
		θj = [θ[j] for θ in elite_θs]
		π.μ[j] = mean(θj)
		π.σ[j] = std(θj) + π.noise_func(t)
	end
end


# ----------------------------------------------------------------

function myplot(t, hists, anim)
	(env,i,sars) -> if mod1(t,3)==1 && mod1(i,10)==1
		plot(env,t,i,hists)
		frame(anim)
	else
		return
	end
end



function LearnBase.learn!(π::CrossEntropyMethodPolicy, env::AbstractEnvironment;
			maxiter = 200,
			cem_iter = 100,
			cem_batch_size = 20,
			cem_elite_frac = 0.2,
			# stopping_reward_std = 1e-2,
			stopping_norm = 1e-2)

	# this is a mappable function of θ to reward
	function cem_episode(θ; kw...)
		π.trans = cem_transformation(env, θ)
		R, T = episode!(env, π; maxiter = maxiter, kw...)
		R
	end
	anim = Animation()

	n_elite = round(Int, cem_batch_size * cem_elite_frac)
	last_μ = copy(π.μ)
	hist_min, hist_mean, hist_max = zeros(0),zeros(0),zeros(0)
	for t=1:cem_iter
		# sample thetas from a multivariate normal distribution
		N = MultivariateNormal(π.μ, π.σ)
		θs = [rand(N) for k=1:cem_batch_size]

		# compute rewards and pick out an elite set
		Rs = map(cem_episode, θs)
		elite_indices = sortperm(Rs, rev=true)[1:n_elite]
		elite_θs = θs[elite_indices]
		info("Iteration $t. mean(R): $(mean(Rs)) max(R): $(maximum(Rs))")

		push!(hist_min, minimum(Rs))
		push!(hist_mean, mean(Rs))
		push!(hist_max, maximum(Rs))

		# update the policy from the elite set
		learn!(π, elite_θs, t)
		@show π.μ π.σ

		# finish the iteration by evaluating an episode with θ = μ
		R = cem_episode(π.μ, stepfunc = myplot(t, (hist_min,hist_mean,hist_max), anim))
		info("Episode $t finished. Total reward: $R")

		# have we converged?
		# stdRs = std(Rs)
		# if stdRs < stopping_reward_std
		normdiff = norm(π.μ - last_μ)
		@show normdiff
		if normdiff < stopping_norm
			info("Converged after $(t*cem_batch_size) episodes.")
			break
		end
		last_μ = copy(π.μ)
	end
	gif(anim)
end


# ----------------------------------------------------------------

# given a (n*m)-length vector θ, split into a matrix and vector
function split_θ(θ, n, m)
	@assert length(θ) == n*m
	cutoff = (n-1) * m
	w = reshape(view(θ, 1:cutoff), m, (n-1))
	b = view(θ, cutoff+1:n*m)
	w, b
end


# construct an appropriate policy given the environment state and action space
function cem_transformation(env, θ)
	# model_type = if typeof(actions(env)) <: DiscreteSet
	# 	Affine
	# else
	# 	DeterministicContinuousLinearPolicy
	# end
	model_type = Affine

	nS = length(state(env))
	nA = length(actions(env))
	w, b = split_θ(θ, nS+1, nA)

	# create a Transformations.Transformation
	transformation(model_type(w, b))
end

# ----------------------------------------------------------------

# initialize a policy, do the learning, then return the policy
function do_cem_test(; env = GymEnv("CartPole-v0"),
					   noise_max = 1.0,
					   noise_steps = 20,
					   noise_func = t -> max(noise_max - t/noise_steps, 0.0),
					   kw...)
	π = CrossEntropyMethodPolicy(env, noise_func)
	learn!(π, env; kw...)
	π
end


end #module
