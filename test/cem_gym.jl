module cemgym

# this is a port of the example found at
#	http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/lab1.html#starter

using Reinforce
using OpenAIGym
using Distributions

# ----------------------------------------------------------------

# given a (n*m)-length vector θ, split into a matrix and vector
function split_θ(θ, n, m)
	@assert length(θ) == n*m
	cutoff = (n-1) * m
	w = reshape(view(θ, 1:cutoff), m, (n-1))
	b = view(θ, cutoff+1:n*m)
	w, b
end



# Notes:
# 	- θ is a parameter vector, W and b are reshaped views of it

type DeterministicDiscreteLinearPolicy{W<:AbstractMatrix,B<:AbstractVector} <: AbstractPolicy
	w::W
	b::B
end

function DeterministicDiscreteLinearPolicy(env::AbstractEnvironment, θ)
	nS = length(state(env))
	nA = length(actions(env))
	w, b = split_θ(θ, nS+1, nA)
	DeterministicDiscreteLinearPolicy(w, b)
end

function Reinforce.action(π::DeterministicDiscreteLinearPolicy, r, s, A)
	# @show map(size, (s, π.w, π.b))
	y = π.w * s + π.b
	A[indmax(y)]
end


# ----------------------------------------------------------------

#
function noisy_evaluation(env, θ; kw...)
	π = policy(env, θ)
	R, T = episode!(env, π; kw...)
	R
end

# construct an appropriate policy given the environment state and action space
function policy(env, θ)
	policy_type = if typeof(actions(env)) <: DiscreteActionSet
		DeterministicDiscreteLinearPolicy
	else
		DeterministicContinuousLinearPolicy
	end
	policy_type(env, θ)
end


function do_cem_test(; env = GymEnv("CartPole-v0"),
					   maxiter = 200,
					   cem_iter = 100,
					   cem_batch_size = 20,
					   cem_elite_frac = 0.2,
					   stopping_reward_std = 1e-2,
					   stopping_norm = 1e-2)
	# do one step to ensure we have state/actions
	# step!(env)
	n = (length(state(env)) + 1) * length(actions(env))
	# reset!(env)

	# helpers... noisy_episode is a mappable function of θ to reward
	noisy_episode = θ -> noisy_evaluation(env, θ, maxiter=maxiter)
	n_elite = round(Int, cem_batch_size * cem_elite_frac)

	# μ	and σ are the mean and standard dev of the θ params
	μ = zeros(n)
	σ = ones(n)
	last_σ = ones(n)

	for i=1:cem_iter
		# sample thetas from a multivariate normal distribution
		N = MultivariateNormal(μ, σ)
		θs = [rand(N) for k=1:cem_batch_size]

		# compute rewards and pick out an elite set
		Rs = map(noisy_episode, θs)
		@show Rs
		elite_indices = sortperm(Rs, rev=true)[1:n_elite]
		elite_θs = θs[elite_indices]

		# update μ and σ with the sample mean/std of the elite set
		for j=1:n
			θj = [θ[j] for θ in elite_θs]
			μ[j] = mean(θj)
			σ[j] = std(θj)
		end

		# finish the iteration by running an episode with θ = μ
		info("Iteration $i. mean(R): $(mean(Rs)) max(R): $(maximum(Rs))")
		R, T = episode!(env, policy(env, μ), maxiter=maxiter)
		info("Episode $i finished after $T steps. Total reward: $R")
		@show μ σ

		# have we converged?
		stdRs = std(Rs)
		diffnorm = norm(σ - last_σ)
		@show diffnorm stdRs
		if diffnorm < stopping_norm || stdRs < stopping_reward_std
			info("Converged after $(i*cem_batch_size) episodes.")
			break
		end
		last_σ[:] = σ
	end
end


end #module
