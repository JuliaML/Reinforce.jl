module cemgym

# this is a port of the example found at
#	http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/lab1.html#starter

using Reinforce
using OpenAIGym

# ----------------------------------------------------------------

# given a (n*m)-length vector θ, split into a matrix and vector
function split_θ(θ, n, m)
	@assert length(θ) == n*m
	cutoff = (n-1) * m
	w = reshape(view(θ, 1:cutoff), (n-1), m)
	b = view(θ, cutoff+1:end)
	w, b
end



# Notes:
# 	- θ is a parameter vector, W and b are reshaped views of it

type DeterministicDiscreteLinearPolicy{W,B} <: AbstractAgent
	w::W
	b::B
	function DeterministicDiscreteLinearPolicy(env, θ)
		nS = length(state(env))
		nA = length(actions(env))
		w, b = split_θ(θ, nS, nA)
		DeterministicDiscreteLinearPolicy(w, b)
	end
end

function Reinforce.action(π, r, s, A)
	y = s * π.w + π.b
	A[indmax(y)]
end


# ----------------------------------------------------------------

function do_episode(π, env, N; render=false)
	total_reward = 0.0
	# TODO
end

function noisy_evaluation(env, θ; kw...)
	π = policy(env, θ)
	do_episode(π, env; kw...)
end

# construct an appropriate policy given the environment state and action space
function policy(env, θ)
	policy_type = if is_discrete(env)
		DeterministicDiscreteLinearPolicy
	else
		DeterministicContinuousLinearPolicy
	end
	policy_type(state_space(env), actions(env), θ)
end




end #module
