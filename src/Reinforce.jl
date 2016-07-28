module Reinforce

using Reexport
@reexport using StatsBase
@reexport using LearnBase

export
	AbstractState,
	AbstractPolicy,
	AbstractEnvironment,

	AbstractActionSet,
	ContinuousActionSet,
	DiscreteActionSet,

	reset!,
	step!,
	episode!,
	observe!,
	reward,
	reward!,
	state,
	state!,
	actions,
	action,

	state,
	state!,
	StateVector,
	History,

	RandomPolicy

# ----------------------------------------------------------------

# # we can use uniform distributions to represent most action sets
# # by overloading `in`, we can check that an action is valid
# Base.in(x::Number, dist::Distributions.ContinuousUniform) = minimum(dist) <= x <= maximum(dist)
# Base.in(x::Integer, dist::Distributions.DiscreteUniform) = x in (minimum(dist):maximum(dist))

abstract AbstractActionSet

immutable ContinuousActionSet{T} <: AbstractActionSet
	amin::T
	amax::T
end
Base.rand(aset::ContinuousActionSet) = rand() * (aset.amax - aset.amin) + aset.amin
Base.in(x, aset::ContinuousActionSet) = aset.amin <= x <= aset.amax

immutable DiscreteActionSet{T} <: AbstractActionSet
	actions::T
end
Base.rand(aset::DiscreteActionSet) = rand(aset.actions)
Base.in(x, aset::DiscreteActionSet) = x in aset.actions
Base.length(aset::DiscreteActionSet) = length(aset.actions)

# ----------------------------------------------------------------

abstract AbstractEnvironment
abstract AbstractState
abstract AbstractPolicy

# `r, s, A = observe!(env)` should return `(reward, state, actions)`
# Note: most environments will not implement this directly
function observe!(env::AbstractEnvironment)
    reward!(env), state!(env), actions(env)
end

# `reset!(env)` resets an episode
function reset! end

# observe and get action from policy, plus any other details.
#	returns false when the episode is finished
function step! end

# `r = reward!(env)` returns the current reward, optionally updating it first
function reward end
function reward! end

# `s = state!(env)` returns the current state, optionally updating it first
function state end
function state! end

# `A = actions(env)` returns a list/set/description of valid actions
function actions end

# `a = action(policy, r, s, A)` should take in the last reward `r`, current state `s`, 
#      and set of valid actions `A`, then return an action `a`
function action end

# ----------------------------------------------------------------

# override these for custom functionality for your environment
on_step(env::AbstractEnvironment, i::Int) = return
# function on_episode_finished(env::AbstractEnvironment, episode_num::Int,
# 							 iteration_num::Int, total_reward::Float64)
# 	info("Episode $episode_num finished after $iteration_num steps.  reward = $total_reward")
# end

# run a single episode. by default, it will run until `step!` returns false
function episode!(env::AbstractEnvironment,
				  policy::AbstractPolicy,
				  episode_num::Int = 1;
				  maxiter = typemax(Int))
	reset!(env)
	i = 1
	total_reward = 0.0
	while true
		done = step!(env, policy)
		on_step(env, i)
		total_reward += reward(env)
		if done || i > maxiter
			break
		end
		i += 1
	end
	# on_episode_finished(env, episode_num, i, total_reward)
	total_reward, i
end


# """
# Policys and environments should implement a small interface:

# - r,s = observe(env)
# - a = act(policy, r, s)
# """

# abstract AbstractState
# abstract AbstractPolicy
# act(policy::AbstractPolicy, reward::Number, state::AbstractState) = error("unimplemented: act($policy, $reward, $state)")

# abstract AbstractEnvironment
# observe(env::AbstractEnvironment) = error("unimplemented: observe($env)")
# actions(env::AbstractEnvironment) = error("unimplemented: actions($env)")

# ----------------------------------------------------------------


include("states.jl")
include("agents.jl")

# ----------------------------------------------------------------

end # module
