
__precompile__()

module Reinforce

using Reexport
@reexport using StatsBase
@reexport using LearnBase
using RecipesBase

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
Base.getindex(aset::DiscreteActionSet, i::Int) = aset.actions[i]

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

# run a single episode. by default, it will run until `step!` returns false
function episode!(env::AbstractEnvironment,
				  policy::AbstractPolicy;
				  maxiter::Int = typemax(Int),
				  stepfunc::Function = on_step)
	reset!(env)
	i = 1
	total_reward = 0.0
	while true
		done = step!(env, policy)
		stepfunc(env, i)
		total_reward += reward(env)
		if done || i > maxiter
			break
		end
		i += 1
	end
	total_reward, i
end


# ----------------------------------------------------------------


include("states.jl")
include("policy.jl")

include("envs/cartpole.jl")

# ----------------------------------------------------------------

end # module
