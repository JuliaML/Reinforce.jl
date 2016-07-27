module Reinforce

using Reexport
@reexport using StatsBase
@reexport using LearnBase

export
	AbstractState,
	AbstractAgent,
	AbstractEnvironment,

	AbstractActionSet,
	ContinuousActionSet,
	DiscreteActionSet,

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
	History

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

# ----------------------------------------------------------------

abstract AbstractEnvironment
abstract AbstractState
abstract AbstractAgent

# `r, s, A = observe!(env)` should return `(reward, state, actions)`
# Note: most environments will not implement this directly
function observe!(env::AbstractEnvironment)
    reward!(env), state!(env), actions(env)
end

# `r = reward!(env)` returns the current reward, optionally updating it first
function reward end
function reward! end

# `s = state!(env)` returns the current state, optionally updating it first
function state end
function state! end

# `A = actions(env)` returns a list/set/description of valid actions
function actions end

# `a = action(agent, r, s, A)` should take in the last reward `r`, current state `s`, 
#      and set of valid actions `A`, then return an action `a`
function action end

# ----------------------------------------------------------------


# """
# Agents and environments should implement a small interface:

# - r,s = observe(env)
# - a = act(agent, r, s)
# """

# abstract AbstractState
# abstract AbstractAgent
# act(agent::AbstractAgent, reward::Number, state::AbstractState) = error("unimplemented: act($agent, $reward, $state)")

# abstract AbstractEnvironment
# observe(env::AbstractEnvironment) = error("unimplemented: observe($env)")
# actions(env::AbstractEnvironment) = error("unimplemented: actions($env)")

# ----------------------------------------------------------------


include("states.jl")
include("agents.jl")

# ----------------------------------------------------------------

end # module
