
__precompile__()

module Reinforce

using Reexport
@reexport using StatsBase
using Distributions
@reexport using LearnBase
using RecipesBase

export
	AbstractActionSet,
	ContinuousActionSet,
	DiscreteActionSet,
	MultiActionSet,

	AbstractEnvironment,
	reset!,
	step!,
	reward,
	state,
	actions,

	AbstractPolicy,
	RandomPolicy,
	action,

	AbstractState,
	StateVector,
	History,
	# state,
	state!,

	Episode,
	episode!

# ----------------------------------------------------------------

abstract AbstractActionSet

# allow continuous value(s) in a range(s)
immutable ContinuousActionSet{T} <: AbstractActionSet
	  amin::T
	  amax::T

    function ContinuousActionSet{S<:AbstractVector}(amin::S, amax::S)
        if !(length(amin) == length(amax))
            error("For multi-valued continuous action sets, min and max value must have same length")
        end
        new(amin, amax)
    end

    ContinuousActionSet{S<:Number}(amin::S, amax::S) = new(amin, amax)
end
ContinuousActionSet{T}(amin::T, amax::T) = ContinuousActionSet{T}(amin, amax)

Base.length(aset::ContinuousActionSet) = length(aset.amin)
Base.rand{T<:Number}(aset::ContinuousActionSet{T}) = rand() * (aset.amax - aset.amin) + aset.amin
Base.rand{T<:AbstractVector}(aset::ContinuousActionSet{T}) = rand(length(aset)) .* (aset.amax - aset.amin) + aset.amin

Base.in{T<:Number}(x::Number, aset::ContinuousActionSet{T}) = aset.amin <= x <= aset.amax
Base.in{T<:AbstractVector}(x::AbstractVector, aset::ContinuousActionSet{T}) =
    length(x) == length(aset) && all(aset.amin .<= x .<= aset.amax)

# choose from discrete actions
immutable DiscreteActionSet{T} <: AbstractActionSet
	actions::T
end
Base.rand(aset::DiscreteActionSet) = rand(aset.actions)
Base.in(x, aset::DiscreteActionSet) = x in aset.actions
Base.length(aset::DiscreteActionSet) = length(aset.actions)
Base.getindex(aset::DiscreteActionSet, i::Int) = aset.actions[i]


# several action sets of varying types
immutable MultiActionSet{T<:Tuple} <: AbstractActionSet
    asets::T
end

MultiActionSet(asets::AbstractActionSet...) = MultiActionSet(asets)

Base.rand(::Type{Vector}, aset::MultiActionSet) = [rand(i) for i in aset.asets]
Base.rand(::Type{Tuple}, aset::MultiActionSet) = ntuple(i->rand(aset.asets[i]), length(aset.asets))
Base.rand(aset::MultiActionSet) = rand(Vector, aset)

Base.in(x, aset::MultiActionSet) = all(map(in, x, aset.asets))

# semantics for this one aren't very clear, so skip it for now
# Base.length(aset::MultiActionSet) = reduce(+, 0, map(length, aset.asets))

# ----------------------------------------------------------------
# ----------------------------------------------------------------
# Implement this interface for a new environment

abstract AbstractEnvironment


"""
`reset!(env)`

Reset an environment.
"""
function reset! end


"""
r, s′ = step!(env, s, a)

Move the simulation forward, collecting a reward and getting the next state.
"""
function step! end


# note for developers: you should also implement Base.done(env) for episodic environments
Base.done(env::AbstractEnvironment) = false


"""
`A′ = actions(env, s′)`

Return a list/set/description of valid actions from state `s′`.
"""
actions(env::AbstractEnvironment) = actions(env, state(env))


# note for developers: you don't need to implement these if you have state/reward fields

"""
`s = state(env)`

Return the current state of the environment.
"""
state(env::AbstractEnvironment) = env.state

"""
`r = reward(env)`

Return the current reward of the environment.
"""
reward(env::AbstractEnvironment) = env.reward


# ----------------------------------------------------------------
# Implement this interface for a new policy

abstract AbstractPolicy

"""
`a′ = action(policy, r, s′, A′)`

Take in the last reward `r`, current state `s′`, and set of valid actions `A′ = actions(env, s′)`,
then return the next action `a′`.

Note that a policy could do a 'sarsa-style' update simply by saving the last state and action `(s,a)`.
"""
function action end


# ----------------------------------------------------------------
# concrete implementations

include("episodes.jl")
include("states.jl")
include("policy.jl")

include("envs/cartpole.jl")
include("envs/pendulum.jl")

# ----------------------------------------------------------------

end # module
