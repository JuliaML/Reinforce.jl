
__precompile__()

module Reinforce

using Reexport
@reexport using StatsBase
using Distributions
@reexport using LearnBase
using RecipesBase
# using ValueHistories
# using Parameters
using StochasticOptimization
using Transformations
using PenaltyFunctions
import OnlineStats: Mean, Variances, Weight, BoundedEqualWeight

import LearnBase: learn!, transform!, grad!, params, grad
import StochasticOptimization: pre_hook, iter_hook, finished, post_hook

export
	AbstractEnvironment,
	reset!,
	step!,
	reward,
	state,
	actions,

	AbstractPolicy,
	RandomPolicy,
    OnlineGAE,
    OnlineActorCritic,
    EpisodicActorCritic,
	action,

	AbstractState,
	StateVector,
	History,
	state!,

	Episode,
    Episodes
	# episode!


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
finished(env::AbstractEnvironment, s′) = false


"""
`A′ = actions(env, s′)`

Return a list/set/description of valid actions from state `s′`.
"""
# actions(env::AbstractEnvironment) = actions(env, state(env))
function actions end


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

# include("episodes.jl")
include("episodes/iterators.jl")
include("states.jl")
include("policies/policies.jl")
include("solvers.jl")

include("envs/cartpole.jl")
include("envs/pendulum.jl")
include("envs/mountain_car.jl")

# ----------------------------------------------------------------
# a keyboard action space

immutable KeyboardAction
    key
end

type KeyboardActionSet <: AbstractSet
    keys::Vector
end

LearnBase.randtype(s::KeyboardActionSet) = KeyboardAction
Base.rand(s::KeyboardActionSet) = KeyboardAction(rand(s.keys))
Base.in(a::KeyboardAction, s::KeyboardActionSet) = a.key in s.keys
Base.length(s::KeyboardActionSet) = 1

# ----------------------------------------------------------------
# a mouse/pointer action space

immutable MouseAction
    x::Int
    y::Int
    button::Int
end

type MouseActionSet <: AbstractSet
    screen_width::Int
    screen_height::Int
    button::DiscreteSet{Vector{Int}}
end

LearnBase.randtype(s::MouseActionSet) = MouseAction
Base.rand(s::MouseActionSet) = MouseAction(rand(1:s.screen_width), rand(1:s.screen_height), rand(s.button))
Base.in(a::MouseAction, s::MouseActionSet) = a.x in 1:s.screen_width && a.y in 1:s.screen_height && a.button in s.button
Base.length(s::MouseActionSet) = 1


# ----------------------------------------------------------------

end # module
