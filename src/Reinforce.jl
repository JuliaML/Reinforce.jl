
__precompile__(true)

module Reinforce

using Reexport
@reexport using StatsBase
using Distributions
using RecipesBase

@reexport using LearnBase
import LearnBase: learn!, transform!, grad!, params, grad

using LearningStrategies
import LearningStrategies: pre_hook, iter_hook, finished, post_hook

export
	AbstractEnvironment,
	reset!,
	step!,
	reward,
	state,
    finished,
	actions,
    ismdp,

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
    Episodes,
    run_episode


# ----------------------------------------------------------------
# Implement this interface for a new environment

abstract type AbstractEnvironment end


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

"""
`ismdp(env) --> bool`

An environment may be fully observable (MDP) or partially observable (POMDP).  In the case of a partially observable environment, the state `s` is really an observation `o`.  To maintain consistency, we call everything a state, and assume that an environment is free to maintain additional (unobserved) internal state.

The `ismdp` query returns true when the environment is MDP, and false otherwise.
"""
ismdp(env::AbstractEnvironment) = false


# ----------------------------------------------------------------
# Implement this interface for a new policy

abstract type AbstractPolicy end

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

struct KeyboardAction
    key
end

mutable struct KeyboardActionSet{T} <: AbstractSet{T}
    keys::Vector
end

LearnBase.randtype(s::KeyboardActionSet) = KeyboardAction
Base.rand(s::KeyboardActionSet) = KeyboardAction(rand(s.keys))
Base.in(a::KeyboardAction, s::KeyboardActionSet) = a.key in s.keys
Base.length(s::KeyboardActionSet) = 1

# ----------------------------------------------------------------
# a mouse/pointer action space

struct MouseAction
    x::Int
    y::Int
    button::Int
end

mutable struct MouseActionSet{T} <: AbstractSet{T}
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
