
__precompile__()

module Reinforce

using Reexport
@reexport using StatsBase
using Distributions
@reexport using LearnBase
using RecipesBase
using ValueHistories
using Parameters
using StochasticOptimization
using Transformations

import LearnBase: learn!, transform!, grad!
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
	action,

	AbstractState,
	StateVector,
	History,
	state!,

	Episode,
    EpisodeLearner,
	episode!


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

include("episodes.jl")
include("states.jl")
include("policy.jl")
include("solvers.jl")

include("envs/cartpole.jl")
include("envs/pendulum.jl")

# ----------------------------------------------------------------

end # module
