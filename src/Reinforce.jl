
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

    function ContinuousActionSet{S<:AbstractVector}(amin::S, amax::S)
        if !(length(amin) == length(amax))
            error("For multi-valued continuous action sets, min and max value must have same length")
        end
        new(amin, amax)
    end

    ContinuousActionSet{S<:Number}(amin::S, amax::S) = new(amin, amax)
end

# outer constructor to handle dispatch to above inner constructors
ContinuousActionSet{T}(amin::T, amax::T) = ContinuousActionSet{T}(amin, amax)

Base.length(aset::ContinuousActionSet) = length(aset.amin)
Base.rand{T<:Number}(aset::ContinuousActionSet{T}) = rand() * (aset.amax - aset.amin) + aset.amin
Base.rand{T<:AbstractVector}(aset::ContinuousActionSet{T}) = rand(length(aset)) .* (aset.amax - aset.amin) + aset.amin

Base.in{T<:Number}(x::Number, aset::ContinuousActionSet{T}) = aset.amin <= x <= aset.amax
Base.in{T<:AbstractVector}(x::AbstractVector, aset::ContinuousActionSet{T}) =
    length(x) == length(aset) && all(aset.amin .<= x .<= aset.amax)

immutable DiscreteActionSet{T} <: AbstractActionSet
	actions::T
end
Base.rand(aset::DiscreteActionSet) = rand(aset.actions)
Base.in(x, aset::DiscreteActionSet) = x in aset.actions
Base.length(aset::DiscreteActionSet) = length(aset.actions)
Base.getindex(aset::DiscreteActionSet, i::Int) = aset.actions[i]

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
check_constraints(env::AbstractEnvironment, s, a) = return

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
