import Reinforce: action, actions, finished, ismdp, maxsteps, reset!, reward, state, step!

###############################################################################
#  Example Usage
###############################################################################

mutable struct FooEnv <: AbstractEnvironment
  s::Int  # state
  r::Int  # reward

  FooEnv() = new(1, 0)
end

state(env::FooEnv) = env.s
reward(env::FooEnv) = env.r
reset!(env::FooEnv) = (env.s = 1; env.r = 0; env)
step!(env::FooEnv, s, a) = (env.s += 1; env.r = -1; (env.r, env.s))
maxsteps(env::FooEnv) = 3
actions(env::FooEnv, s′) = [1, 2, 3]

struct FooPolicy <: AbstractPolicy
end

action(π::FooPolicy, r, s, A) = rand(A)

# Iterating a Episode:
#
# ep = Episode(FooEnv(), FooPolicy())
# for (s, a, r, s′) ∈ ep
#   ...
# end
