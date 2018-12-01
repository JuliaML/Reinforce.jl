module MultiArmedBanditEnv

using ..Reinforce
using Distributions

export MultiArmedBandit

"""
The multi-armed bandit environment.

    MultiArmedBandit(k, n = 100; σ = 1)

- `k` is the number of available arms.
  The reward distribution for each arms is a Normal distribution centered
  at the range of [-1, 1] and the standard deviation is 1.
- `n` is the max steps for each episode.
- `σ` controls the the standard deviation for all Normal distribution.

    MultiArmedBandit(x::Vector{<:Distribution}...)

In case that you want to other distributions as the reward distribution.
"""
mutable struct MultiArmedBandit{K,D<:Vector} <: AbstractEnvironment
  arms::D
  n::Int  # max steps
  r::Float64
end

function MultiArmedBandit(k::Int, n::Int = 1000; σ::Real = 1)
  k ≤ 0 && throw(ArgumentError("k must > 0"))
  arms = map(i -> Normal(rand(Uniform(-1, 1)), σ), 1:k)
  MultiArmedBandit{k,typeof(arms)}(arms, n, 0)
end

function MultiArmedBandit(x::Vararg{Distribution,N}) where {N}
  y = collect(x)
  MultiArmedBandit{N,typeof(y)}(y, N, 0)
end

Reinforce.state(::MultiArmedBandit) = nothing
Reinforce.reward(env::MultiArmedBandit) = env.r
Reinforce.reset!(env::MultiArmedBandit) = (env.r = 0; env)
Reinforce.actions(::MultiArmedBandit{K}, s) where {K} = Base.OneTo(K)
Reinforce.step!(env::MultiArmedBandit, s, a::Int) = (env.r = rand(env.arms[a]); (env.r, nothing))
Reinforce.maxsteps(env::MultiArmedBandit) = env.n
Reinforce.ismdp(::MultiArmedBandit) = true

end  # module MultiArmedBanditEnv
