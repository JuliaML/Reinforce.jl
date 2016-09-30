
type RandomPolicy <: AbstractPolicy end
action(policy::RandomPolicy, r, s′, A′) = rand(A′)

# ------------------------------------------------------------------

# NOTE: see http://qwone.com/~jason/writing/multivariateNormal.pdf
#   for derivations of gradients wrt μ and Λ:
#       ∇ log π(a | s) = ∇ log π(a | ϕ)

# TODO: the policy should be separated from the generating distribution

"""
A StochasticPolicy is parameterized by the action set from which it chooses actions.

More specifically, it samples from a distribution D(ϕ), where ϕ contains all sufficient statistics.
In the case of a MultivariateNormal N(μ, Σ):
    Σ = ΛΛ'   (Λ is the cholesky decomp to ensure Σ is positive definite)
    ϕ := vcat(μ, vec(Λ))
    ϕ = f(s)

So ϕ is the output of our transformation f(s)... mapping state s to sufficient statistics μ/Λ.
We then sample from our multivariate distribution: z ~ N(μ, ΛΛ')
and (deterministically) map that sample to actions: a(z) = project(logistic(z) --> A)

Since the last step is deterministic: P(a(z)|z,s) = P(a(z)|z)P(z|s) = 1 * P(z|s)
So:
    ∇log π(a|θ) = ∇log P(z|ϕ) * (∂ϕ / ∂θ)

If η(θ,s) := E[R|π,s] is the expected return of a policy (π is the generating distribution for actions given s)
    ∇ηₜ(θ,s) = Rₜ ∑ (∂log P(z|ϕ) / ∂ϕ) * (∂ϕ / ∂θ)

tl;dr The gradient wrt transformation params θ can be broken
"""
type StochasticPolicy{T, ASET <: AbstractSet, DIST <: Distribution, TRANS <: Transformation} <: AbstractPolicy
    dist::DIST
    A::ASET
    f::TRANS                        # This transformation maps: s --> ϕ
    act::Activation{:logistic,T}    # After sampling from dist(μ,Σ), squash to [0,1].
                                    # The actions are activation outputs mapped to [lo,hi]
    actions::Vector{T}
end

function StochasticPolicy{T, DIST<:MvNormal}(::Type{T}, ::Type{DIST}, A::AbstractSet, f::Transformation)
    n = length(A)
    nϕ = n * (n+1)  # size of μ + size of Z
    StochasticPolicy(
        MultivariateNormal(zeros(T,n), eye(T,n)),
        A,
        f,
        Activation{:logistic,T}(n),
        zeros(T,n)
    )
end
function StochasticPolicy(A::AbstractSet, f::Transformation)
    StochasticPolicy(Float64, MultivariateNormal, A, f)
end

function Reinforce.action(policy::StochasticPolicy, r, s′, A′)
    n = length(A′)
    dist = policy.dist

    # TODO: do something useful with reward (using eligibility trace, update params?)

    # apply the mapping from state to suff. stats
    ϕ = transform!(policy.f, s′)

    # update μ with the first n input values
    dist.μ[:] = view(ϕ, 1:n)

    # next update the stored cholesky decomp Λ in: Σ = ΛΛ'
    # this allows us to ensure positive definiteness on Σ
    dist.Σ.chol.factors[:] = view(ϕ, n+1:length(ϕ))

    # sample from the distribution, then squash to [0,1]
    # z ~ N(μ, Σ)
    # â = logistic(z)
    z = rand(dist)
    â = transform!(policy.act, z)

    # project our squashed sample onto into the action space to get our actions
    # a = (â --> [lo,hi])
    for i in 1:n
        lo, hi = A′.lo[i], A′.hi[i]
        policy.actions[i] = lo + â[i] * (hi - lo)
    end
    policy.actions
end
