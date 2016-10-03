
type RandomPolicy <: AbstractPolicy end
action(policy::RandomPolicy, r, s′, A′) = rand(A′)

# ------------------------------------------------------------------

# NOTE: see http://qwone.com/~jason/writing/multivariateNormal.pdf
#   for derivations of gradients wrt μ and Λ:
#       ∇log P(z|ϕ)

# TODO: the policy should be separated from the generating distribution

"""
A StochasticPolicy is parameterized by the action set from which it chooses actions.

More specifically, it samples from a distribution D(ϕ), where ϕ contains all sufficient statistics.
In the case of a MultivariateNormal N(μ, Σ):
    Σ = ΛΛ'   (we output Λ to ensure Σ is positive definite)
    ϕ := vcat(μ, vec(Λ))
    ϕ(s,θ) is a learnable transformation from states to sufficient statistics

We then sample from our multivariate distribution:
    z ~ D(ϕ)
and (deterministically) map that sample to actions (i.e. project squashed samples `z` into the action space `A`):
    a = π(z)    or    a ~ π(s,θ)

---

Define:
    a ~ π(s,θ)        is the generating distribution for actions given s
    τ ~ episode(π)    is the full trajectory of an episode given π: (s₀, a₀, r₀, s₁, a₁, ..., rₜ₋₁, sₜ)
    R(τ)              is the total return of episode τ: ∑rᵢ
    η = E[R(τ)]       is the expected total return when following policy π

---

We want to estimate the gradient:
    ∇ = ∂η/∂θ

Note: The gradient wrt transformation params θ can be broken into the components of ϕ which map to μ/Λ.
Note: Unless stated, sums are {t = 0 --> T-1}

Using the policy gradient theorem:
    Ĝ(τ) = R(τ) ∇log P(τ|θ)

where:
    ∇log P(τ|θ) = ∇ ∑ log π(a|s,θ)

But that's annoying.  We could instead estimate the gradient of one reward:
    ĝₜ = rₜ ∑₀₋ₜ ∇log π(a|s,θ)
or of one time step for all subsequent rewards:
    ĝₜ = ∑ [ ∇log π(a|s,θ) (∑{>t} rₜ) ]

---

Notes:

Assuming the first and last steps are deterministic:
    P(a|z,s,θ) = P(a|z) * P(z|ϕ) * P(ϕ|s,θ)
               = 1 * P(z|ϕ) * 1
               = P(z|ϕ)
So:
    ∇log P(a|z,s,θ) = ∇log P(z|ϕ)
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
