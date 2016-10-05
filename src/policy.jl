
type RandomPolicy <: AbstractPolicy end
action(policy::RandomPolicy, r, s′, A′) = rand(A′)

# ------------------------------------------------------------------

# NOTE: see http://qwone.com/~jason/writing/multivariateNormal.pdf
#   for derivations of gradients wrt μ and Λ:
#       ∇log P(z|ϕ)

# TODO: the policy should be separated from the generating distribution



#=
Note: Much of the notation below tries to follow that of John Schulman et al in High-Dimensional Continuous
    Control using Generalized Advantage Estimation 2016


A OnlineGAE is parameterized by the action set from which it chooses actions.

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
    a ~ π(s,θ)          is the generating distribution for actions given s
    τ ~ episode(π)      is the full trajectory of an episode given π: (s₀, a₀, r₀, s₁, a₁, ..., rₜ₋₁, sₜ)
    R(τ)                is the total return of episode τ: ∑rᵢ
    η(π) = E[R(τ)]      is the expected total return of a trajectory when following policy π

---

We want to estimate the gradient:
    ∇ = ∂η/∂θ

Note: The gradient wrt transformation params θ can be broken into the components of ϕ which map to μ/Λ.
Note: Unless stated, sums are {t = 0 --> T-1}

Using the policy gradient theorem:
    Ĝ(τ) = R(τ) ∇log P(τ|θ)

where:
    ∇log P(τ|θ) = ∇ ∑ log π(a|s,θ)

But that's annoying.  We could instead estimate the gradient of one reward (good for online calcs with eligibility traces):
    ĝₜ = rₜ ∑₀₋ₜ ∇log π(a|s,θ)                 (1)

We can subtract a baseline bₜ from rₜ without adding bias to ĝ:
    ĝ = (rₜ - bₜ) ∑₀₋ₜ ∇log π(a|s,θ)

We could possibly use a single-return value function:
    bₜ = E[rₜ|sₜ]

---

We could also reorganize to one time step for all subsequent rewards:
    ĝₜ = ∑ [ ∇log π(a|s,θ) (E(∑{>t} rₜ) - bₜ) ]

which is near-optimal when bₜ ≈ V(sₜ) (the value function).

If we also note that:
    Q(sₜ,aₜ) = E(∑{>t} rₜ)
and that the advantage function is:
    A(sₜ,aₜ) = Q(sₜ,aₜ) - V(sₜ)

then it follows that:
    ĝ = ∑ [ ∇log π(aₜ|sₜ,θ) A(sₜ,aₜ) ]

and we can use discounted rewards with discount factor γ:
    ĝ = ∑ [ ∇log π(aₜ|sₜ,θ) Aᵞ(sₜ,aₜ) ]         (2)

---

Generalized Advantage Estimation (GAE) concerns itself with estimating "good" values of Aᵞ(sₜ,aₜ)
in equation 2, where we can trade off bias against variance in our estimation of ĝ.

Start with the TD (Bellman) residual:
    δₜ = rₜ + γVᵞ(sₜ+₁) - Vᵞ(sₜ)

GAE can be simply summarized with the estimate of the advantage:
    Âₜ = δₜ + (γλ)δₜ+₁ + (γλ)²δₜ+₂ + (γλ)³δₜ+₃ + ...
       = ∑ (γλ)ˡδₜ+ₗ

which is an estimate of the next step and discounted subsequent steps Bellman residuals,
where the discounting is controlled with parameter λ.  When λ is 0, it reverts to the TD residual,
and when λ is 1, it is an unbiased estimate for ĝ.  This allows us to precisely control the bias/variance
tradeoff when estimating ĝ:
    ĝᵞ = ∑ [ ∇log π(aₜ|sₜ,θ) ∑ (γλ)ˡδₜ+ₗ ]        (3)

---

Notes:

Assuming the first and last steps are deterministic:
    P(a|z,s,θ) = P(a|z) * P(z|ϕ) * P(ϕ|s,θ)
               = 1 * P(z|ϕ) * 1
               = P(z|ϕ)
So:
    ∇log P(a|z,s,θ) = ∇log P(z|ϕ)
=#

"""
Transforms states (input) to actions (output) using learnable parameters θ

We assume the general form of mapping states (s) to actions (a):
    s --> ϕ(s,θ) --> D(ϕ) --> a

ϕ is:
    - a learnable transformation with learnable parameters θ
    - the sufficient statistics of distribution D
    - some concatenated combination of μ/Λ/σ for multivariate normals
"""
type StochasticPolicy{ASET,PHI,DIST,ACT} <: Learnable
    A::ASET     # the action space
    ϕ::PHI      # learnable transformation to output sufficient statistics of D
    D::DIST     # generative transformation for sampling inputs to a
    a::ACT      # transformation mapping the output of D to action space A
end

function StochasticPolicy(A::AbstractSet, ϕ::Learnable, D::MvNormalTransformation)
    #= TODO:
        -
    =#
end

# compute: ∇log π(s,a)
function gradlog(π::StochasticPolicy)
end

function Reinforce.action(π::StochasticPolicy, r, s′, A′)
    ϕ = π.ϕ(s′)
    z = π.D()
    a = π.a()

    # TODO: learn/update params θ

    a
end

# ----------------------------------------------------------------------

"Online Generalized Advantage Estimation for Actor-Critic Reinforcement Learning"
type OnlineGAE{ASET <: AbstractSet, Π <: Learnable, VAL} <: AbstractPolicy
    π::Π        # the action function: a ~ π(s,θ)
    V::VAL      # learnable value function: V(s)
    # f::TRANS                        # This transformation maps: s --> ϕ
    # act::Activation{:logistic,T}    # After sampling from dist(μ,Σ), squash to [0,1].
    #                                 # The actions are activation outputs mapped to [lo,hi]
    # actions::Vector{T}
end

function OnlineGAE{T, DIST<:MvNormal}(::Type{T}, ::Type{DIST}, A::AbstractSet, f::Transformation)
    n = length(A)
    nϕ = n * (n+1)  # size of μ + size of Z
    OnlineGAE(
        MultivariateNormal(zeros(T,n), eye(T,n)),
        A,
        f,
        Activation{:logistic,T}(n),
        zeros(T,n)
    )
end
function OnlineGAE(A::AbstractSet, f::Transformation)
    OnlineGAE(Float64, MultivariateNormal, A, f)
end

function Reinforce.action(policy::OnlineGAE, r, s′, A′)
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
