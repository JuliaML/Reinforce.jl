
type RandomPolicy <: AbstractPolicy end
action(policy::RandomPolicy, r, s′, A′) = rand(A′)

# ------------------------------------------------------------------

#=
See http://www.breloff.com/DeepRL-OnlineGAE/ for details/explanation of OnlineGAE

Samples from a distribution D(ϕ), where ϕ contains all sufficient statistics.
In the case of a MultivariateNormal N(μ, Σ):
    Σ = ΛΛ'   (we output Λ to ensure Σ is positive definite)
    ϕ := vcat(μ, vec(Λ))
    ϕ(s,θ) is a learnable transformation from states to sufficient statistics

We then sample from our multivariate distribution:
    z ~ D(ϕ)
and (deterministically) map that sample to actions (i.e. project squashed samples `z` into the action space `A`):
    a = π(z)    or    a ~ π(s,θ)

---

Notes:

The gradient wrt transformation params θ can be broken into the components of ϕ which map to μ/Λ.

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
