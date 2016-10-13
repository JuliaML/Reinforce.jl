
type RandomPolicy <: AbstractPolicy end
action(policy::RandomPolicy, r, s′, A′) = rand(A′)

# ------------------------------------------------------------------

#=
See http://www.breloff.com/DeepRL-OnlineGAE/ for details/explanation of OnlineGAE

Samples from a distribution D(ϕ), where ϕ contains all sufficient statistics.
In the case of a MultivariateNormal N(μ, Σ):
    Σ = U'U   (we output U to ensure Σ is positive definite)
    ϕ := vcat(μ, vec(U))
    ϕ(s,θ) is a learnable transformation from states to sufficient statistics

We then sample from our multivariate distribution:
    z ~ D(ϕ)
and (deterministically) map that sample to actions (i.e. project squashed samples `z` into the action space `A`):
    a = π(z)    or    a ~ π(s,θ)

---

Notes:

The gradient wrt transformation params θ can be broken into the components of ϕ which map to μ/U.

Assuming the first and last steps are deterministic:
    P(a|z,s,θ) = P(a|z) * P(z|ϕ) * P(ϕ|s,θ)
               = 1 * P(z|ϕ) * 1
               = P(z|ϕ)
So:
    ∇log P(a|z,s,θ) = ∇log P(z|ϕ)
=#

"""
Maps state to value: V(s)

We update the learnable parameters using gradient vector δ
"""
type ValueCritic{T,TRANS<:Learnable} <: Learnable
    trans::TRANS  # nS --> 1  transformation which outputs a value V(s) for state s
    γ::T            # discount
    lastv::T       # V(s)
    δ::T          # TD(0) delta: δ = r + γV(s′) - V(s)
    ValueCritic(trans::TRANS, γ::T) = new(trans, γ, zero(T), zero(T))
end

function transform!(critic::ValueCritic, s)
    critic.lastv = output_value(critic.trans)[1]
    v(s)
end

# give reward r, compute output grad: δ = r + γV(s′) - V(s)
# then backprop to get ∇θ
function grad!(critic::ValueCritic, r)
    Vs′ = output_value(critic.trans)[1]
    Vs = critic.lastv
    critic.δ = r + critic.γ * Vs′ - Vs
    output_grad(critic.trans)[1] = critic.δ
    grad!(critic.trans)
end


"""
Online Generalized Advantage Estimation for Actor-Critic Reinforcement Learning

Transforms states (input) to actions (output) using learnable parameters θ

We assume the general form of mapping states (s) to actions (a):
    s --> ϕ(s,θ) --> D(ϕ) --> a

ϕ is:
    - a learnable transformation with learnable parameters θ
    - the sufficient statistics of distribution D
    - some concatenated combination of μ/U/σ for multivariate normals
"""
type OnlineGAE{T      <: Number,
               ASET   <: AbstractSet,
               PHI    <: Learnable,
               DIST   <: MvNormalTransformation,
               CRITIC <: ValueCritic,
               P      <: Params
              } <: AbstractPolicy
    A::ASET           # the action space
    ϕ::PHI            # learnable transformation to output sufficient statistics of D
    D::DIST           # generative transformation for sampling inputs to a
    critic::CRITIC    # the critic... wraps a learnable estimating value function V(s)
    γ::T              # the discount for the critic
    λ::T              # the extra discount for the actor
    ϵ::Vector{T}      # eligibility traces for the learnable params θ in transformation ϕ
    t::Int            # current timestep
    ∇logP::Vector{T}  # policy gradient: ∇log P(a | s) == ∇log P(z | ϕ)
    params::P         # the combined parameters from the actor transformation ϕ and the critic transformation
end

function OnlineGAE{T}(A::AbstractSet,
                      ϕ::Learnable,
                      D::MvNormalTransformation,
                      critic_trans::Learnable,
                      γ::T,
                      λ::T)
    # connect transformations, init the critic
    link_nodes!(ϕ, D)
    critic = ValueCritic(critic_trans, γ)
    ϵ = zeros(T, params_length(ϕ))
    ∇logP = zeros(T, input_length(D))
    params = consolidate_params(T, ϕ, critic_trans)
    OnlineGAE(A, ϕ, D, critic, γ, λ, ϵ, 1, ∇logP, params)
end

function Reinforce.action(π::OnlineGAE, r, s′, A′)
    # sample z ~ N(μ,Σ) which is determined by ϕ
    ϕ = transform!(π.ϕ, s′)
    z = transform!(π.D)

    # project our squashed sample onto into the action space to get our actions
    # a = (â --> [lo,hi])
    a = A′.lo .+ z .* (A′.hi .- A′.lo)

    # Note: the rest of the function populates parameter gradients for the actor and critic

    # update the critic
    transform!(π.critic, s′)
    grad!(π.critic, r)
    # TODO: update the params??  do we do this outside of action,
    #   by wrapping the params of the actor and critic in one vector??

    #=
    update the actor using the OnlineGAE formulas:
        ϵₜ = (γλ)ϵₜ₋₁ + ∇
        ĝₜ = δₜϵₜ
    =#

    # update the grad-log-prob of distribution D
    Transformations.gradlogprob!(π.∇logP, π.D)
    # grad!(π.D)
    # ∇logP = input_grad(π.D)

    # we use ∇logP to update the eligibility trace ϵ
    # then we set the gradient estimate: ĝ = δϵ
    γλ = π.γ * π.λ
    ϵ = π.ϵ
    ∇ = grad(ϕ)
    for i=1:length(ϵ)
        ϵ[i] = γλ * ϵ[i] + ∇logP
        ∇[i] = π.critic.δ * ϵ[i]
    end

    a
end

# ----------------------------------------------------------------------

# "Online Generalized Advantage Estimation for Actor-Critic Reinforcement Learning"
# type OnlineGAE{ASET <: AbstractSet, Π <: Learnable, VAL} <: AbstractPolicy
#     π::Π        # the action function: a ~ π(s,θ)
#     V::VAL      # learnable value function: V(s)
#     # f::TRANS                        # This transformation maps: s --> ϕ
#     # act::Activation{:logistic,T}    # After sampling from dist(μ,Σ), squash to [0,1].
#     #                                 # The actions are activation outputs mapped to [lo,hi]
#     # actions::Vector{T}
# end
#
# function OnlineGAE{T, DIST<:MvNormal}(::Type{T}, ::Type{DIST}, A::AbstractSet, f::Transformation)
#     n = length(A)
#     nϕ = n * (n+1)  # size of μ + size of Z
#     OnlineGAE(
#         MultivariateNormal(zeros(T,n), eye(T,n)),
#         A,
#         f,
#         Activation{:logistic,T}(n),
#         zeros(T,n)
#     )
# end
# function OnlineGAE(A::AbstractSet, f::Transformation)
#     OnlineGAE(Float64, MultivariateNormal, A, f)
# end
#
# function Reinforce.action(policy::OnlineGAE, r, s′, A′)
#     n = length(A′)
#     # dist = policy.dist
#
#     # TODO: do something useful with reward (using eligibility trace, update params?)
#
#     # # apply the mapping from state to suff. stats
#     # ϕ = transform!(policy.f, s′)
#     #
#     # # update μ with the first n input values
#     # dist.μ[:] = view(ϕ, 1:n)
#     #
#     # # next update the stored cholesky decomp U in: Σ = U'U
#     # # this allows us to ensure positive definiteness on Σ
#     # dist.Σ.chol.factors[:] = view(ϕ, n+1:length(ϕ))
#
#     # sample from the distribution, then squash to [0,1]
#     # z ~ N(μ, Σ)
#     # â = logistic(z)
#     z = rand(dist)
#     â = transform!(policy.act, z)
#
#     # project our squashed sample onto into the action space to get our actions
#     # a = (â --> [lo,hi])
#     for i in 1:n
#         lo, hi = A′.lo[i], A′.hi[i]
#         policy.actions[i] = lo + â[i] * (hi - lo)
#     end
#     policy.actions
# end
