
# WARNING: This should not be used as-is.  It's unfinished experimentation.

using Transformations
using StochasticOptimization
using PenaltyFunctions

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
mutable struct ValueCritic{T,TRANS<:Learnable} <: Learnable
    trans::TRANS  # nS --> 1  transformation which outputs a value V(s) for state s
    γ::T            # discount
    lastv::T       # V(s)
    δ::T          # TD(0) delta: δ = r + γV(s′) - V(s)
    # lastδ::T
end

function ValueCritic(::Type{T}, trans::Learnable, γ::T) where T
    ValueCritic{T,typeof(trans)}(trans, γ, zero(T), zero(T))
end

# function transform!(critic::ValueCritic, s::AbstractArray)
#     # critic.lastv = output_value(critic.trans)[1]
#     transform!(critic.trans, s)
# end

# give reward r, compute output grad: δ = r + γV(s′) - V(s)
# then backprop to get ∇θ
function grad!(critic::ValueCritic{T}, r::Number) where T
    Vs′ = output_value(critic.trans)[1]
    Vs = critic.lastv
    # critic.lastδ = critic.δ

    # the loss function is L2 loss with "truth" (r+λVₛ′) and "estimate" Vₛ
    # the output gradient is:
    #   ∂(δ²/2)/∂Vₛ′

    # this is the discounted return δ:
    critic.δ = r + critic.γ * Vs′ - Vs
    output_grad(critic.trans)[1] = -critic.δ
    # output_grad(critic.trans)[1] = -critic.γ * critic.δ

    # # this tries to solve for the average future reward
    # critic.δ = critic.γ * r + (one(T) - critic.γ) * Vs′ - Vs
    # output_grad(critic.trans)[1] = -critic.γ * critic.δ


    # critic.lastv = Vs′
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
mutable struct OnlineGAE{T      <: Number,
               ASET   <: AbstractSet,
               PHI    <: Learnable,
               DIST   <: MvNormalTransformation,
               CRITIC <: ValueCritic,
            #    P      <: Params,
               PEN    <: Penalty,
               AL     <: LearningStrategy,
               CL     <: LearningStrategy
              } <: Learnable
    A::ASET           # the action space
    ϕ::PHI            # learnable transformation to output sufficient statistics of D
    D::DIST           # generative transformation for sampling inputs to a
    critic::CRITIC    # the critic... wraps a learnable estimating value function V(s)
    γ::T              # the discount for the critic
    λ::T              # the extra discount for the actor
    ϵ::Vector{T}      # eligibility traces for the learnable params θ in transformation ϕ
    # a::Vector{T}      # the action
    # t::Int            # current timestep
    # ∇logP::Vector{T}  # policy gradient: ∇log P(a | s) == ∇log P(z | ϕ)
    # lastr::T          # most recent return
    # params::P         # the combined parameters from the actor transformation ϕ and the critic transformation
    penalty::PEN      # a penalty to add to param gradients
    actor_learner::AL
    critic_learner::CL
end

function OnlineGAE(A::AbstractSet,
                   ϕ::Learnable,
                   D::MvNormalTransformation,
                   critic_trans::Learnable,
                   γ::T,
                   λ::T,
                   actor_learner::LearningStrategy,
                   critic_learner::LearningStrategy;
                   penalty::Penalty = NoPenalty()) where T
    # connect transformations, init the critic
    link_nodes!(ϕ, D)
    critic = ValueCritic(T, critic_trans, γ)
    np = params_length(ϕ)
    ϵ = zeros(T, np)
    # ∇logP = zeros(T, np)
    # params = consolidate_params(T, ϕ, critic_trans)
    pre_hook(actor_learner, ϕ)
    pre_hook(critic_learner, critic.trans)
    OnlineGAE(A, ϕ, D, critic, γ, λ, ϵ, penalty, actor_learner, critic_learner)
end

# don't do anything here... we'll update later
LearnBase.update!(π::OnlineGAE, ::Void) = return

function Reinforce.reset!(π::OnlineGAE{T}) where T
    fill!(π.ϵ, zero(T))
    # π.critic.lastv = 0
    # pre_hook(π.actor_learner, π.ϕ)
    # pre_hook(π.critic_learner, π.critic.trans)
end

function Reinforce.action(π::OnlineGAE, r, s′, A′)
    # sample z ~ N(μ,Σ) which is determined by ϕ
    transform!(π.ϕ, s′)
    z = transform!(π.D)

    # project our squashed sample onto into the action space to get our actions
    # a = (â --> [lo,hi])
    a = A′.lo .+ logistic.(z) .* (A′.hi .- A′.lo)
    if !(a in A′)
        warn("a=$a not in A=$(A′)")
        a = rand(A′)
    end
    a
end


function learn!(π::OnlineGAE, s, a, r, s′)
    # update the critic.  we use the current model to get the lastv == Vₛ
    # as well as the current == Vₛ′
    π.critic.lastv = transform!(π.critic.trans, s)[1]
    transform!(π.critic.trans, s′)
    grad!(π.critic, r)
    # π.lastr = r
    t = π.critic.trans
    addgrad!(grad(t), π.penalty, params(t))

    learn!(t, π.critic_learner, nothing)

    #=
    update the actor using the OnlineGAE formulas:
        ϵₜ = (γλ)ϵₜ₋₁ + ∇
        ĝₜ = δₜϵₜ

    note: we use the grad-log-prob from the last timestep, since we
    can't update until we compute the critic's δ, which depends
    on the next timestep
    =#

    # update the grad-log-prob of distribution D, and store that for the next timestep
    # NOTE: grad(π.ϕ) now contains the grad-log-prob of this timestep... but we don't use this
    #   until the next timestep
    grad!(π.D)
    grad!(π.ϕ)

    # we use last timestep's ∇logP to update the eligibility trace of the last timestep ϵ
    γλ = π.γ * π.λ
    ϵ = π.ϵ
    ∇ = grad(π.ϕ)
    addgrad!(∇, π.penalty, params(π.ϕ))
    for i=1:length(ϵ)
        ϵ[i] = γλ * ϵ[i] + ∇[i]
    end

    # copy!(π.∇logP, grad(π.ϕ))

    # overwrite the gradient estimate: ĝ = δϵ
    δ = π.critic.δ
    for i=1:length(ϵ)
        ∇[i] = -δ * ϵ[i]
    end

    # # add the penalty to the gradient
    # addgrad!(∇, π.penalty, params(π.ϕ))

    # learn the actor
    learn!(π.ϕ, π.actor_learner, nothing)
end
