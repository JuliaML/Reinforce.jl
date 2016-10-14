
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
end

function ValueCritic{T}(::Type{T}, trans::Learnable, γ::T)
    ValueCritic{T,typeof(trans)}(trans, γ, zero(T), zero(T))
end

function transform!(critic::ValueCritic, s::AbstractArray)
    critic.lastv = output_value(critic.trans)[1]
    transform!(critic.trans, s)
end

# give reward r, compute output grad: δ = r + γV(s′) - V(s)
# then backprop to get ∇θ
function grad!(critic::ValueCritic, r::Number)
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
               P      <: Params,
               PEN    <: Penalty
              } <: Learnable
    A::ASET           # the action space
    ϕ::PHI            # learnable transformation to output sufficient statistics of D
    D::DIST           # generative transformation for sampling inputs to a
    critic::CRITIC    # the critic... wraps a learnable estimating value function V(s)
    γ::T              # the discount for the critic
    λ::T              # the extra discount for the actor
    ϵ::Vector{T}      # eligibility traces for the learnable params θ in transformation ϕ
    # t::Int            # current timestep
    # ∇logP::Vector{T}  # policy gradient: ∇log P(a | s) == ∇log P(z | ϕ)
    params::P         # the combined parameters from the actor transformation ϕ and the critic transformation
    penalty::PEN      # a penalty to add to param gradients
end

function OnlineGAE{T}(A::AbstractSet,
                      ϕ::Learnable,
                      D::MvNormalTransformation,
                      critic_trans::Learnable,
                      γ::T,
                      λ::T;
                      penalty::Penalty = NoPenalty())
    # connect transformations, init the critic
    link_nodes!(ϕ, D)
    critic = ValueCritic(T, critic_trans, γ)
    ϵ = zeros(T, params_length(ϕ))
    # ∇logP = zeros(T, input_length(D))
    params = consolidate_params(T, ϕ, critic_trans)
    OnlineGAE(A, ϕ, D, critic, γ, λ, ϵ, params, penalty)
end

# don't do anything here... we'll update during action
LearnBase.update!(π::OnlineGAE, ::Void) = return

function Reinforce.reset!{T}(π::OnlineGAE{T})
    fill!(π.ϵ, zero(T))
end

function Reinforce.action(π::OnlineGAE, r, s′, A′)
    # sample z ~ N(μ,Σ) which is determined by ϕ
    transform!(π.ϕ, s′)
    z = transform!(π.D)

    # project our squashed sample onto into the action space to get our actions
    # a = (â --> [lo,hi])
    a = A′.lo .+ logistic.(z) .* (A′.hi .- A′.lo)
    # @show a

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
    grad!(π.D)
    grad!(π.ϕ)
    ∇logP = grad(π.ϕ)

    # we use ∇logP to update the eligibility trace ϵ
    # then we overwrite it with the gradient estimate: ĝ = δϵ
    γλ = π.γ * π.λ
    ϵ = π.ϵ
    # ∇ = grad(π.ϕ)
    for i=1:length(ϵ)
        ϵ[i] = γλ * ϵ[i] + ∇logP[i]
        ∇logP[i] = π.critic.δ * ϵ[i]
    end

    # add the penalty to the gradient
    addgrad!(π.params.∇, π.penalty, π.params.θ)

    a
end
