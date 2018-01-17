
# WARNING: This should not be used as-is.  It's unfinished experimentation.

using Transformations
using StochasticOptimization
using PenaltyFunctions
import OnlineStats: Mean, Variances, Weight, BoundedEqualWeight

mutable struct Actor{PHI<:Learnable, DIST<:MvNormalTransformation} <: AbstractPolicy
    ϕ::PHI      # map states to dist inputs. ∇logπ is grad(ϕ)
    D::DIST     # the N(ϕ) = N(μ,σ) from which we sample actions
    prep::PreprocessStep
    last_action
    testing::Bool
end
Actor(ϕ,D,prep=NoPreprocessing()) = Actor(ϕ, D, prep, zeros(D.n), false)

# TODO: specialize this on the distribution type
function Reinforce.action(actor::Actor, r, s′, A′)
    z = transform!(actor, s′)
    # @show map(extrema, (s′, z, params(actor), grad(actor)))

    # TODO: make this a general utility method... project_to_actions?
    # project our squashed sample onto into the action space to get our actions
    # a = (â --> [lo,hi])
    a = A′.lo .+ logistic.(z) .* (A′.hi .- A′.lo)
    if !(a in A′)
        warn("a=$a not in A=$(A′)")
        dump(actor)
        error()
        a = rand(A′)
    end
    copy!(actor.last_action, a)
    a
end

@with actor function grad!(actor::Actor)
    grad!(D)
    grad!(ϕ, input_grad(D))
end
@with actor function transform!(actor::Actor, xs::AbstractVector)
    x = if isa(prep, NoPreprocessing)
        xs
    else
        learn!(prep, xs)
        transform!(prep, xs)
    end
    phi = transform!(ϕ, x)
    if testing
        # if we're testing, return the mean exactly
        phi[1:D.n]
    else
        transform!(D, phi)
    end
end
params(actor::Actor) = params(actor.ϕ)
grad(actor::Actor) = grad(actor.ϕ)


#=
This is an implementation of the "AC" algorithm (Actor-critic Algorithm) from:
    Degris et al, "Model-Free Reinforcement Learning with Continuous Action in Practice"
=#

mutable struct OnlineActorCritic{ALGO, T, WGT<:Weight, PEN<:Penalty, ACTOR} <: AbstractPolicy
    δ::T            # last TD δ
    # r̄::T            # estimate of average return
    r̄::Mean{WGT}
    # svar::Variances{WGT}       # feaure whitener TODO: do actual whitening!
    penalty::PEN
    ns::Int         # number of inputs
    nv::Int         # 2ns+na because we do: vcat(s′, s′-s, a)
    na::Int         # the number of actions
    nu::Int         # the number of policy params (2na, since ϕ = vcat(μ,σ))
    x::Function     # a "feature mapping" function x(s)
    v::Vector{T}    # critic params
    eᵛ::Vector{T}   # eligibility trace for updating v (critic params)
    # ϕ::PHI            # u from the paper is really params(ϕ), ∇logπ is grad(ϕ)
    # D::DIST         # the N(ϕ) = N(μ,σ) from which we sample actions
    actor::ACTOR
    w::Vector{T}    # for INAC algos
    eᵘ::Vector{T}   # eligibility trace for updating u (actor params)
    γ::T            # δ decay
    λ::T            # e decay
    # αʳ::T           # learning rate for r̄
    # αᵛ::T           # learning rate for v
    # αᵘ::T           # learning rate for u
    gaᵛ
    gaᵘ
    # gaʷ
    # last_sars′
    xs
end

function OnlineActorCritic(s::AbstractVector, na::Int;
                     T::DataType = eltype(s),
                     algo::Symbol = :AC,
                     wgt_lookback::Int = 20000,
                     prep::PreprocessStep = Whiten(T,2length(s)+na,2length(s)+na,lookback=wgt_lookback),
                     penalty::Penalty = L2Penalty(1e-5),
                     ϕ::Learnable = nnet(2length(s)+na, 2na, [], :relu),
                     D::MvNormalTransformation = MvNormalTransformation(zeros(T,na),zeros(T,na)),
                     x::Function = identity,
                     γ::Number = 1.0,
                     λ::Number = 0.5,
                    #  αʳ::Number = 0.01,
                     αᵛ::Number = 0.01,
                     αᵘ::Number = 0.01,
                     αʷ::Number = 0.01,
                     gaᵛ = OnlineGradAvg(100, lr=αᵛ),
                     gaᵘ = OnlineGradAvg(100, lr=αᵘ)
                    #  gaʷ = OnlineGradAvg(100, lr=αʷ)
                    )
    @assert algo in (:AC, :INAC)
    T = eltype(s)
    ns = length(x(s))
    nv = 2ns+na
    nu = length(params(ϕ))
    wgt = BoundedEqualWeight(wgt_lookback)
    r̄ = Mean(wgt)
    # svar = Variances(nv, wgt)
    actor = Actor(ϕ,D,prep)
    link_nodes!(ϕ, D)

    pre_hook(gaᵛ, zeros(nv))
    pre_hook(gaᵘ, ϕ)
    # pre_hook(gaʷ, ϕ)

    OnlineActorCritic{algo,T,typeof(wgt),typeof(penalty),typeof(actor)}(
        zero(T),
        # zero(T),
        r̄,
        # svar,
        penalty,
        ns,
        nv,
        na,
        nu,
        x,
        zeros(T,nv),
        zeros(T,nv),
        # ϕ,
        # D,
        actor,
        zeros(T,nu),
        zeros(T,nu),
        γ,
        λ,
        # αʳ,
        # αᵛ,
        # αᵘ
        gaᵛ,
        gaᵘ,
        # gaʷ,
        # (s, zeros(na), 0.0, s)
        vcat(s, zeros(T,ns+na))
    )
end

function Reinforce.reset!(ac::OnlineActorCritic{A,T}) where {A,T}
    # reset!(ac.actor)
    fill!(ac.eᵛ, zero(T))
    fill!(ac.eᵘ, zero(T))
end

@with ac function Reinforce.action(ac::OnlineActorCritic, r, s′, A′)
    # ignore s′ and use latest xs (which should include s′)
    for i=1:ns
        xs[i+ns] = s′[i] - xs[i]
        xs[i] = s′[i]
    end
    xs[2ns+1:end] = actor.last_action
    a = action(actor, r, xs, A′)
    # if !isa(actor.prep, NoPreprocessing)
    #     xs[:] = output_value(actor.prep)
    # end
    a
    # # for i=1:ns
    # #     xs[i+ns] = s′[i] - xs[i]
    # #     xs[i] = s′[i]
    # # end
    # # xs[ns+1:2ns] = s′ -
    # # s′ = vcat(s′, s′-last_sars′[1], last_sars′[2])
    # # transform!(ϕ, xs)
    # # z = transform!(D)
    # z = transform!(actor)
    #
    # # TODO: make this a general utility method... project_to_actions?
    # # project our squashed sample onto into the action space to get our actions
    # # a = (â --> [lo,hi])
    # a = A′.lo .+ logistic.(z) .* (A′.hi .- A′.lo)
    # if !(a in A′)
    #     warn("a=$a not in A=$(A′)")
    #     dump(ac)
    #     error()
    #     a = rand(A′)
    # end
    # # xs[2ns+1:end] = a
    # a
end


#= Notes:
    - we can replace αᵛ/αᵘ with a GradientLearner and call update! instead
    - the @with macro (in StochasticOptimization) replaces variables with the
        dot versions if that object has a field of that name
=#

# function whiten(x::AbstractArray, vars::Variances)
#     return x
#     σ = vars.value
#     μ = vars.μ
#     @assert length(x) == length(σ)
#     out = zeros(x)
#     @inbounds for (i,j) in enumerate(eachindex(x))
#         out[j] = (x[j] - μ[i])
#         if σ[i] > 0
#             out[j] /= sqrt(σ[i])
#         end
#     end
#     out
# end


# This is similar to replacing traces for continuous values.
# If we're reversing the trace then we simply add, if we're increasing
# the magnitude then we take the larger of ei/xi.
# This should help with the stability of traces.
# Note: invented by @tbreloff, but I'm sure it exists somewhere already.
function update_eligibilty!(e::AbstractArray{T}, x::AbstractArray{T},
                           γλ::Number; clip::Number = 1e2) where T
    @assert length(e) == length(x)
    @inbounds for i=1:length(e)
        ei = γλ * e[i]
        xi = clamp(x[i], -clip, clip)
        e[i] = if ei < zero(T)
            xi < zero(T) ? min(ei, xi) : ei+xi
        else
            xi > zero(T) ? max(ei, xi) : ei+xi
        end
    end
end

@with ac function learn!(ac::OnlineActorCritic, s, a, r, s′)
    actor.testing && return
    # xs = whiten(x(s), svar)
    # xs′ = whiten(x(s′), svar)
    #
    # # update our input whitener
    # fit!(svar, x(s′))

    # xs = vcat(s, s-last_sars′[1], last_sars′[2])
    xs′ = vcat(s′, s′-s, a)

    prepped_xs = transform!(actor.prep, xs)
    prepped_xs′ = transform!(actor.prep, xs′)

    # compute TD delta
    δ = r - mean(r̄) + γ * dot(v, prepped_xs′) - dot(v, prepped_xs)

    # update average reward
    # r̄ += αʳ * δ
    # r̄ = αʳ * r + (one(αʳ) - αʳ) * r̄
    fit!(r̄, r)

    # update critic
    γλ = γ * λ
    update_eligibilty!(eᵛ, prepped_xs, γλ)
    chg = zeros(v)
    for i=1:nv
        # eᵛ[i] = γλ * eᵛ[i] + xs[i]
        chg[i] = -δ * eᵛ[i] + deriv(penalty, v[i])
        # v[i] += αᵛ * chg #+ (one(αᵛ) - αᵛ) * v[i]
    end
    learn!(v, gaᵛ, chg)

    # compute ∇logπ
    # TODO: add penalty?
    # grad!(D)
    # grad!(ϕ)

    # update actor eligibility trace
    grad!(actor)
    ∇logπ = grad(actor)
    # @show extrema(∇logπ), extrema(eᵘ)
    update_eligibilty!(eᵘ, ∇logπ, γλ)
    # for i=1:nu
    #     eᵘ[i] = γλ * eᵘ[i] + ∇logπ[i]
    # end

    # update the actor (different by algo)
    update_actor!(ac, params(actor), ∇logπ)
    return
end

@with ac function update_actor!(ac::OnlineActorCritic{:AC}, u, ∇logπ)
    # σ² = D.dist.Σ.diag
    chg = zeros(u)
    for i=1:nu
        # note: we multiply by σ² to reduce instabilities
        chg[i] = -δ * eᵘ[i] + deriv(penalty, u[i])
        if isnan(chg[i])
            @show i, δ, eᵘ[i], u[i]
        end
        # chg = δ * eᵘ[i] * σ²[mod1(i,na)] - deriv(penalty, u[i])
        # u[i] += αᵘ * chg
    end
    learn!(u, gaᵘ, chg)
end

# # BROKEN:
# @with ac function update_actor!{T}(ac::OnlineActorCritic{:INAC,T}, u, ∇logπ)
#     ∇ᵀw = dot(∇logπ, w)
#
#     # update w
#     chg = zeros(u)
#     for i=1:nu
#         chg[i] = -δ * eᵘ[i] + ∇logπ[i] * ∇ᵀw
#     end
#     learn!(w, gaʷ, chg)
#
#     # update u
#     for i=1:nu
#         chg[i] = w[i] - deriv(penalty, u[i])
#     end
#     learn!(u, gaᵘ, chg)
# end

# # BROKEN:
# @with ac function update_actor!{T}(ac::OnlineActorCritic{:INAC,T}, u, ∇logπ)
#     ∇ᵀw = dot(∇logπ, w)
#     # @show extrema(∇logπ), ∇ᵀw
#     chg = zeros(u)
#     for i=1:nu
#         w[i] += αᵘ * (δ * eᵘ[i] - ∇logπ[i] * ∇ᵀw) #+ (one(αᵘ) - αᵘ) * w[i]
#         if !isfinite(w[i])
#             @show w δ eᵘ ∇logπ ∇ᵀw
#             @show map(extrema, (w, δ, eᵘ, ∇logπ, ∇ᵀw))
#             error()
#         end
#         chg[i] = w[i] - deriv(penalty, u[i])
#         # u[i] += αᵘ * chg
#     end
#     learn!(u, gaᵘ, chg)
# end

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

mutable struct EpisodicActorCritic

end
