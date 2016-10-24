
#=
This is an implementation of the "AC" algorithm (Actor-critic Algorithm) from:
    Degris et al, "Model-Free Reinforcement Learning with Continuous Action in Practice"
=#

type ActorCritic{ALGO, T, WGT<:Weight, PEN<:Penalty, PHI<:Learnable,
                DIST<:MvNormalTransformation} <: AbstractPolicy
    δ::T            # last TD δ
    # r̄::T            # estimate of average return
    r̄::Mean{WGT}
    svar::Variances{WGT}       # feaure whitener TODO: do actual whitening!
    penalty::PEN
    nv::Int         # length(v) == length(x(s))
    na::Int         # the number of actions
    nu::Int         # the number of policy params (2na, since ϕ = vcat(μ,σ))
    x::Function     # a "feature mapping" function x(s)
    v::Vector{T}    # critic params
    eᵛ::Vector{T}   # eligibility trace for updating v (critic params)
    ϕ::PHI            # u from the paper is really params(ϕ), ∇logπ is grad(ϕ)
    D::DIST         # the N(ϕ) = N(μ,σ) from which we sample actions
    w::Vector{T}    # for INAC algos
    eᵘ::Vector{T}   # eligibility trace for updating u (actor params)
    γ::T            # δ decay
    λ::T            # e decay
    # αʳ::T           # learning rate for r̄
    # αᵛ::T           # learning rate for v
    # αᵘ::T           # learning rate for u
    gaᵛ
    gaᵘ
end

function ActorCritic(s::AbstractVector, na::Int;
                     T::DataType = eltype(s),
                     algo::Symbol = :AC,
                     wgt_lookback::Int = 10000,
                     penalty::Penalty = L2Penalty(1e-5),
                     ϕ::Learnable = nnet(length(s), 2na, [], :relu),
                     D::MvNormalTransformation = MvNormalTransformation(zeros(T,na),zeros(T,na)),
                     x::Function = identity,
                     γ::Number = 1.0,
                     λ::Number = 0.5,
                    #  αʳ::Number = 0.01,
                     αᵛ::Number = 0.01,
                     αᵘ::Number = 0.01,
                     gaᵛ = OnlineGradAvg(100, lr=αᵛ),
                     gaᵘ = OnlineGradAvg(100, lr=αᵘ)
                    )
    @assert algo in (:AC, :INAC)
    T = eltype(s)
    nv = length(x(s))
    nu = length(params(ϕ))
    wgt = BoundedEqualWeight(wgt_lookback)
    r̄ = Mean(wgt)
    svar = Variances(nv, wgt)
    link_nodes!(ϕ, D)

    pre_hook(gaᵛ, zeros(nv))
    pre_hook(gaᵘ, ϕ)

    ActorCritic{algo,T,typeof(wgt),typeof(penalty),typeof(ϕ),typeof(D)}(
        zero(T),
        # zero(T),
        r̄,
        svar,
        penalty,
        nv,
        na,
        nu,
        x,
        zeros(T,nv),
        zeros(T,nv),
        ϕ,
        D,
        zeros(T,nu),
        zeros(T,nu),
        γ,
        λ,
        # αʳ,
        # αᵛ,
        # αᵘ
        gaᵛ,
        gaᵘ
    )
end

function Reinforce.reset!{A,T}(ac::ActorCritic{A,T})
    fill!(ac.eᵛ, zero(T))
    fill!(ac.eᵘ, zero(T))
end

@with ac function Reinforce.action(ac::ActorCritic, r, s′, A′)
    transform!(ϕ, whiten(s′, svar))
    z = transform!(D)

    # TODO: make this a general utility method... project_to_actions?
    # project our squashed sample onto into the action space to get our actions
    # a = (â --> [lo,hi])
    a = A′.lo .+ logistic.(z) .* (A′.hi .- A′.lo)
    if !(a in A′)
        warn("a=$a not in A=$(A′)")
        dump(ac)
        error()
        a = rand(A′)
    end
    a
end


#= Notes:
    - we can replace αᵛ/αᵘ with a GradientLearner and call update! instead
    - the @with macro (in StochasticOptimization) replaces variables with the
        dot versions if that object has a field of that name
=#

function whiten(x::AbstractArray, vars::Variances)
    σ = vars.value
    μ = vars.μ
    @assert length(x) == length(σ)
    out = zeros(x)
    @inbounds for (i,j) in enumerate(eachindex(x))
        out[j] = (x[j] - μ[i])
        if σ[i] > 0
            out[j] /= sqrt(σ[i])
        end
    end
    out
end

@with ac function learn!(ac::ActorCritic, s, a, r, s′)
    xs = whiten(x(s), svar)
    xs′ = whiten(x(s′), svar)

    # update our input whitener
    fit!(svar, x(s′))

    # compute TD delta
    δ = r - mean(r̄) + γ * dot(v, xs′) - dot(v, xs)

    # update average reward
    # r̄ += αʳ * δ
    # r̄ = αʳ * r + (one(αʳ) - αʳ) * r̄
    fit!(r̄, r)

    # update critic
    γλ = γ * λ
    chg = zeros(v)
    for i=1:nv
        eᵛ[i] = γλ * eᵛ[i] + (1-γλ) * xs[i]
        chg[i] = δ * eᵛ[i] - deriv(penalty, v[i])
        # v[i] += αᵛ * chg #+ (one(αᵛ) - αᵛ) * v[i]
    end
    learn!(v, gaᵛ, chg)

    # compute ∇logπ
    # TODO: add penalty?
    grad!(D)
    grad!(ϕ)

    # if αᵘ > 0
        # update actor eligibility trace
        ∇logπ = grad(ϕ)
        # @show extrema(∇logπ), extrema(eᵘ)
        for i=1:nu
            eᵘ[i] = γλ * eᵘ[i] + (1-γλ) * ∇logπ[i]
        end

        # update the actor (different by algo)
        update_actor!(ac, params(ϕ), ∇logπ)
    # end
end

@with ac function update_actor!(ac::ActorCritic{:AC}, u, ∇logπ)
    # σ² = D.dist.Σ.diag
    chg = zeros(u)
    for i=1:nu
        # note: we multiply by σ² to reduce instabilities
        chg[i] = δ * eᵘ[i] - deriv(penalty, u[i])
        # chg = δ * eᵘ[i] * σ²[mod1(i,na)] - deriv(penalty, u[i])
        # u[i] += αᵘ * chg
    end
    learn!(u, gaᵘ, chg)
end

# BROKEN:
@with ac function update_actor!{T}(ac::ActorCritic{:INAC,T}, u, ∇logπ)
    ∇ᵀw = dot(∇logπ, w)
    # @show extrema(∇logπ), ∇ᵀw
    for i=1:nu
        w[i] = αᵘ * (δ * eᵘ[i] - ∇logπ[i] * ∇ᵀw) + (one(αᵘ) - αᵘ) * w[i]
        if !isfinite(w[i])
            @show w δ eᵘ ∇logπ ∇ᵀw
            @show map(extrema, (w, δ, eᵘ, ∇logπ, ∇ᵀw))
            error()
        end
        chg = w[i] - deriv(penalty, u[i])
        u[i] += αᵘ * chg
    end
end
