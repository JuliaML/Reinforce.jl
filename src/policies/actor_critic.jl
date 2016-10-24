
#=
This is an implementation of the "AC" algorithm (Actor-critic Algorithm) from:
    Degris et al, "Model-Free Reinforcement Learning with Continuous Action in Practice"
=#

type ActorCritic{T, U<:Learnable, DIST<:MvNormalTransformation} <: AbstractPolicy
    r̄::T           # estimate of average return
    nv::Int        # length(v) == length(x(s))
    na::Int        # the number of actions
    nu::Int        # the number of policy params (2na, since u = vcat(μ,σ))
    x::Function    # a "feature mapping" function x(s)
    v::Vector{T}   # critic params
    eᵛ::Vector{T}  # eligibility trace for updating v (critic params)
    # u::Vector{T}  # actor params
    u::U           # u from the paper is really params(u), ∇logP is grad(u)
    D::DIST        # the N(u) = N(μ,σ) from which we sample actions
    # ∇logP::Vector{T} # grad-log-prob of the policy distribution N(u) = N(μ,σ)
    eᵘ::Vector{T}  # eligibility trace for updating u (actor params)
    γ::T           # δ decay
    λ::T           # e decay
    αʳ::T          # learning rate for r̄
    αᵛ::T          # learning rate for v
    αᵘ::T          # learning rate for u
end

function ActorCritic(s::AbstractVector, na::Int;
                     T::DataType = eltype(s),
                     u::Learnable = nnet(length(s), 2na, [], :relu),
                     D::MvNormalTransformation = MvNormalTransformation(zeros(T,na),zeros(T,na)),
                     x::Function = identity,
                     γ::Number = 1.0,
                     λ::Number = 0.5,
                     αʳ::Number = 0.01,
                     αᵛ::Number = 0.01,
                     αᵘ::Number = 0.01
                    )
    T = eltype(s)
    nv = length(x(s))
    nu = 2na
    link_nodes!(u, D)

    ActorCritic(
        zero(T),
        nv,
        na,
        nu,
        x,
        zeros(T,nv),
        zeros(T,nv),
        # zeros(T,nu),
        u,
        D,
        zeros(T,nu),
        γ,
        λ,
        αʳ,
        αᵛ,
        αᵘ
    )
end

function Reinforce.reset!{T}(ac::ActorCritic{T})
    fill!(ac.eᵛ, zero(T))
    fill!(ac.eᵘ, zero(T))
end

function Reinforce.action(ac::ActorCritic, r, s′, A′)
    transform!(ac.u, s′)
    z = transform!(ac.D)

    # TODO: make this a general utility method... project_to_actions?
    # project our squashed sample onto into the action space to get our actions
    # a = (â --> [lo,hi])
    a = A′.lo .+ logistic.(z) .* (A′.hi .- A′.lo)
    if !(a in A′)
        warn("a=$a not in A=$(A′)")
        a = rand(A′)
    end
    a
end


#= Notes:
    - we can replace αᵛ/αᵘ with a GradientLearner and call update! instead
    - the @with macro (in StochasticOptimization) replaces variables with the
        dot versions if that object has a field of that name
=#

@with ac function learn!(ac::ActorCritic, s, a, r, s′)
    xs = x(s)
    xs′ = x(s′)

    # compute TD delta
    δ = r - r̄ + γ * dot(v, xs′) - dot(v, xs)

    # update average reward
    r̄ += αʳ * δ

    # update critic
    γλ = γ * λ
    for i=1:nv
        eᵛ[i] = γλ * eᵛ[i] + xs[i]
        v[i] += αᵛ * δ * eᵛ[i]
    end

    # compute ∇logP
    # TODO: add penalty?
    grad!(D)
    grad!(u)

    # update actor
    Θ = params(u)
    ∇logP = grad(u)
    for i=1:nu
        eᵘ[i] = γλ * eᵘ[i] + ∇logP[i]
        Θ[i] += αᵘ * δ * eᵘ[i]
    end
end
