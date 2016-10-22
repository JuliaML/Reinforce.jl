
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
=#

function learn!(ac::ActorCritic, s, a, r, s′)
    x = ac.x(s)
    x′ = ac.x(s′)

    # compute TD delta
    δ = r - ac.r̄ + ac.γ * dot(ac.v, x′) - dot(ac.v, x)

    # update average reward
    ac.r̄ += ac.αʳ * δ

    # update critic
    γλ = ac.γ * ac.λ
    for i=1:ac.nv
        ac.eᵛ[i] = γλ * ac.eᵛ[i] + x[i]
        ac.v[i] += ac.αᵛ * δ * ac.eᵛ[i]
    end
    # ac.eᵛ .= γλ .* ac.eᵛ .+ x
    # ac.v .+= (αᵛ * δ) .* ac.eᵛ

    # compute ∇logP
    # TODO: add penalty?
    grad!(ac.D)
    grad!(ac.u)

    # update actor
    Θ = params(ac.u)
    ∇logP = grad(ac.u)
    for i=1:ac.nu
        ac.eᵘ[i] = γλ * ac.eᵘ[i] + ∇logP[i]
        Θ[i] += ac.αᵘ * δ * ac.eᵘ[i]
    end
    # ac.eᵘ .= γλ .* ac.eᵘ .+ grad(ac.u)
    # params(ac.u) .+= (αᵘ * δ) .* ac.eᵘ
end
