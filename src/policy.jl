
type RandomPolicy <: AbstractPolicy end
action(policy::RandomPolicy, r, s′, A′) = rand(A′)

# ------------------------------------------------------------------

"""
A StochasticPolicy is parameterized by the action set from which it chooses actions.

More specifically, it samples from a distribution D(ϕ), where ϕ contains all sufficient statistics.
In the case of a MultivariateNormal N(μ, Σ):
    Σ = ZZ'   (Z is the cholesky decomp to ensure Σ is positive definite)
    ϕ := vcat(μ, vec(Z))
"""
type StochasticPolicy{T, ASET <: AbstractSet, DIST <: Distribution, TRANS <: Transformation} <: AbstractPolicy
    dist::DIST
    A::ASET
    # input::Node{:input,T,1}         # The input to the policy: ϕ
    trans::TRANS                    # This transformation maps: s --> ϕ
    act::Activation{:logistic,T}    # After sampling from dist(μ,Σ), squash to [0,1].
                                    # The actions are activation outputs mapped to [lo,hi]
    actions::Vector{T}
end

function StochasticPolicy{T, DIST<:MvNormal}(::Type{T}, ::Type{DIST}, A::AbstractSet, trans::Transformation)
    n = length(A)
    nϕ = n * (n+1)  # size of μ + size of Z
    StochasticPolicy(
        MultivariateNormal(zeros(T,n), eye(T,n)),
        A,
        # Node(T, :input, nϕ),
        trans,
        Activation{:logistic,T}(n),
        zeros(T,n)
    )
end
function StochasticPolicy(A::AbstractSet, trans::Transformation)
    StochasticPolicy(Float64, MultivariateNormal, A, trans)
end

function Reinforce.action(policy::StochasticPolicy, r, s′, A′)
    ϕ = transform!(policy.trans, s′)

    # TODO: do something useful with reward

    n = length(A′)
    # input = input_value(policy)
    dist = policy.dist

    # update μ with the first n input values
    dist.μ[:] = view(ϕ, 1:n)

    # next update the stored cholesky decomp L in: Σ = LL'
    # this allows us to ensure positive definiteness on Σ
    dist.Σ.chol.factors[:] = view(ϕ, n+1:length(ϕ))

    # z ~ N(μ, Σ)
    # â = logistic(z)
    â = transform!(policy.act, rand(dist))

    # a = (â --> [lo,hi])
    for i in 1:n
        lo, hi = A′.lo[i], A′.hi[i]
        policy.actions[i] = lo + â[i] * (hi - lo)
    end
    policy.actions
end
