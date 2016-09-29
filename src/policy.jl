
type RandomPolicy <: AbstractPolicy end
action(policy::RandomPolicy, r, s′, A′) = rand(A′)

# ------------------------------------------------------------------

"""
A StochasticPolicy is parameterized by the action set from which it chooses actions.

More specifically, it samples from a distribution D(ϕ), where ϕ contains all sufficient statistics.
In the case of a MultivariateNormal: ϕ = vcat(μ, vec(lower_triangle(Σ)))
"""
type StochasticPolicy{T, ASET <: AbstractSet, DIST <: Distribution} <: Transformation
    dist::DIST
    A::ASET
    input::Node{:input,T,1}
end

function StochasticPolicy{T, DIST<:MvNormal}(::Type{T}, ::Type{DIST}, A::AbstractSet)
    n = length(A)
    dist = MultivariateNormal(zeros(n), eye(n))
    nϕ = div(n * (n+1), 2)
    input = Node(T, :input, nϕ)
    StochasticPolicy(dist, A, input)
end

StochasticPolicy(A::AbstractSet) = StochasticPolicy(Float64, MultivariateNormal, A)

function Reinforce.action(policy::StochasticPolicy, r, s′, A′)
    n = length(policy.dist.μ)
    input = input_value(policy)
    d = policy.dist
    d.μ[:] = view(input, 1:n)

    # TODO:
    #   - using view(input, n+1:input_length(policy)), update the lower triangle of d.Σ
    #   - then fill in the rest of d.Σ (or do it while filling the lower triangle)
    #   - Sample from d
    #   - clamp to [-1,1] and then scale to intervals
    #   - return those values as actions
end
