
type ActorCritic <: AbstractPolicy
    r̄::Float64             # estimate of average return
    v::Vector{Float64}      # critic params
    eᵛ::Vector{Float64}     # eligibility trace for updating v (critic params)
    u::Vector{Float64}      # actor params
    eᵘ::Vector{Float64}     # eligibility trace for updating u (actor params)
    αʳ::Float64             # learning rate for r̄
    αᵛ::Float64             # learning rate for v
    αᵘ::Float64             # learning rate for u
    ActorCritic() = new()
end

function learn!(ac::ActorCritic, s, a, r, s′)
    
end
