
# default reset
reset!(policy::AbstractPolicy) = return

type RandomPolicy <: AbstractPolicy end
action(policy::RandomPolicy, r, s′, A′) = rand(A′)

# include("online_gae.jl")
# include("actor_critic.jl")
