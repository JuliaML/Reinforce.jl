module DDPG

using OpenAIGym
using DataStructures

# NOTE: to find other valid environment names, look at the universe code that registers them:
#   https://github.com/openai/universe/blob/master/universe/__init__.py
env = gym("flashgames.DuskDrive-v0")
# env = gym("Pong-v3")
# env = gym("wob.mini.CircleCenter-v0")
# env = gym("Breakout-v0")
@show actions(env, state(env))
# error()

# agent/policy
policy = RandomPolicy()

# -----------------------------------
# quick AR process for arbitrary vectors
# used for exploration policy of DDPG
using Distributions
mutable struct ARProcess{T}
    prev::T
    reversion
    noise
end
function Base.get(ar::ARProcess)
    ar.prev .= ar.reversion .* ar.prev .+ rand(noise)
end
# -----------------------------------

mutable struct DdpgPolicy
    ns; na; nϕ
    features
    actor
    actor_target
    critic
    critic_target
    experience
end

function build_actor_critic(env)
    ns = length(state(env))
    na = length(actions(env))

    # shared feature map: ϕ(s,a)
    nϕ = 10
    features = nnet(ns, nϕ, [10], :relu)

    # actor: μ(s | Θμ)
    actor = Chain(features, Affine(nϕ, na))
    actor_target = copy(actor)

    # critic: Q(s,a | ΘQ)
    critic = Chain(features, Concat(nϕ+na), Affine(nϕ+na, 1))
    critic_target = copy(critic)

    # experience replay buffer
    experience = CircularBuffer{Tuple}(100)

    DdpgPolicy(ns, na, nϕ, features, actor, actor_target, critic, critic_target, experience)
end

# main loop... run one episode, getting a tuple: (s, a, r, s′)
for sars in Episode(env, policy)
    # @show sars
    s,a,r,s′ = sars
    # @show a,r

    # push!(experience, sars)

    OpenAIGym.render(env)

    #= TODO
    - actor/critic share "feature transformation" ϕ(s,a)
    - update policy (agent) by sampling from the experience replay
    =#
end
# @show length(experience)

end # module
