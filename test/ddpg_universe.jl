module DDPG

using OpenAIGym
using DataStructures

# NOTE: to find other valid environment names, look at the universe code that registers them:
#   https://github.com/openai/universe/blob/master/universe/__init__.py
# env = GymEnv("flashgames.DuskDrive-v0")
# env = GymEnv("Pong-v3")
env = GymEnv("wob.mini.CircleCenter-v0")

# experience replay buffer
experience = CircularBuffer{Tuple}(100)

# agent/policy
policy = RandomPolicy()

# main loop... run one episode, getting a tuple: (s, a, r, s′)
for sars in Episode(env, policy)
    # @show sars
    s,a,r,s′ = sars
    @show a,r

    push!(experience, sars)

    render(env)

    #= TODO
    - actor/critic share "feature transformation" ϕ(s,a)
    - update policy (agent) by sampling from the experience replay
    =#
end
@show length(experience)

end # module
