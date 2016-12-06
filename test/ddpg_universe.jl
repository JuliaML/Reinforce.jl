module DDPG

using OpenAIGym
using DataStructures

env = GymEnv("flashgames.DuskDrive-v0")
experience = CircularBuffer{typeof(state(env))}(100)

policy = RandomPolicy()

for sars in Episode(env, policy)
    @show sars
    push!(experience, sars)
end
@show length(experience)

end # module
