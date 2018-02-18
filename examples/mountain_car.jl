using Reinforce
using Reinforce.MountainCarEnv: MountainCar

using Plots
gr()

# Deterministic policy that is solving the problem
mutable struct BasicCarPolicy <: Reinforce.AbstractPolicy end

Reinforce.action(policy::BasicCarPolicy, r, s, A) = s.velocity < 0 ? 1 : 3

# Environment setup
env = MountainCar()

function episode!(env, π = RandomPolicy())
  ep = Episode(env, π)
  for (s, a, r, s′) in ep
    gui(plot(env))
  end
  ep.total_reward, ep.niter
end

# Main part
R, n = episode!(env, BasicCarPolicy())
println("reward: $R, iter: $n")

# This one can be really long...
R, n = episode!(env, RandomPolicy())
println("reward: $R, iter: $n")
