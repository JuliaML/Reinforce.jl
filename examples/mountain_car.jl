using Reinforce, Plots
using Reinforce.MountainCarEnv.MountainCar
gr()

# Deterministic policy that is solving the problem
mutable struct BasicCarPolicy <: Reinforce.AbstractPolicy end
import Reinforce.action

function action(policy::BasicCarPolicy, r, s, A)
  if s.velocity < 0
    return 1
  else
    return 3
  end
end

# Environment setup
env = MountainCar()
on_step(env::MountainCar, niter, sars) = gui(plot(env))

function episode!(env, policy = RandomPolicy(); stepfunc = on_step, kw...)
    ep = Episode(env, policy; kw...)
    for sars in ep
        stepfunc(env, ep.niter, sars)
    end
    ep.total_reward, ep.niter
end

# Main part
println(episode!(env, BasicCarPolicy()))

# This one can be really long...
println(episode!(env, RandomPolicy()))
