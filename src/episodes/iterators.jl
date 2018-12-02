
mutable struct Episode{E<:AbstractEnvironment,P<:AbstractPolicy,F<:AbstractFloat}
  env::E
  policy::P
  total_reward::F # total reward of the episode
  last_reward::F
  niter::Int      # current step in this episode
  freq::Int       # number of steps between choosing actions
  maxn::Int       # max steps in an episode - should be constant during an episode
end

Episode(env::AbstractEnvironment, π::AbstractPolicy; freq = 1, maxn = maxsteps(env)) =
  Episode(env, π, 0.0, 0.0, 1, freq, maxn)

function _start!(ep::Episode{E,P,F}) where {E,P,F}
  reset!(ep.env)
  reset!(ep.policy)
  ep.total_reward = zero(F)
  ep.niter = 1
end

_done(ep::Episode, i) =
  (ep.maxn != 0 && ep.niter >= ep.maxn) || finished(ep.env, state(ep.env))

# take one step in the enviroment after querying the policy for an action
function Base.iterate(ep::Episode, i = _start!(ep))
  _done(ep::Episode, i) && return nothing

  env = ep.env
  π = ep.policy
  s = state(env)
  A = actions(env, s)  # action space
  r = reward(env)
  a = action(π, r, s, A)

  @assert(a ∈ A, "action $a is not in $A")

  # take freq steps using action a
  last_reward = 0.0
  s′ = s
  for _ ∈ 1:ep.freq
    r, s′ = step!(env, s′, a)
    last_reward += r
    _done(ep, ep.niter) && break
  end

  ep.total_reward += last_reward
  ep.last_reward = last_reward
  ep.niter = i

  (s, a, r, s′), i + 1
end

"""
    run_episode(f, env, policy)
    run_episode(env, policy) do (s, a, r, s′)
      # render or something else
    end

Helper function for running an episode,
and return the total reward gained in this episode.
"""
function run_episode(f::Base.Callable, env::AbstractEnvironment, π::AbstractPolicy)
  ep = Episode(env, π)
  for sars in ep
    f(sars)
  end
  ep.total_reward
end

# ---------------------------------------------------------------------
# iterate through many episodes

mutable struct Episodes
  env
  kw

  # note: we have different groups of strategies depending on when they should be applied
  episode_strats    # learning strategies for each episode
  epoch_strats      # learning strategies for each complete episode
  iter_strats       # learning strategies applied at every iteration
end

function Episodes(env;
                  episode_strats = [],
                  epoch_strats = [],
                  iter_strats = [],
                  kw...)
  Episodes(
    env,
    kw,
    MetaLearner(episode_strats...),
    MetaLearner(epoch_strats...),
    MetaLearner(iter_strats...)
  )
end

length_state(eps::Episodes) = length(state(eps.env)) + length(eps.last_action)

# the main function... run episodes until stopped by one of the epoch/iter strats
function learn!(policy, eps::Episodes)
  # setup
  setup!(eps.epoch_strats, policy)
  setup!(eps.iter_strats, policy)

  # loop over epochs until done
  done = false
  epoch = 1
  iter = 1

  while !done
    # one episode
    setup!(eps.episode_strats, policy)
    ep = Episode(eps.env, policy; eps.kw...)
    for sars′ in ep
      learn!(policy, sars′...)

      # learn steps
      for metalearner in (eps.episode_strats, eps.epoch_strats, eps.iter_strats)
        for strat in metalearner.managers
          learn!(policy, strat, sars′)
        end
      end

      # iter steps
      timestep = ep.niter
      hook(eps.episode_strats, ep, timestep)
      hook(eps.epoch_strats, ep, epoch)
      hook(eps.iter_strats, ep, iter)

      # finish the timestep with checks
      if finished(eps.episode_strats, policy, timestep)
        break
      end
      if finished(eps.epoch_strats, policy, epoch) || finished(eps.iter_strats, policy, iter)
        done = true
        break
      end
      iter += 1
    end
    info("Finished episode $epoch after $(ep.niter) steps. Reward: $(ep.total_reward) mean(Reward): $(ep.total_reward/max(ep.niter,1))")
    cleanup!(eps.episode_strats, policy)
    epoch += 1

  end

  # tear down
  cleanup!(eps.epoch_strats, policy)
  cleanup!(eps.iter_strats, policy)
  return
end

# function hook(policy, ep::Episodes, i)
#     if ep.should_reset
#         reset!(ep.env)
#         reset!(policy)
#         ep.should_reset = false
#         ep.total_reward = 0.0
#         ep.nsteps = 0
#         for learner in ep.learners
#             setup!(learner, policy)
#         end
#     end
#
#     # take one step in the enviroment after querying the policy for an action
# 	env = ep.env
# 	s = state(env)
#     A = actions(env, s)
#     r = reward(env)
# 	a = action(policy, r, s, A)
#     if !(a in A)
#         warn("action $a is not in $A")
#         # a = rand(A)
#     end
#     @assert a in A
# 	r, s′ = step!(env, s, a)
# 	ep.total_reward += r
#
#     # "sars" learn step for the policy...
#     #   note: ensures that the final reward is included in the learning
#     learn!(policy, s, a, r, s′)
#
#     ep.nsteps += 1
#     for learner in ep.learners
#         learn!(policy, learner, ep.nsteps)
#         hook(learner, policy, ep.nsteps)
#     end
#
#     # if this episode is done, just flag it so we reset next time
#     if finished(env, s′) || any(learner -> finished(learner, policy, ep.nsteps), ep.learners)
#         ep.should_reset = true
#         ep.nepisode += 1
#         for learner in ep.learners
#             cleanup!(learner, policy)
#         end
#         info("Finished episode $(ep.nepisode) after $(ep.nsteps) steps. Reward: $(ep.total_reward)")
#     end
#     return
# end
