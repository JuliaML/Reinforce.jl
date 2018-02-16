
mutable struct Episode
    env
    policy
    total_reward   # total reward of the episode
    last_reward
    niter::Int             # current step in this episode
    freq::Int               # number of steps between choosing actions
    # append_action::Bool
    # last_action
end

function Episode(env, policy; freq=1) #, append_action=false)
    Episode(env, policy,
        0.0, 0.0, 1, freq,
        # append_action,
        # append_action ? rand(actions(env, state(env))) : zeros(0)
    )
end

function Base.start(ep::Episode)
    reset!(ep.env)
    reset!(ep.policy)
	ep.total_reward = 0.0
	ep.niter = 1
end

function Base.done(ep::Episode, i)
    finished(ep.env, state(ep.env))
end

# take one step in the enviroment after querying the policy for an action
function Base.next(ep::Episode, i)
	env = ep.env
	s = state(env)
    A = actions(env, s)
    r = reward(env)
	# a = action(ep.policy, r, ep.append_action ? vcat(s,ep.last_action) : s, A)
	a = action(ep.policy, r, s, A)
    if !(a in A)
        warn("action $a is not in $A")
        # a = rand(A)
    end
    @assert a in A

    # take freq steps using action a
    last_reward = 0.0
    s′ = s
    for _=1:ep.freq
        r, s′ = step!(env, s′, a)
        last_reward += r
        done(ep, ep.niter) && break
    end

    ep.total_reward += last_reward
    ep.last_reward = last_reward
    ep.niter = i

    # if ep.append_action
    #     s = vcat(s, ep.last_action)
    #     s′ = vcat(s′, a)
    #     ep.last_action = a
    # end

	(s, a, r, s′), i+1
end

"""
Use do-block notation to run an episode:

```
run_episode(env,policy) do
    # render or something else
end
```
"""
function run_episode(f::Function, env::AbstractEnvironment, policy::AbstractPolicy; maxsteps = typemax(Int))
    R = 0.
    for (i, sars) in enumerate(Episode(env, policy))
        R += sars[3]
        f() # the do block
        i >= maxsteps && break
    end
    R
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
    pre_hook(eps.epoch_strats, policy)
    pre_hook(eps.iter_strats, policy)

    # loop over epochs until done
    done = false
    epoch = 1
    iter = 1
    while !done

        # one episode
        pre_hook(eps.episode_strats, policy)
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
            iter_hook(eps.episode_strats, ep, timestep)
            iter_hook(eps.epoch_strats, ep, epoch)
            iter_hook(eps.iter_strats, ep, iter)

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
        post_hook(eps.episode_strats, policy)
        epoch += 1

    end

    # tear down
    post_hook(eps.epoch_strats, policy)
    post_hook(eps.iter_strats, policy)
    return
end

# function iter_hook(policy, ep::Episodes, i)
#     if ep.should_reset
#         reset!(ep.env)
#         reset!(policy)
#         ep.should_reset = false
#         ep.total_reward = 0.0
#         ep.nsteps = 0
#         for learner in ep.learners
#             pre_hook(learner, policy)
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
#         iter_hook(learner, policy, ep.nsteps)
#     end
#
#     # if this episode is done, just flag it so we reset next time
#     if finished(env, s′) || any(learner -> finished(learner, policy, ep.nsteps), ep.learners)
#         ep.should_reset = true
#         ep.nepisode += 1
#         for learner in ep.learners
#             post_hook(learner, policy)
#         end
#         info("Finished episode $(ep.nepisode) after $(ep.nsteps) steps. Reward: $(ep.total_reward)")
#     end
#     return
# end
