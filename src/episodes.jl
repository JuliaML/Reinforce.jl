
# ----------------------------------------------------------------
# Episode iteration

type Episode{E<:AbstractEnvironment,P<:AbstractPolicy}
    env::E
    policy::P
    total_reward::Float64
    niter::Int
    maxsteps::Int
end

function Episode(env::AbstractEnvironment, policy::AbstractPolicy; maxsteps=typemax(Int))
	Episode(env, policy, 0.0, 0, maxsteps)
end

function Base.start(ep::Episode)
	reset!(ep.env)
	ep.total_reward = 0.0
	ep.niter = 1
end

function Base.done(ep::Episode, i)
	finished(ep.env, state(ep.env)) || i >= ep.maxsteps
end

function Base.next(ep::Episode, i)
	env = ep.env
	s = state(env)
    A = actions(env, s)
    r = reward(env)
	a = action(ep.policy, r, s, A)
    @assert a in A
    # if !(a in A)
    #     warn("action $a is not in $A")
    #     a = rand(A)
    # end
	r, s′ = step!(env, s, a)
	ep.total_reward += r
	ep.niter = i + 1
	(s, a, r, s′), i+1
end


# TODO: replace this with something better
function episode!(env, policy = RandomPolicy(); stepfunc = on_step, kw...)
	ep = Episode(env, policy; kw...)
	for sars in ep
		stepfunc(env, ep.niter, sars)
	end
	ep.total_reward, ep.niter
end


# # override these for custom functionality for your environment
on_step(env::AbstractEnvironment, i::Int, sars) = return
# check_constraints(env::AbstractEnvironment, s, a) = return

# # run a single episode. by default, it will run until `step!` returns false
# function episode!(env::AbstractEnvironment,
# 				  policy::AbstractPolicy;
# 				  maxsteps::Int = typemax(Int),
# 				  stepfunc::Function = on_step)
# 	reset!(env)
# 	i = 1
# 	total_reward = 0.0
# 	reset!(env)
# 	r = reward(env)
# 	while true
# 		s = state(env)
# 		a = action(policy, r, s, actions(env))
# 		r, s = step!(env, s, a)
# 		stepfunc(env, i)
# 		total_reward += r
# 		if done(env) || i > maxsteps
# 			break
# 		end
# 		i += 1
# 	end
# 	total_reward, i
# end

# ----------------------------------------------------------------

"""
This is a learning strategy that can infinitely generate steps of episodes,
and seemlessly reset when appropriate.
"""
type EpisodeLearner <: LearningStrategy
    env
    total_reward::Float64   # total reward of the episode
    nepisode::Int           # number of completed episodes
    nsteps::Int             # number of completed steps in this episode
    should_reset::Bool      # should we reset the episode on the next call?
    learners                # tuple of LearningStrategies
end

function EpisodeLearner(env, learners::LearningStrategy...)
    EpisodeLearner(env, 0.0, 0, 0, true, learners)
end

function learn!(policy, ep::EpisodeLearner, i)
    if ep.should_reset
        reset!(ep.env)
        reset!(policy)
        ep.should_reset = false
        ep.total_reward = 0.0
        ep.nsteps = 0
        for learner in ep.learners
            pre_hook(learner, policy)
        end
    end

    # take one step in the enviroment after querying the policy for an action
	env = ep.env
	s = state(env)
    A = actions(env, s)
    r = reward(env)
	a = action(policy, r, s, A)
    if !(a in A)
        warn("action $a is not in $A")
        # a = rand(A)
    end
    @assert a in A
	r, s′ = step!(env, s, a)
	ep.total_reward += r

    # "sars" learn step for the policy...
    #   note: ensures that the final reward is included in the learning
    learn!(policy, s, a, r, s′)

    ep.nsteps += 1
    for learner in ep.learners
        learn!(policy, learner, ep.nsteps)
        iter_hook(learner, policy, ep.nsteps)
    end

    # if this episode is done, just flag it so we reset next time
    if finished(env, s′) || any(learner -> finished(learner, policy, ep.nsteps), ep.learners)
        ep.should_reset = true
        ep.nepisode += 1
        for learner in ep.learners
            post_hook(learner, policy)
        end
        info("Finished episode $(ep.nepisode) after $(ep.nsteps) steps. Reward: $(ep.total_reward)")
    end
    return
end
