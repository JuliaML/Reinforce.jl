
# ----------------------------------------------------------------
# Episode iteration

immutable Episode{E<:AbstractEnvironment,P<:AbstractPolicy}
    env::E
    policy::P
    total_reward::Float64
    niter::Int
    maxiter::Int
end

function Episode(env::AbstractEnvironment, policy::AbstractPolicy; maxiter=typemax(Int))
	Episode(env, policy, 0.0, 0, maxiter)
end

function Base.start(ep::Episode)
	reset!(env)
	ep.total_reward = 0.0
	ep.niter = 1
end

function Base.done(ep::Episode, i)
	done(ep.env) || i >= ep.maxiter
end

function Base.next(ep::Episode, i)
	s = state(ep.env)
	a = action(ep.policy, reward(env), s, actions(ep.env))
	check_constraints(ep.env, s, a)
	r, s′ = step!(ep.env, s, a)
	total_reward += r
	ep.niter = i + 1
	(s, a, r, s′), i+1
end


# TODO: replace this with something better
function episode!(env, policy; stepfunc = on_step, kw...)
	ep = Episode(env, policy; kw...)
	for (i, sars) in enumerate(ep)
		stepfunc(env, i)
	end
	ep.total_reward, ep.niter
end


# # override these for custom functionality for your environment
# on_step(env::AbstractEnvironment, i::Int) = return
check_constraints(env::AbstractEnvironment, s, a) = return

# # run a single episode. by default, it will run until `step!` returns false
# function episode!(env::AbstractEnvironment,
# 				  policy::AbstractPolicy;
# 				  maxiter::Int = typemax(Int),
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
# 		if done(env) || i > maxiter
# 			break
# 		end
# 		i += 1
# 	end
# 	total_reward, i
# end

