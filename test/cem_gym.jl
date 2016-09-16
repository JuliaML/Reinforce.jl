module cemgym

# this is a port of the example found at
#	http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/lab1.html#starter

using Reinforce
# using OpenAIGym
# using Distributions
using Transformations
using StochasticOptimization

using Plots; gr(size=(500,200))

# ----------------------------------------------------------------

"""
Wraps a LearnBase.Transformation which converts an input vector to action values.
For discrete actions, it chooses the action which produces the highest value.
For continuous (interval) actions, it squashes actions to [0,1].
"""
immutable TransformPolicy{T} <: AbstractPolicy
	trans::T
end

Transformations.params(tp::TransformPolicy) = params(tp.trans)


# discrete: our action is the action which maximizes the affine transform
function Reinforce.action(π::TransformPolicy, r, s, A::DiscreteSet)
	A[indmax(transform!(π.trans, s))]
end

# # continuous: return the transform value, squashed to [0,1]
# # TODO: remove the squashing when a "Affine + Sigmoid" transformation is available
# function Reinforce.action(π::TransformPolicy, r, s, A::IntervalSet)
# 	Transformations.sigmoid(transform(π.trans, s)[1]) * (A.amax-A.amin) + A.amin
# end

# ----------------------------------------------------------------

function myplot(t, hist_min, hist_mean, hist_max, anim=nothing)
	(env,i,sars) -> if mod1(t,3)==1 && mod1(i,10)==1
		plot(
			plot(hist_mean, c=:black, fill=((hist_min,hist_max), 0.2), title="Progress", leg=false),
			plot(env, title = "Episode: $t  Iter: $i")
		)
		if anim == nothing
			gui()
		else
			frame(anim)
		end
	else
		return
	end
end

# ----------------------------------------------------------------



# initialize a policy, do the learning, then return the policy
function do_cem_test(sublearners...; env = GymEnv("CartPole-v0"),
                       maxsteps = 200, # max number of steps in one episode
                       maxiter = 1000, # max learning iterations
					   # noise_max = 1.0,
					   # noise_steps = 20,
					   # noise_func = t -> max(noise_max - t/noise_steps, 0.0),
					   kw...)
    s = state(env)
    A = actions(env,s)
    @assert isa(A, DiscreteSet) # this is the only kind that will work right now
	nS, nA = map(length, (s, A))
    policy = TransformPolicy(Affine(nS, nA))

    strat = CrossEntropyMethod(;maxsteps=maxsteps, kw...)
    learner = make_learner(strat, sublearners...; maxiter=maxiter, kw...)
    learn!(policy, learner, forever(env))

    @show policy strat
end


end #module
