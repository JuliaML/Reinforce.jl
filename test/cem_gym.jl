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
struct TransformPolicy{T} <: AbstractPolicy
	trans::T
end
Transformations.params(tp::TransformPolicy) = params(tp.trans)
Reinforce.action(π::TransformPolicy, r, s, A::DiscreteSet) = A[indmax(transform!(π.trans, s))]



# # continuous: return the transform value, squashed to [0,1]
# # TODO: remove the squashing when a "Affine + Sigmoid" transformation is available
# function Reinforce.action(π::TransformPolicy, r, s, A::IntervalSet)
# 	Transformations.sigmoid(transform(π.trans, s)[1]) * (A.amax-A.amin) + A.amin
# end

# ----------------------------------------------------------------


# initialize a policy, do the learning, then return the policy
function do_cem_test(sublearners...; env = GymEnv("CartPole-v0"),
                       maxsteps = 200, # max number of steps in one episode
                       maxiter = 1000, # max learning iterations
					   # noise_max = 1.0,
					   # noise_steps = 20,
					   # noise_func = t -> max(noise_max - t/noise_steps, 0.0),
					   kw...)
    # generic query of state and action size
    s = state(env)
    A = actions(env,s)
    @assert isa(A, DiscreteSet) # this is the only kind that will work right now
	nS, nA = map(length, (s, A))

    # create a simple policy: action = argmax(wx+b)
    policy = TransformPolicy(Affine(nS, nA))

    # initialize the CEM, which will learn Θ = {w,b}
    strat = CrossEntropyMethod(;maxsteps=maxsteps, kw...)

    # keep a vector of test episode returns.  after each iteration, run (and plot)
    # an episode using the current CEM μ
    Rs = zeros(0)
    theme(:dark)
    function iterfunc(model, i)
        copy!(params(policy), strat.μ)
        R = run_episode(env, policy, maxsteps=maxsteps) do
            plot(plot(env), plot(Rs, label="Test Reward")) |> display
        end
        push!(Rs, R)
    end

    # create a MetaLearner driven by the CEM strategy
    learner = make_learner(strat, sublearners...;
        maxiter=maxiter,
        oniter=iterfunc,
        kw...)

    # do the learning.  our iterator just repeatedly gives us the environment
    learn!(policy, learner, repeated(env))

    @show policy strat
    policy, strat
end


end #module
