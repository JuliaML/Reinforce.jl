module pgrad

# this is a port of the example found at
#	http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/lab1.html#starter

using Reinforce
using OpenAIGym
# using Distributions
using Transformations
using StochasticOptimization

using MLPlots; gr(size=(500,500))

# ----------------------------------------------------------------

# """
# Wraps a LearnBase.Transformation which converts an input vector to action values.
# For discrete actions, it chooses the action which produces the highest value.
# For continuous (interval) actions, it squashes actions to [0,1].
# """
# immutable TransformPolicy{T} <: AbstractPolicy
# 	trans::T
# end
#
# Transformations.params(tp::TransformPolicy) = params(tp.trans)
#
# get_action(A::DiscreteSet, ŷ::AbstractVector) = A[indmax(ŷ)]
# function get_action(A::IntervalSet, ŷ)
#     val = 0.5 * (1.0 + clamp(first(ŷ), -1.0, 1.0))
#     val * (A.hi - A.lo) + A.lo
# end
#
# # # discrete: our action is the action which maximizes the affine transform
# # function Reinforce.action(π::TransformPolicy, r, s, A::DiscreteSet)
# # 	A[indmax(transform!(π.trans, s))]
# # end
#
# # most actionsets pass through to get_action
# Reinforce.action(π::TransformPolicy, r, s, A) = get_action(A)
#
# # get an action where each part of the TupleSet is associated with part of the
# # transformation output
# function Reinforce.action(π::TransformPolicy, r, s, A::TupleSet)
#     ŷ = transform!(π.trans, s)
#     a = []
#     i = 0
#     for Aᵢ in A
#         nᵢ = length(Aᵢ)
#         push!(a, get_action(Aᵢ, view(ŷ, i+1:i+nᵢ)))
#         i += nᵢ
#     end
#     # @show a
#     a
# end
#
# # # continuous: return the transform value, squashed to [0,1]
# # # TODO: remove the squashing when a "Affine + Sigmoid" transformation is available
# # function Reinforce.action(π::TransformPolicy, r, s, A::IntervalSet)
# # 	Transformations.sigmoid(transform(π.trans, s)[1]) * (A.amax-A.amin) + A.amin
# # end

# ----------------------------------------------------------------

# function myplot(t, hist_min, hist_mean, hist_max, anim=nothing)
# 	(env,i,sars) -> if mod1(t,3)==1 && mod1(i,10)==1
# 		plot(
# 			plot(hist_mean, c=:black, fill=((hist_min,hist_max), 0.2), title="Progress", leg=false),
# 			plot(env, title = "Episode: $t  Iter: $i")
# 		)
# 		if anim == nothing
# 			gui()
# 		else
# 			frame(anim)
# 		end
# 	else
# 		return
# 	end
# end

# ----------------------------------------------------------------



# initialize a policy, do the learning, then return the policy
function doit(sublearners...; env = GymEnv("BipedalWalker-v2"),
                       maxsteps = 500, # max number of steps in one episode
                       maxiter = 1000, # max learning iterations
					   # noise_max = 1.0,
					   # noise_steps = 20,
					   # noise_func = t -> max(noise_max - t/noise_steps, 0.0),
					   kw...)
    s = state(env)
    nin = length(s)
    @show s nin

    # create a stochastic policy which can sample actions from a multivariate normal dist
    A = actions(env,s)
    nA = length(A)
    # policy = StochasticPolicy(A)
    # @show A policy

    # create a neural net mapping states to μ/Σ of the MvNormal
    nϕ = nA * (nA+1)  # size of μ + size of Z
    nn = nnet(nin, nϕ, [5], :softplus)
    # Transformations.link_nodes!(nn.output, policy.input)
    @show nn

    # this is a stochastic policy which learns the parameters of a
    # multivariate normal distribution for each state
    policy = StochasticPolicy(A, nn)

    # update the policy from the state, then sample actions
    a = action(policy, 0.0, s, A)
    @show a

    # tp = TracePlot(2, layout=@layout([a;b{0.2h}]))
    # tracer = IterFunction((policy,i) -> begin
    #     mod1(i,10)==1 || return
    #
    #     #run one episode
    #     R,N = episode!(env, policy, stepfunc = OpenAIGym.render)
    #     @show R, N
    #     push!(tp, i, [R,N])
    #     gui(tp.plt)
    # end)

    # learner = make_learner(cem, tracer, sublearners...; maxiter=maxiter, kw...)
    # learn!(policy, learner, repeated(env))

end


end #module
