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
    ns = length(s)
    A = actions(env,s)
    nA = length(A)
    @show s ns

    # create a stochastic policy which can sample actions from a multivariate normal dist

    # create a multivariate normal transformation with underlying params μ/σ
    μ = zeros(nA)
    σ = zeros(nA)  # diagonal
    D = MvNormalTransformation(μ, σ)
    @show D

    # create a neural net mapping: s --> ϕ = vec(μ,U) of the MvNormal
    nϕ = 2nA
    ϕ = nnet(ns, nϕ, [5], :softplus, :identity)
    @show ϕ

    # the critic's value function... mapping state to value
    C = nnet(ns, 1, [5], :softplus, :identity)

    # our discount rates # TODO: can we learn these too??
    γ = 0.95
    λ = 0.8

    # this is a stochastic policy which follows http://www.breloff.com/DeepRL-OnlineGAE/
    policy = OnlineGAE(A, ϕ, D, C, γ, λ)

    # this is our sub-learner to manage episode state
    episodes = EpisodeLearner(env, MaxIter(2000))

    # our main metalearner.
    #   stop after 500 total steps or 1 minute
    #   render the frame each iteration
    learner = make_learner(
        GradientLearner(1e-1, Adamax()),
        episodes,
        MaxIter(100000),
        TimeLimit(60),
        # ShowStatus(100),
        IterFunction((m,i) -> begin
            @show i, norm(params(policy))
            OpenAIGym.render(env, i, nothing)
        end, 200)
    )

    # our metalearner will infinitely take a step in an episode,
    learn!(policy, learner)

    # R,N = episode!(env, policy, stepfunc = OpenAIGym.render)
    # @show R,N

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

    env, policy
end


end #module
