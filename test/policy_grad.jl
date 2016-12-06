module pgrad

# this is a port of the example found at
#	http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/lab1.html#starter

using Reinforce
using OpenAIGym
# using Distributions
using Learn

using MLPlots; gr(size=(1400,1400))

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

# type Critic
#     return_func::Function
#     baseline_func::Function

# ----------------------------------------------------------------



# initialize a policy, do the learning, then return the policy
function doit(sublearners...; env = GymEnv("BipedalWalker-v2"),
                       maxsteps = 500, # max number of steps in one episode
                       maxiter = 1000, # max learning iterations
					   kw...)
    s = state(env)
    ns = length(s)
    A = actions(env,s)
    nA = length(A)
    @show s ns

    # create a stochastic policy which can sample actions from a multivariate normal dist

    # create a multivariate normal transformation with underlying params μ/σ
    μ = zeros(nA)

    # diagonal
    σ = zeros(nA)
    D = MvNormalTransformation(μ, σ)
    nϕ = 2nA

    # # upper-triangular
    # U = eye(nA,nA)
    # D = MvNormalTransformation(μ, U)
    # nϕ = nA*(nA+1)

    @show D

    # create a neural net mapping: s --> ϕ = vec(μ,U) of the MvNormal
    nh = Int[30,20]
    ϕ = nnet(ns, nϕ, nh, :relu, :identity)
    @show ϕ

    # the critic's value function... mapping state to value
    C = nnet(ns, 1, nh, :relu, :identity)
    @show C

    # our discount rates # TODO: can we learn these too??
    γ = 0.9
    λ = 0.6

    # this is a stochastic policy which follows http://www.breloff.com/DeepRL-OnlineGAE/
    policy = OnlineGAE(A, ϕ, D, C, γ, λ,
                       OnlineGradAvg(400, lr=0.1, pu=Adadelta()),
                       OnlineGradAvg(200, lr=0.1, pu=Adadelta()),
                    #    penalty = ElasticNetPenalty(1e-1,0.5)
                        penalty = L2Penalty(1e-4)
                      )

    # --------------------------------
    # set up the custom visualizations

    # chainplots... put one for each of ϕ/C side-by-side
    cp_ϕ = ChainPlot(ϕ)
    cp_C = ChainPlot(C)

    # this will be called on every timestep of every episode
    function eachiteration(ep,i)
        @show i, ep.total_reward
        update!(cp_ϕ)
        update!(cp_C)
        hm1 = heatmap(reshape(D.dist.μ,nA,1), yflip=true,
                    title=string(maximum(D.dist.μ)),
                    xguide=string(minimum(D.dist.μ)),
                    left_margin=150px)
        # Σ = UpperTriangular(D.dist.Σ.chol.factors)
        # Σ = Diagonal(D.dist.Σ.diag)
        Σ = Diagonal(abs.(output_value(ϕ)[nA+1:end]))
        hm2 = heatmap(Σ, yflip=true,
                    title=string(maximum(Σ)),
                    xguide=string(minimum(Σ)))
        plot(cp_ϕ.plt, cp_C.plt, hm1, hm2, layout = @layout([ϕ; C; hm1{0.2w,0.2h} hm2]))
        # i%2000==0 && gui()
        gui()
    end

    function renderfunc(ep,i)
        if mod1(ep.niter, 10) == 1
            OpenAIGym.render(env, ep.niter, nothing)
        end
    end


    learn!(policy, Episodes(
        env,
        freq = 5,
        episode_strats = [MaxIter(1000)],
        epoch_strats = [MaxIter(5000), IterFunction(renderfunc, every=5)],
        iter_strats = [IterFunction(eachiteration, every=1000)]
    ))

    env, policy
end


end #module
