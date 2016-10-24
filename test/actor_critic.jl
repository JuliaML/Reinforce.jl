module pgrad

# this is a port of the example found at
#	http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/lab1.html#starter

using Reinforce
using OpenAIGym
# using Distributions
using Transformations
using StochasticOptimization
using Penalties

using MLPlots; gr(size=(1400,1400))

# ----------------------------------------------------------------

# initialize a policy, do the learning, then return the policy
function doit(env = GymEnv("BipedalWalker-v2"))
    s = state(env)
    ns = length(s)
    A = actions(env,s)
    nA = length(A)
    @show s ns

    policy = ActorCritic(s, nA,
        γ = 0.9,
        λ = 0.6,
        αᵛ = 0.3,
        αᵘ = 0.1,
    )

    # --------------------------------
    # set up the custom visualizations

    # chainplots... put one for each of ϕ/C side-by-side
    ϕ = policy.u
    D = policy.D
    cp_ϕ = ChainPlot(ϕ)
    # cp_C = ChainPlot(C)

    # this will be called on every timestep of every episode
    function eachiteration(ep,i)
        @show i, ep.total_reward
        update!(cp_ϕ)
        # update!(cp_C)
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
        plot(cp_ϕ.plt, hm1, hm2, layout = @layout([ϕ; hm1{0.2w,0.2h} hm2]))
        gui()
    end

    function renderfunc(ep,i)
        if mod1(ep.niter, 1) == 1
            OpenAIGym.render(env, ep.niter, nothing)
        end
    end


    learn!(policy, Episodes(
        env,
        freq = 5,
        episode_strats = [MaxIter(1000)],
        epoch_strats = [MaxIter(5000), IterFunction(renderfunc, every=3)],
        iter_strats = [IterFunction(eachiteration, every=1000)]
    ))

    env, policy
end


end #module
