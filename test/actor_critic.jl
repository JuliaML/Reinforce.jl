module pgrad

# this is a port of the example found at
#	http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/lab1.html#starter

using Reinforce
using OpenAIGym
# using Distributions
using Transformations
using StochasticOptimization
using Penalties

using MLPlots; gr(size=(1400,1400), leg=false)

# ----------------------------------------------------------------

# initialize a policy, do the learning, then return the policy
function doit(env = GymEnv("BipedalWalker-v2"))
    s = state(env)
    ns = length(s)
    A = actions(env,s)
    nA = length(A)
    @show s ns

    policy = ActorCritic(s, nA,
        # algo = :INAC,
        penalty = L2Penalty(1e-4),
        γ = 0.5,
        λ = 0.5,
        # αʳ = 0.0001,
        αᵛ = 0.05,
        αᵘ = 0.01,
    )

    # --------------------------------
    # set up the custom visualizations

    # chainplots... put one for each of ϕ/C side-by-side
    ϕ = policy.ϕ
    D = policy.D

    cp_ϕ = ChainPlot(ϕ)
    # nϕplts = length(cp_ϕ.plt.subplots)
    # cp_C = ChainPlot(C)
    # μ = reshape(D.dist.μ,nA,1)
    # hm1 = heatmap(μ, yflip=true,
    #             title=string(maximum(D.dist.μ)),
    #             xguide=string(minimum(D.dist.μ)),
    #             left_margin=150px)
    # Σ = UpperTriangular(D.dist.Σ.chol.factors)
    # Σ = Diagonal(D.dist.Σ.diag)
    # Σ = Diagonal(abs.(output_value(ϕ)[nA+1:end]))
    # hm2 = heatmap(Σ, yflip=true,
    #             title=string(maximum(Σ)),
    #             xguide=string(minimum(Σ)))
    tpplt = plot(layout=grid(4,2))
    tp_δ = TracePlot(1, sp=tpplt[1], title="delta")
    tp_r̄ = TracePlot(1, sp=tpplt[2], title="r")
    tp_eᵛ = TracePlot(ns, sp=tpplt[3], title="ev")
    tp_v = TracePlot(ns, sp=tpplt[4], title="v")
    tp_eμ = TracePlot(nA, sp=tpplt[5], title="e_mu")
    tp_μ = TracePlot(nA, sp=tpplt[6], title="mu")
    tp_eσ = TracePlot(nA, sp=tpplt[7], title="e_sigma")
    tp_σ = TracePlot(nA, sp=tpplt[8], title="sigma")
    plt = plot(cp_ϕ.plt, tpplt, layout=grid(2,1,heights=[0.3,0.7]))

    # this will be called on every timestep of every episode
    function eachiteration(ep,i)
        @show i, ep.total_reward
        # i<5000 && return

        # i>=2000 && (policy.αᵘ = 0.0005)
        update!(cp_ϕ)
        # update!(cp_C)
        for (tp,var) in [(tp_δ, policy.δ), (tp_r̄, mean(policy.r̄)),
                         (tp_eᵛ, policy.eᵛ), (tp_v, policy.v),
                         (tp_eμ, policy.eᵘ[1:nA]), (tp_μ, output_value(ϕ)[1:nA]),
                         (tp_eσ, policy.eᵘ[nA+1:end]), (tp_σ, output_value(ϕ)[nA+1:end]),
                        ]
            push!(tp, i, var)
        end
        # hm1 = heatmap(reshape(D.dist.μ,nA,1), yflip=true,
        #             title=string(maximum(D.dist.μ)),
        #             xguide=string(minimum(D.dist.μ)),
        #             left_margin=150px)
        # # Σ = UpperTriangular(D.dist.Σ.chol.factors)
        # # Σ = Diagonal(D.dist.Σ.diag)
        # Σ = Diagonal(abs.(output_value(ϕ)[nA+1:end]))
        # hm2 = heatmap(Σ, yflip=true,
        #             title=string(maximum(Σ)),
        #             xguide=string(minimum(Σ)))
        # plot(cp_ϕ.plt, hm1, hm2,
        #     tp_δ.plt, tp_r̄.plt, tp_eᵛ.plt, tp_v.plt, tp_eᵘ.plt,
        #     layout = @layout([ϕ{0.3h}; hm1{0.2w,0.2h} hm2; d; grid(2,2)])
        # )
        gui()
    end

    function renderfunc(ep,i)
        if mod1(ep.niter, 1) == 1
            OpenAIGym.render(env, ep.niter, nothing)
        end
    end


    learn!(policy, Episodes(
        env,
        freq = 2,
        episode_strats = [MaxIter(500)],
        epoch_strats = [MaxIter(5000), IterFunction(renderfunc, every=3)],
        iter_strats = [IterFunction(eachiteration, every=500)]
    ))

    env, policy
end


end #module
