module pgrad

using Reinforce
using OpenAIGym
using Transformations
using StochasticOptimization
using PenaltyFunctions

using MLPlots; gr(size=(1400,1400), leg=false)

# ----------------------------------------------------------------

# initialize a policy, do the learning, then return the policy
function doit(env = GymEnv("BipedalWalker-v2"))
    s = state(env)
    ns = length(s)
    A = actions(env,s)
    nA = length(A)
    @show s ns

    policy = OnlineActorCritic(s, nA,
        # ϕ = nnet(2ns+nA, 2nA, [100], :softplus, lookback=10000),
        ϕ = resnet(2ns+nA, 2nA, 1, nh=[100], inner_activation=:softplus, lookback=10000),
        penalty = L2Penalty(1e-5),
        γ = 0.995,
        λ = 0.95,
        # αʳ = 0.0001,
        # αᵛ = 0.01,
        # αᵘ = 0.01,
        gaᵛ = OnlineGradAvg(50, lr=0.1, pu=RMSProp()),
        gaᵘ = OnlineGradAvg(50, lr=0.01, pu=RMSProp()),
        # gaʷ = OnlineGradAvg(50, lr=0.5, pu=Adamax())
    )

    # --------------------------------
    # set up the custom visualizations

    # chainplots... put one for each of ϕ/C side-by-side
    ϕ = policy.actor.ϕ
    D = policy.actor.D

    cp_ϕ = ChainPlot(ϕ)
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

        policy.gaᵛ.lr *= 0.999
        policy.gaᵘ.lr *= 0.999

        # i>=2000 && (policy.αᵘ = 0.0005)
        update!(cp_ϕ)
        for (tp,var) in [(tp_δ, policy.δ), (tp_r̄, mean(policy.r̄)),
                         (tp_eᵛ, policy.eᵛ), (tp_v, policy.v),
                         (tp_eμ, policy.eᵘ[1:nA]), (tp_μ, output_value(ϕ)[1:nA]),
                         (tp_eσ, policy.eᵘ[nA+1:end]), (tp_σ, output_value(ϕ)[nA+1:end]),
                        ]
            push!(tp, i, var)
        end
        gui()
    end

    function renderfunc(ep,i)
        policy.actor.testing = true
        if mod1(ep.niter, 1) == 1
            OpenAIGym.render(env, ep.niter, nothing)
        end
        policy.actor.testing = false
    end


    learn!(policy, Episodes(
        env,
        freq = 1,
        episode_strats = [MaxIter(1000)],
        epoch_strats = [MaxIter(10000), IterFunction(renderfunc, every=5)],
        iter_strats = [IterFunction(eachiteration, every=1000)]
        # append_action = true
    ))

    env, policy
end


end #module
