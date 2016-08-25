
export
    CrossEntropyMethod,
    CrossEntropyMethodState

@with_kw type CrossEntropyMethod <: IM.IterationManager
    f::Function # maps θ --> r
    noise_func::Function = t->0.0 # additional deviation at each timestep
    maxiter::Int = 200
    cem_iter::Int = 100
    cem_batchsize::Int = 20
    cem_elitefrac::Float64 = 0.2
    stopping_norm::Float64 = 1e-2
    doanim::Bool = false
    dotrace::Bool = true
    doplot::Bool = true
end
CrossEntropyMethod(f::Function; kw...) = CrossEntropyMethod(; f=f, kw...)

@with_kw type CrossEntropyMethodState <: IM.InterationState
    n::Int
    t::Int = 0
    μ::Vector{Float64} = zeros(n)
    last_μ::Vector{Float64} = zeros(n)
    σ::Vector{Float64} = ones(n)
    Z::Vector{Float64} = zeros(n) # extra variance
    anim = nothing
    h::MVHistory = MVHistory()
end
   
# do something before the iterations start 
function IM.pre_hook(mgr::CrossEntropyMethod, istate::CrossEntropyMethodState)
    istate.anim = mgr.doanim ? Animation() : nothing
end

# # the core loop, act/step in a simulation, update the env state, get a reward, update a model, etc
function IM.update!(mgr::CrossEntropyMethod, istate::CrossEntropyMethodState)
    istate.last_μ = copy(istate.μ)
    istate.t += 1

    # sample thetas from a multivariate normal distribution
    N = MultivariateNormal(istate.μ, istate.σ)
    θs = [rand(N) for k=1:mgr.cem_batchsize]

    # compute rewards and pick out an elite set
    Rs = map(mgr.f, θs)
    n_elite = round(Int, mgr.cem_batchsize * mgr.cem_elitefrac)
    elite_indices = sortperm(Rs, rev=true)[1:n_elite]
    elite_θs = θs[elite_indices]
    info("Iteration $t. mean(R): $(mean(Rs)) max(R): $(maximum(Rs))")

    # update the policy from the elite set
    for j=1:length(istate.μ)
        θj = [θ[j] for θ in elite_θs]
        istate.μ[j] = mean(θj)
        istate.Z[j] = mgr.noise_func(t)
        istate.σ[j] = sqrt(var(θj) + istate.Z[j])
    end
    @show istate.μ istate.σ istate.Z

    if mgr.dotrace
        hist_min = minimum(Rs)
        hist_mean = mean(Rs)
        hist_max = maximum(Rs)
        @trace istate.t hist_min hist_mean hist_max
    end
end

# do something for each iteration, but after update! has finished
function IM.iter_hook(mgr::CrossEntropyMethod, istate::CrossEntropyMethodState)

    # TODO: handle the plotting/anim here by overriding this method?

    # # finish the iteration by evaluating an episode with θ = μ
    # R, T = episode!(
    #     env,
    #     cem_policy(env, solver.μ),
    #     maxiter = solver.options[:maxiter],
    #     stepfunc = myplot(t, hist_min, hist_mean, hist_max, anim)
    # )
    # info("Iteration $t finished. Total reward: $R")
end

# are we done iterating?  check for convergence, etc
function IM.finished(mgr::CrossEntropyMethod, istate::CrossEntropyMethodState)
    normdiff = norm(istate.μ - istate.last_μ)
    if normdiff < mgr.stopping_norm
        info("Converged after $(istate.t * mgr.cem_batchsize) episodes.")
        return true
    end
    istate.t >= mgr.cem_iter
end

# iterations are done... do something at the very end
function IM.post_hook(mgr::CrossEntropyMethod, istate::CrossEntropyMethodState)
    if mgr.doanim
        gif(istate.anim)
    end
end
