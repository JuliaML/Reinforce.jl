
export
    CrossEntropyMethod

# ----------------------------------------------------------------------------

# TODO: add this to MLPlots??

# export
#     AnimationStrategy

# # add this to your MasterLearner to save Plots animations of your learning process
# type AnimationStrategy <: LearningStrategy
#     anim::Animation
#     f::Function
# end
# AnimationStrategy(f::Function) = AnimationStrategy(Animation(), f)
# iter_hook(strat::AnimationStrategy, policy, i) = (strat.f(policy, i); frame(strat.anim))
# post_hook(strat::AnimationStrategy, policy) = gif(strat.anim)

# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------

# TODO: this is a strategy and policy combined... they should really be split into a Learnable and a LearningStrategy
mutable struct CrossEntropyMethod <: LearningStrategy
    noise_func::Function    # additional deviation at each timestep
    maxsteps::Int           # max num steps in one episode
    cem_batchsize::Int
    cem_elitefrac::Float64
    stopping_norm::Float64
    t::Int
    n::Int
    μ::Vector{Float64}
    last_μ::Vector{Float64}
    σ::Vector{Float64}
    Z::Vector{Float64}      # extra variance
    CrossEntropyMethod(nf, iter, bs, ef, sn) = new(nf, iter, bs, ef, sn, 0)
end
function CrossEntropyMethod(; #f::Function;
           noise_func = t->0.0,
           maxsteps::Int = 100,
           cem_batchsize::Int = 20,
           cem_elitefrac::Float64 = 0.2,
           stopping_norm::Float64 = 1e-2)
    CrossEntropyMethod(noise_func, maxsteps, cem_batchsize, cem_elitefrac, stopping_norm)
end

function pre_hook(strat::CrossEntropyMethod, policy)
    n = length(params(policy))
    strat.n = n
    strat.μ = zeros(n)
    strat.last_μ = zeros(n)
    strat.σ = ones(n)
    strat.Z = zeros(n)
    return
end

# # the core loop, act/step in a simulation, update the env state, get a reward, update a policy, etc
function learn!(policy::AbstractPolicy, strat::CrossEntropyMethod, env::AbstractEnvironment)
    strat.last_μ = copy(strat.μ)
    strat.t += 1

    # sample thetas from a multivariate normal distribution
    N = MultivariateNormal(strat.μ, strat.σ)
    θs = [rand(N) for k=1:strat.cem_batchsize]

    # overwrite the parameters of the policy and run an episode for each θ
    Rs = map(θ -> begin
        params(policy)[:] = θ
        R = 0
        for (i,sars) in enumerate(Episode(env,policy))
            R += sars[3]
            i < strat.maxsteps || break
        end
        # R, T = episode!(env, policy; maxsteps = strat.maxsteps)
        R
    end, θs)

    # pick out the elite set
    n_elite = round(Int, strat.cem_batchsize * strat.cem_elitefrac)
    elite_indices = sortperm(Rs, rev=true)[1:n_elite]
    elite_θs = θs[elite_indices]
    info("Iteration $(strat.t). mean(R): $(mean(Rs)) max(R): $(maximum(Rs)) ‖μ‖²: $(norm(strat.μ)) ‖σ‖²: $(norm(strat.σ))")

    # update the policy from the empirical statistics of the elite set
    for j=1:length(strat.μ)
        θj = [θ[j] for θ in elite_θs]
        strat.μ[j] = mean(θj)
        strat.Z[j] = strat.noise_func(strat.t)
        strat.σ[j] = sqrt(var(θj) + strat.Z[j])
    end
    # @show strat.μ strat.σ strat.Z
end


# are we done iterating?  check for convergence, etc
function finished(strat::CrossEntropyMethod, policy::AbstractPolicy, i::Int)
    strat.t > 0 || return false
    normdiff = norm(strat.μ - strat.last_μ)
    if normdiff < strat.stopping_norm
        info("Converged after $(strat.t * strat.cem_batchsize) episodes.")
        return true
    end
    false
end
