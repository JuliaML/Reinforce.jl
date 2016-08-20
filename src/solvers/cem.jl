
export
    CrossEntropyMethod,
    CrossEntropyMethodState

@with_kw type CrossEntropyMethod <: IM.IterationManager
    noise_func::Function = t->0.0 # additional deviation at each timestep
    maxiter::Int = 200
    cem_iter::Int = 100
    cem_batchsize::Int = 20
    cem_elitefrac::Float64 = 0.2
    stopping_norm::Float64 = 1e-2
    doanim::Bool = false
end

@with_kw type CrossEntropyMethodState <: IM.InterationState
    n::Int
    μ::Vector{Float64} = zeros(n)
    σ::Vector{Float64} = ones(n)
    Z::Vector{Float64} = zeros(n) # extra variance
end
   
# do something before the iterations start 
function IM.pre_hook(mgr::CrossEntropyMethod, istate::CrossEntropyMethodState)
end

# function IM.update!(istate::CrossEntropyMethodState, dest; by::Base.Callable = IM.default_by)
# end

# # the core loop, act/step in a simulation, update the env state, get a reward, update a model, etc
function IM.update!(mgr::CrossEntropyMethod, istate::CrossEntropyMethodState)
end

# do something for each iteration, but after update! has finished
function IM.iter_hook(msg::CrossEntropyMethod, istate::CrossEntropyMethodState)
end

# are we done iterating?  check for convergence, etc
function IM.finished(mgr::CrossEntropyMethod, istate::CrossEntropyMethodState)
end

# iterations are done... do something at the very end
function IM.post_hook(mgr::CrossEntropyMethod, istate::CrossEntropyMethodState)
end
