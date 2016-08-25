
import IterationManagers
const IM = IterationManagers
using Parameters

# TODO: implement this for each solver:

# function managed_iteration!{T<:AbstractArray}(f!::Base.Callable,
#                                               mgr::IterationManager,
#                                               dest::T,
#                                               istate::IterationState{T};
#                                               by::Base.Callable=default_by)
#     pre_hook(mgr, istate)

#     while !(finished(mgr, istate))
#         f!(dest, istate.prev)
#         update!(istate, dest; by=by)
#         iter_hook(mgr, istate)
#     end

#     post_hook(mgr, istate)
#     istate
# end

# TODO: add this to IM
function IM.managed_iteration!(mgr::IM.IterationManager, istate::IM.IterationState)
    IM.pre_hook(mgr, istate)

    while !(IM.finished(mgr, istate))
        IM.update!(mgr, istate)
        IM.iter_hook(mgr, istate)
    end

    IM.post_hook(mgr, istate)
    istate
end

function LearnBase.learn!(mgr::IM.IterationManager, istate::IM.IterationState)
    IM.managed_iteration!(mgr, istate)
end

include("solvers/cem.jl")
