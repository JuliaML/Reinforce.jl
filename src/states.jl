
abstract type AbstractState end

# ----------------------------------------------------------------

"A StateVector holds both the functions which will populate the state, and the most recent state."
mutable struct StateVector{S} <: AbstractState
	queries::Vector{Function}
	state::Vector{S}
	names::Vector{String}
end

function StateVector(queries::AbstractVector{Function}; names=fill("",length(queries)))
	StateVector(queries, zeros(length(queries)), names)
end
function StateVector(queries::Function...; names=fill("",length(queries)))
	StateVector(Function[f for f in queries], names=names)
end

Base.length(sv::StateVector) = length(sv.queries)

"retreive the most recently calculated state"
state(sv::StateVector) = sv.state

"update the state, then return it"
function state!(sv::StateVector)
	for (i,f) in enumerate(sv.queries)
		sv.state[i] = f()
	end
	sv.state
end

# ----------------------------------------------------------------

mutable struct History{T}
    sv::StateVector{T}
    states::Matrix{T}
end
History(sv::StateVector{T}) where {T} = History(sv, Matrix{T}(length(sv),0))

function state!(hist::History)
	s = state!(hist.sv)
	hist.states = hcat(hist.states, s)
	s
end

StatsBase.nobs(hist::History) = size(hist.states, 2)


# ----------------------------------------------------------------
