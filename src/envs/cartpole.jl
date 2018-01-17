
# Ported from: https://github.com/openai/gym/blob/996e5115621bf57b34b9b79941e629a36a709ea1/gym/envs/classic_control/cartpole.py
# which has header:
# 		Classic cart-pole system implemented by Rich Sutton et al.
#		Copied from https://webdocs.cs.ualberta.ca/~sutton/book/code/pole.c

const gravity = 9.8
const mass_cart = 1.0
const mass_pole = 0.1
const total_mass = mass_cart + mass_pole
const pole_length = 0.5 # actually half the pole's length
const mass_pole_length = mass_pole * pole_length
const force_mag = 10.0
const τ = 0.02 # seconds between state updates

# angle at which to fail the episode
const θ_threshold = 24π / 360
const x_threshold = 2.4

mutable struct CartPole <: AbstractEnvironment
	state::Vector{Float64}
	reward::Float64
end
CartPole() = CartPole(0.1rand(4)-0.05, 0.0)

reset!(env::CartPole)  = (env.state = 0.1rand(4)-0.05; env.reward = 0.0; return)
actions(env::CartPole, s) = DiscreteSet(1:2)

function step!(env::CartPole, s, a)
	s = state(env)
	x, xvel, θ, θvel = s

	force = (a == 1 ? -1 : 1) * force_mag
	tmp = (force + mass_pole_length * sin(θ) * (θvel^2)) / total_mass
	θacc = (gravity * sin(θ) - tmp * cos(θ)) /
			(pole_length * (4/3 - mass_pole * (cos(θ)^2) / total_mass))
	xacc = tmp - mass_pole_length * θacc * cos(θ) / total_mass

	# update state
	s[1] = x    += τ * xvel
	s[2] = xvel += τ * xacc
	s[3] = θ    += τ * θvel
	s[4] = θvel += τ * θacc

	env.reward = finished(env, s) ? 0.0 : 1.0
	env.reward, s
end

function finished(env::CartPole, s′)
	x, xvel, θ, θvel = s′
	!(-x_threshold <= x <= x_threshold &&
	  -θ_threshold <= θ <= θ_threshold)
end


# ------------------------------------------------------------------------

@recipe function f(env::CartPole)
    x, xvel, θ, θvel = state(env)
    legend := false
    xlims := (-x_threshold, x_threshold)
    ylims := (-Inf, 2pole_length)
    grid := false
    ticks := nothing

    # pole
    @series begin
        linecolor := :red
        linewidth := 10
        [x, x + 2pole_length * sin(θ)], [0.0, 2pole_length * cos(θ)]
    end

    # cart
    @series begin
        seriescolor := :black
        seriestype := :shape
        hw = 0.5
        l, r = x-hw, x+hw
        t, b = 0.0, -0.1
        [l, r, r, l], [t, t, b, b]
    end
end
