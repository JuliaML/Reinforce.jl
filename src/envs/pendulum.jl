
# Ported from: https://github.com/openai/gym/blob/996e5115621bf57b34b9b79941e629a36a709ea1/gym/envs/classic_control/pendulum.py

const max_speed = 8.0
const max_torque = 2.0

angle_normalize(x) = ((x+π) % (2π)) - π


mutable struct Pendulum <: AbstractEnvironment
	θs::Tuple{Float64,Float64}
	reward::Float64
	a::Float64 # last action for rendering
	steps::Int
	maxsteps::Int
end
Pendulum(maxsteps=500) = Pendulum((0.,0.),0.,0.,0,maxsteps)

function reset!(env::Pendulum)
	env.θs = (rand(Uniform(-π, π)), rand(Uniform(-1., 1.)))
	env.reward = 0.0
	env.a = 0.0
	env.steps = 0
	return
end

actions(env::Pendulum, s) = IntervalSet(-max_torque, max_torque)

function step!(env::Pendulum, s, a)
	θ, θvel = env.θs
	g = 10.0
	m = 1.0
	l = 1.0
	dt = 0.05

	env.a = a
	a = clamp(a, -max_torque, max_torque)
	env.reward = -(angle_normalize(θ)^2 + 0.1θvel^2 + 0.001a^2)

	# update state
	newθvel = θvel + (-1.5g/l * sin(θ+π) + 3/(m*l^2)*a) * dt
	newθ = θ + newθvel * dt
	newθvel = clamp(newθvel, -max_speed, max_speed)
	env.θs = newθ, newθvel

	env.steps += 1
	env.reward, state(env)
end


function state(env::Pendulum)
	θ, θvel = env.θs
	Float64[cos(θ), sin(θ), θvel]
end

finished(env::Pendulum, s′) = env.steps >= env.maxsteps

# ------------------------------------------------------------------------

@recipe function f(env::Pendulum)
	legend := false
    xlims := (-1,1)
    ylims := (-1,1)
    grid := false
    ticks := nothing

	# pole
	@series begin
		w = 0.2
		x = [-w,w,w,-w]
		y = [-.1,-.1,1,1]
		θ = env.θs[1]
		fillcolor := :red
		seriestype := :shape
		x*cos(θ) - y*sin(θ), y*cos(θ) + x*sin(θ)
	end

	# center
	@series begin
		seriestype := :scatter
		markersize := 10
		markercolor := :black
		annotations := [(0, -0.2, Plots.text("a: $(env.a)", :top))]
		[0],[0]
	end
end
