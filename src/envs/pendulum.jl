
# Ported from: https://github.com/openai/gym/blob/996e5115621bf57b34b9b79941e629a36a709ea1/gym/envs/classic_control/pendulum.py

const max_speed = 8.0
const max_torque = 2.0

angle_normalize(x) = ((x+π) % (2π)) - π


type Pendulum <: AbstractEnvironment
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

Base.done(env::Pendulum) = env.steps >= env.maxsteps
actions(env::Pendulum, s) = ContinuousActionSet(-max_torque, max_torque)

# ------------------------------------------------------------------------

@recipe function f(env::Pendulum, t, iter, hists)
	@eval import Plots
	subplot := 2
	layout := 2
	legend := false

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
		xlims := (-1,1)
		ylims := (-1,1)
		grid := false
		ticks := nothing
		annotations := [(0, -0.2, Plots.text("a: $(env.a)", :top))]
		[0],[0]
	end

	# # pole
	# @series begin
	# 	linecolor := :red
	# 	linewidth := 10
	# 	[x, x + 2pole_length * sin(θ)], [0.0, 2pole_length * cos(θ)]
	# end

	# # cart
	# @series begin
	# 	seriescolor := :black
	# 	seriestype := :shape
	# 	hw = 0.2pole_length
	# 	xlims := (-x_threshold, x_threshold)
	# 	ylims := (-Inf, 2pole_length)
	# 	grid := false
	# 	ticks := nothing
	# 	if iter > 0
	# 		title := "Episode: $t  Iter: $iter"
	# 	end
	# 	hw = 0.5
	# 	l, r = x-hw, x+hw
	# 	t, b = 0.0, -0.1
	# 	[l, r, r, l], [t, t, b, b]
	# end

	subplot := 1
	title := "Progress"
	@series begin
		linecolor := :black
		fillrange := (hists[1], hists[3])
		fillcolor := :black
		fillalpha := 0.2
		hists[2]
	end
end