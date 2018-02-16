# Ported from https://github.com/openai/gym/blob/996e5115621bf57b34b9b79941e629a36a709ea1/gym/envs/classic_control/mountain_car.py
# which has header
#  https://webdocs.cs.ualberta.ca/~sutton/MountainCar/MountainCar1.cp

module MountainCarEnv

using Reinforce.AbstractEnvironment
using LearnBase.DiscreteSet
using RecipesBase
using Distributions

import Reinforce: reset!, actions, finished, step!

export
  MountainCar,
  reset!,
  step!,
  actions,
  finished,
  f

const min_position = -1.2
const max_position = 0.6
const max_speed = 0.07
const goal_position = 0.5
const min_start = -0.6
const max_start = 0.4

const car_width = 0.05
const car_height = car_width/2.0
const clearance = 0.2*car_height
const flag_height = 0.05

mutable struct MountainCarState
  position::Float64
  velocity::Float64
end

mutable struct MountainCar <: AbstractEnvironment
  state::MountainCarState
  reward::Float64
  seed::Int
end
MountainCar(seed=-1) = MountainCar(MountainCarState(0.0, 0.0), 0.0, seed)

function reset!(env::MountainCar)
  if env.seed >= 0
    srand(env.seed)
    env.seed = -1
  end

  env.state.position = rand(Uniform(min_start, max_start))
  env.state.velocity = 0.0

  return
end

actions(env::MountainCar, s) = DiscreteSet(1:3)
finished(env::MountainCar, s′) = env.state.position >= goal_position

function step!(env::MountainCar, s::MountainCarState, a::Int)
  position = env.state.position
  velocity = env.state.velocity
  velocity += (a - 2)*0.001 + cos(3*position)*(-0.0025)
  velocity = clamp(velocity, -max_speed, max_speed)
  position += velocity
  if position <= min_position && velocity < 0
    velocity = 0
  end
  position = clamp(position, min_position, max_position)
  env.state = MountainCarState(position, velocity)
  env.reward = -1

  return env.reward, env.state
end

# ------------------------------------------------------------------------
height(xs) = sin(3 * xs)*0.45 + 0.55
rotate(xs::Array{Float64}, ys::Array{Float64}, Θ::Float64) =
  xs*cos(Θ) - ys*sin(Θ), ys*cos(Θ) + xs*sin(Θ)

translate(xs::Array{Float64}, ys::Array{Float64}, t::Array{Float64}) =
  xs + t[1], ys + t[2]

@recipe function f(env::MountainCar)
  legend := false
  xlims := (min_position, max_position)
  ylims := (0, 1.1)
  grid := false
  ticks := nothing

  # Mountain
  @series begin
    xs = linspace(min_position, max_position, 100)
    ys = height(xs)
    seriestype := :path
    linecolor --> :blue
    xs, ys
  end

  # Car
  @series begin
    fillcolor --> :black
    seriestype := :shape

    θ = cos(3 * env.state.position)
    xs = [-car_width/2, -car_width/2, car_width/2, car_width/2]
    ys = [0, car_height, car_height, 0]
    ys += clearance
    xs, ys = rotate(xs, ys, θ)
    translate(xs, ys, [env.state.position, height(env.state.position)])
  end

  # Flag
  @series begin
    linecolor --> :red
    seriestype := :path

    [goal_position, goal_position], [height(goal_position), height(goal_position) + flag_height]
  end
end
end
