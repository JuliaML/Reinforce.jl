using Reinforce
using Test

@testset "Reinforce" begin
  include("foo.jl")
  include("interface.jl")

  @testset "env" begin
    include("env/mountain_car.jl")
    include("env/multi-armed-bandit.jl")
  end
end
