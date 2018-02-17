using Reinforce
using Base.Test

@testset "Reinforce" begin
  include("interface.jl")

  @testset "env" begin
    include("env/mountain_car.jl")
  end

end
