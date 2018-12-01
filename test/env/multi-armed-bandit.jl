using Distributions
using Reinforce
using Test


struct TestEpisodePolicy <: AbstractPolicy end
action(::TestEpisodePolicy, r, s, A) = 2


@testset "MultiArmedBanditEnv" begin


@info "Reinforce.MultiArmedBandit"

@testset "constructor" begin
  let
    σ = 5
    env = MultiArmedBandit(10, 42; σ = σ)

    @test iszero(reward(env))
    @test maxsteps(env) == 42
    @test actions(env, nothing) == 1:10
    @test length(env.arms) == 10
    for arm ∈ env.arms
      @test arm.σ == σ
    end
  end

  let
    d₁ = Uniform(1, 42)
    d₂ = Gamma()
    env = MultiArmedBandit(d₁, d₂)

    @test iszero(reward(env))
    @test actions(env, nothing) == 1:2
    @test length(env.arms) == 2
    @test env.arms[1] == d₁
    @test env.arms[2] == d₂
  end
end  # @testset "constructor"


@testset "episode iterator" begin

  let
    env = MultiArmedBandit(10, 5)
    π = TestEpisodePolicy()

    @test iszero(reward(env))
    for (s, a, r, s′) ∈ Episode(env, π)
      @test a == 2
    end
    @test !iszero(reward(env))
  end
end


end  # @testset "MultiArmedBanditEnv"
