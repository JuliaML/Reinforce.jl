import Reinforce: action, actions, finished, ismdp, maxsteps, reset!, reward, state, step!

function test_ep_iteration()
  @info "interface::iteration::maxsteps"
  env = FooEnv()
  π = FooPolicy()
  ep = Episode(env, π)

  # given the default of `finished` is `false`, iteration should hit `maxsteps`
  for (i, (s, a, r, s′)) ∈ enumerate(ep)
    @test s == i
    @test r == -1
    @test a ∈ [1, 2, 3]
    @test s′ == i + 1
  end

  @test ep.niter  == 3
  @test ep.total_reward == -3
end  # function test_ep_iteration

function test_ep_finished()
  @info "interface::iteration::finished"
  env = FooEnv()
  π = FooPolicy()
  ep = Episode(env, π)

  for (s, a, r, s′) ∈ ep
    nothing
  end

  # @eval finished(::FooEnv, s′) = false

  @test ep.niter == 1
  @test ep.total_reward == -1
end  # function test_ep_iteration

function test_run_episode()
  @info "interface::run_episode"

  env = FooEnv()
  π = FooPolicy()

  run_episode(env, π) do sars
    s, a, r, s′ = sars

    @test a ∈ [1, 2, 3]
    @test r == -1
    @test s + 1 == s′
  end
end  # function test_run_episode

@testset "interface" begin
  @info "interface::iteration"
  test_ep_iteration()
  begin
    @eval finished(::FooEnv, s′) = (s′ == 2)
    test_ep_finished()
    @eval finished(::FooEnv, s′) = false  # reset to default
  end

  test_run_episode()
end  # @testset "env interface"
