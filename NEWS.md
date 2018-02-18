# v0.1.0

- Drop Julia 0.5 support. (#15)

- New interface for controlling episode termination `maxsteps(env)::Int` ([#17]).
  The condition of termination is `finished(...) || maxsteps(...)` now.

- New field for `CartPole` environment: `maxsteps`.
  An keyword of constructor is added: `CartPole(; maxsteps = 42)` ([#16], [#17]).

  Also, there are helper functions of CartPole v0 and v1:
    - `CartPoleV0()`: this is equal to `CartPole(maxsteps = 200)`
    - `CartPoleV1()`: this is equal to `CartPole(maxsteps = 500)`

- Keyword `maxsteps` of `run_episode` is deprecated,
  please overload  `maxsteps`. ([#19])

[#15]: https://github.com/JuliaML/Reinforce.jl/pull/15
[#16]: https://github.com/JuliaML/Reinforce.jl/pull/16
[#17]: https://github.com/JuliaML/Reinforce.jl/pull/17
[#19]: https://github.com/JuliaML/Reinforce.jl/pull/19
