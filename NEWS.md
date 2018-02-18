# v0.1.0

- New interface for controlling episode termination `maxsteps(env)::Int` ([#17]).
  The condition of termination is `finished(...) || maxsteps(...)` now.

- New field for `CartPole` environment: `maxsteps`.
  An keyword of constructor is added: `CartPole(; maxsteps = 42)` ([#17]).

  Also, there are helper functions of CartPole v0 and v1:
    - `CartPoleV0()`: this is equal to `CartPole(maxsteps = 200)`
    - `CartPoleV1()`: this is equal to `CartPole(maxsteps = 500)`

[#17]: https://github.com/JuliaML/Reinforce.jl/pull/17
