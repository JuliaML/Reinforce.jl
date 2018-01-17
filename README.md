# Reinforce

[![Build Status](https://travis-ci.org/JuliaML/Reinforce.jl.svg?branch=master)](https://travis-ci.org/JuliaML/Reinforce.jl)
[![Gitter](https://badges.gitter.im/reinforcejl/Lobby.svg)](https://gitter.im/reinforcejl/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

Reinforce.jl is an interface for Reinforcement Learning.  It is intended to connect modular environments, policies, and solvers with a simple interface.

![](https://cloud.githubusercontent.com/assets/933338/17670982/8923a2f6-62e2-11e6-943f-bd0a2a7b5c1f.gif)
![](https://cloud.githubusercontent.com/assets/933338/17703784/f3e18414-63a0-11e6-9f9e-f531278216f9.gif)

---

Packages which build on Reinforce:

- [AtariAlgos](https://github.com/JuliaML/AtariAlgos.jl): Environment which wraps Atari games using [ArcadeLearningEnvironment](https://github.com/nowozin/ArcadeLearningEnvironment.jl)
- [OpenAIGym](https://github.com/JuliaML/OpenAIGym.jl): Wrapper for OpenAI's python package: gym

---

New environments are created by subtyping `AbstractEnvironment` and implementing a few methods:

- `reset!(env)`
- `actions(env, s) --> A`
- `step!(env, s, a) --> r, s′`
- `finished(env, s′)`

and optional overrides:

- `state(env) --> s`
- `reward(env) --> r`

which map to `env.state` and `env.reward` respectively when unset.

- `ismdp(env) --> bool`

An environment may be fully observable (MDP) or partially observable (POMDP).  In the case of a partially observable environment, the state `s` is really an observation `o`.  To maintain consistency, we call everything a state, and assume that an environment is free to maintain additional (unobserved) internal state.  The `ismdp` query returns true when the environment is MDP, and false otherwise.

---

TODO: more details and examples

---

Agents/policies are created by subtyping `AbstractPolicy` and implementing `action`.  The built-in random policy is a short example:

```julia
mutable struct RandomPolicy <: AbstractPolicy end
action(policy::RandomPolicy, r, s′, A′) = rand(A′)
```

The `action` method maps the last reward and current state to the next chosen action: `(r, s′) --> a′`.

---

Iterate through episodes using the `Episode` iterator.  A 4-tuple `(s,a,r,s′)` is returned from each step of the episode:

```julia
ep = Episode(env, policy)
for (s, a, r, s′) in ep
    # do some custom processing of the sars-tuple
end
R = ep.total_reward
T = ep.niter
```

There is also a convenience method `run_episode`.  The following is an equivalent method to the last example:

```julia
R = run_episode(env, policy) do
    # anything you want... this section is called after each step
end
```

---

## Author: [Tom Breloff](https://github.com/tbreloff)
