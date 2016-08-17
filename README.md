# Reinforce (WIP)

[![Build Status](https://travis-ci.org/tbreloff/Reinforce.jl.svg?branch=master)](https://travis-ci.org/tbreloff/Reinforce.jl)
[![Gitter](https://badges.gitter.im/reinforcejl/Lobby.svg)](https://gitter.im/reinforcejl/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

Reinforce.jl is an interface for Reinforcement Learning.  It is intended to connect modular environments, policies, and solvers with a simple interface.

![](https://cloud.githubusercontent.com/assets/933338/17670982/8923a2f6-62e2-11e6-943f-bd0a2a7b5c1f.gif)
![](https://cloud.githubusercontent.com/assets/933338/17703784/f3e18414-63a0-11e6-9f9e-f531278216f9.gif)

---

Packages which build on Reinforce:

- [AtariAlgos](https://github.com/tbreloff/AtariAlgos.jl): Environment which wraps Atari games using [ArcadeLearningEnvironment](https://github.com/nowozin/ArcadeLearningEnvironment.jl)

---

New environments are created by subtyping `AbstractEnvironment` and implementing a few methods:

- `reset!(env)`
- `step!(env, s, a) --> r, s′`
- `done(env)`
- `actions(env, s) --> A`

and optional overrides:

- `state(env) --> s`
- `reward(env) --> r`

which map to `env.state` and `env.reward` respectively when unset.

TODO: more details and examples

---

Agents/policies are created by subtyping `AbstractPolicy` and implementing `action`.  The built-in random policy is a short example:

```julia
type RandomPolicy <: AbstractPolicy end
action(policy::RandomPolicy, r, s′, A′) = rand(A′)
```

The `action` method maps the last reward and current state to the next chosen action: `(r, s′) --> a′`.

---

Iterate through episodes using the `Episode` iterator.  The convenience method `episode!` demonstrates this:

```julia
function episode!(env, policy = RandomPolicy(); stepfunc = on_step, kw...)
	ep = Episode(env, policy; kw...)
	for sars in ep
		stepfunc(env, ep.niter, sars)
	end
	ep.total_reward, ep.niter
end
```

A 4-tuple `(s,a,r,s′)` is returned from each step of the episode.  Whether we write `r` or `r′` is a matter of convention.
