using Reinforce.MountainCarEnv

@testset "MountainCarEnv" begin


@info "Reinforce.MountainCarEnv"

env = MountainCar()

i = 0
for sars in Episode(env, RandomPolicy())
    i += 1
end

@test i > 0


end  # @testset "MountainCarEnv"
