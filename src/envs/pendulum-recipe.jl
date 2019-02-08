@recipe function f(env::Pendulum)
  legend := false
  xlims := (-1,1)
  ylims := (-1,1)
  grid := false
  ticks := nothing

	# pole
  @series begin
    w = 0.2
    x = [-w,w,w,-w]
    y = [-.1,-.1,1,1]
    θ = env.θs[1]
    fillcolor := :red
    seriestype := :shape
    x*cos(θ) - y*sin(θ), y*cos(θ) + x*sin(θ)
  end

  # center
  @series begin
    seriestype := :scatter
    markersize := 10
    markercolor := :black
    annotations := [(0, -0.2, Plots.text("a: $(env.a)", :top))]
    [0],[0]
  end
end
