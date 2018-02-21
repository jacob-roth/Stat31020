using JuMP
using Stat31020
using Base.Test

x0 = ones(2); x1 = x0[1]; x2=x0[2]

# testing:
## -----------------------------------------------------------------------------
## min:  f = sum(x[1]^4 - 2.0*x[1]^2*x[2] + x[1]^2 + x[2]^2 - 2*x[1] for i=I)
## s.t.: -(x[1] + 0.25)^2 + 0.75*x[2] >= 0)
##       -(x[1] + 0.25) + 0.75*x[2]^2 >= 0)
## -----------------------------------------------------------------------------
f, g, h, j, kkt = Stat31020.constrained_wrap(x0, 2; kkt=true, μ=1.0, λ=ones(2))

function h_analytic(x1, x2)
    h = [[(12.0*x1^2 - 4.0*x2 + 2.) (-4.0*x1)];[(-4.0*x1) (2.)]] +
        [[-2. 0.];[0. 0.]] + [[0. 0.];[0. 1.5]]
    return h
end
function j_analytic(x1, x2)
    j = [[(-2.0*(x1 + 0.25)) 0.75];[-1.0 (1.5*x2)]]
    return j
end

h_a = h_analytic(x1, x2)
j_a = j_analytic(x1, x2)
z = zeros(2, 2)
kkt_a = [h_a j_a'; j_a z]

display(full(kkt))
println();
display(full(kkt_a))
println();
@test full(kkt) == full(kkt_a)
