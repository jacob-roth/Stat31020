module Stat31020
using JuMP
using MathProgBase
"""
...
# Arguments
- `x::Array{Float64,1}`: current iterate
- `n::Int64`: number of derivatives to query
- `kkt::Bool`: T/F to return KKT matrix
- `μ::Float64`: objective function multiplier (default = 1.0)
- `λ::Array{Float64,1}`: constraint multiplier (default = [1.0, ..., 1.0])
# Output
- `f::SparseMatrixCSC`: function evaulation
- `g::SparseMatrixCSC`: gradient evaulation
- `H::SparseMatrixCSC`: sparse, symmetric hessian evaulation
- `J::SparseMatrixCSC`: sparse constraint jacobian evaluation
- `KKT::SparseMatrixCSC`: sparse KKT system evaulation
...
"""
function constrained_wrap(x::Array{Float64,1}, nderivs::Int64;
                          kkt=true, μ=nothing, λ=nothing)
    ## setup
    n = length(x)                          ## problem dimension
    m = 0                                  ## number of constraints (TBD below)
    I = 1:n-1                              ## index set

    ## -------------------------------------------------------------------------
    ## TODO: fill in function evaluation
    ## -------------------------------------------------------------------------
    f = sum(x[i]^4 - 2.0*x[i]^2*x[i+1] + x[i]^2 + x[i+1]^2 - 2*x[i] for i=I)
    ## -------------------------------------------------------------------------

    if nderivs == 0
        return f

    elseif nderivs >= 1
        ## initialize JuMP model and MathProgBase interface
        NLmodel = Model()                   ## JuMP object
        @variable(NLmodel, y[1:n])
        ## ---------------------------------------------------------------------
        ## TODO: fill in objective function and constraints
        ## ---------------------------------------------------------------------
        @NLobjective(NLmodel, Min, sum(y[i]^4 -
                                       2.0*y[i]^2*y[i+1] +
                                       y[i]^2 +
                                       y[i+1]^2 -
                                       2*y[i] for i=I))
        @NLconstraint(NLmodel, -(y[1] + 0.25)^2 + 0.75*y[2] >= 0)
        @NLconstraint(NLmodel, -(y[1] + 0.25) + 0.75*y[2]^2 >= 0)
        ## ---------------------------------------------------------------------
        d = JuMP.NLPEvaluator(NLmodel)      ## MPB object
        m = MathProgBase.numconstr(NLmodel) ## number of contraints
        ## initialize dual variable values (if not specified)
        if μ == nothing
            μ = 1.
        end
        if λ == nothing
            λ = ones(m)
        end

        ## evaluate derivatives
        if nderivs == 1
            MathProgBase.initialize(d, [:Grad])
            ## evaluate gradient
            g = zeros(n)
            MathProgBase.eval_grad_f(d, g, x)
            return (f, g)

        elseif nderivs == 2
            MathProgBase.initialize(d, [:Grad, :Hess, :Jac])
            ## evaluate Hessian
            ij_H = MathProgBase.hesslag_structure(d)
            h = ones(length(ij_H[1]))
            MathProgBase.eval_hesslag(d, h, x, μ, λ)
            H = populate_hess_sparse(ij_H[1], ij_H[2], h, n)
            ## evalute gradient (partially pre-computed from above)
            g = zeros(n)
            MathProgBase.eval_grad_f(d, g, x)
            ## evaluate Jacobian (constraints)
            if kkt == true
                ij_J = MathProgBase.jac_structure(d)
                j = ones(length(ij_J[1]))
                MathProgBase.eval_jac_g(d, j, x)
                J = sparse(ij_J[1], ij_J[2], j)
                KKT = [H J'; J zeros(m, m)]
                println(KKT)
                return (f, g, H, J, KKT)
            else
               return (f, g, H)
            end
        end
    end
end


"""
...
# Arguments
- `i::Array{Int64,1}`: row indices
- `j::Array{Int64,1}`: col indices
- `h::Array{Int64,1}`: values s.t. h[1] = H[i[1], j[1]]
- `n::Int64`: dimension of hessian
# Output
- `H::SparseMatrixCSC`: sparse, symmetric hessian
...
"""
function populate_hess_sparse(
    i::Array{Int64,1},
    j::Array{Int64,1},
    h::Array{Float64,1},
    n::Int64)
    ## build matrix; probably a better way to do this...
    H = sparse([i;j], [j;i], [h;h], n, n, +)  ## NOTE: symmetrizing doubles the diagonal
    H = H - Diagonal(H)/2.                    ## NOTE: required removal of 1/2 digonal
    return H
end
end # module
