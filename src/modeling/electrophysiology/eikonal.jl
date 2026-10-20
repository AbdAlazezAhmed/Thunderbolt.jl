@doc raw"""
    EikonalModel(κ)

Model formulated as ``\sqrt{{\nabla t_a}^T \kappa(x) \nabla t_a} = 1`` for the activation time ``t_a``,
with ``\kappa`` the conduction velocity tensor squared (a `SpectralTensorCoefficient`).
"""
struct EikonalModel{DiffusivityCoefficientType}
    κ::DiffusivityCoefficientType
end

"""
    UniformEndocardialActivationProtocol(subdomains_offsets, cs)

Activation protocol for the eikonal model: the endocardium -- the node set `"endocardium"` of the mesh,
or the nodes with a transmural coordinate of `cs` up to 0.05 -- is activated at once, at the time
`subdomains_offsets[subdomain]` in every listed subdomain. Stands in for the conduction system.
"""
struct UniformEndocardialActivationProtocol{CST} <: TransmembraneStimulationProtocol
    subdomains_offsets::Dict{String, Float64}
    cs::CST
end

"""
    ReactionEikonalSplit(model, cs)

Annotation for the reaction-eikonal split of a given model (Neic et al. 2017): the eikonal equation
gives the activation times, at which the cell models are stimulated. The argument `cs` is a coefficient
describing the input `x` for the reaction model rhs, which is usually some generalized coordinate.
"""
struct ReactionEikonalSplit{mType, csType}
    model::mType
    cs::csType
end

"""
    ReactionEikonalDiffusionSplit(model, cs)

Annotation for the reaction-eikonal-diffusion split of a given model: the eikonal activation times only
add a foot current to a reaction-diffusion model.
"""
struct ReactionEikonalDiffusionSplit{mType, csType}
    model::mType
    cs::csType
end

@doc raw"""
    StimulatedCellModel(; cell_model, stim_length = 1.0, stim_strength = 0.91, τᶠ = 0.25)

Wrapper around an ionic cell model that adds the foot current of [NeicCamposPrassl:2017:ECE](@citet)
at the activation time ``t_a`` of the point, which the point brings along as its coordinate `x`:
```math
\partial_t \varphi_m \mathrel{+}= \frac{s}{\tau_f} e^{(t - t_a)/\tau_f}, \qquad t_a \le t \le t_a + \ell .
```
A point with ``t_a = \infty`` is never stimulated.
"""
Base.@kwdef struct StimulatedCellModel{TC <: AbstractIonicModel, T} <: AbstractIonicModel
    cell_model::TC
    stim_length::T = 1.0
    stim_strength::T = 0.91
    τᶠ::T = 0.25
end

num_states(m::StimulatedCellModel) = num_states(m.cell_model)
default_initial_state(m::StimulatedCellModel) = default_initial_state(m.cell_model)
transmembranepotential_index(m::StimulatedCellModel) = transmembranepotential_index(m.cell_model)
state_symbol(m::StimulatedCellModel, i::Int) = state_symbol(m.cell_model, i)

@inline function cell_rhs!(du, u, activation_time::Real, t, m::StimulatedCellModel)
    cell_rhs!(du, u, nothing, t, m.cell_model)
    if activation_time ≤ t ≤ activation_time + m.stim_length
        idx = transmembranepotential_index(m.cell_model)
        du[idx] += m.stim_strength / m.τᶠ * exp((t - activation_time) / m.τᶠ)
    end
    return nothing
end
