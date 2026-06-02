@doc raw"""
    LinearIntegrator

Represents the integrand a the linear form over some function space.
"""
struct LinearIntegrator{IntegrandType, QRC <: Union{<:QuadratureRuleCollection, Nothing}} <:
       AbstractLinearIntegrator
    integrand::IntegrandType
    qrc::QRC
end

function setup_element_cache(i::LinearIntegrator, sdh::SubDofHandler)
    return setup_element_cache(i.integrand, getquadraturerule(i.qrc, sdh), sdh)
end

# A trivial zero source integrand for convenience
struct ZeroSource <: AbstractSourceTerm end
struct ZeroSourceElementCache <: AbstractVolumetricElementCache end

function duplicate_for_device(device, cache::ZeroSourceElementCache)
    return cache
end

function setup_element_cache(::ZeroSource, qr::QuadratureRule, sdh::SubDofHandler)
    return ZeroSourceElementCache()
end

function assemble_element!(bₑ::AbstractVector, cell, ::ZeroSourceElementCache, time)
    return nothing
end
