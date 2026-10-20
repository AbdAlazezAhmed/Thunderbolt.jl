"""
    FastIterativeMethodDiscretization(; activation_protocol, subdomains = String[], order = 1)

Descriptor of the discretization of an [`EikonalModel`](@ref) on the vertices of a tetrahedral mesh, to
be solved with the Fast Iterative Method (FastIterativeMethod.jl). `subdomains` are solved one after the
other, each started by the `activation_protocol`.
"""
Base.@kwdef struct FastIterativeMethodDiscretization{ActivationProtocolT <: TransmembraneStimulationProtocol}
    order::Int = 1
    activation_protocol::ActivationProtocolT
    subdomains::Vector{String} = String[]
end

"""
    EikonalFunction

A discrete eikonal problem ``\\sqrt{\\nabla t_a^T \\mathbb{V} \\nabla t_a} = 1`` for the nodal activation
times ``t_a`` of a tetrahedral mesh: its `vertices`, `cells` and `vertex_to_cell` map, the
`activation_points` with their `activation_points_offsets` (`NaN`: no prescribed time), the
conduction velocity tensor field `diffusion_tensor_field` and the vertex sets of the `subdomains`.
"""
struct EikonalFunction{
    T <: Number,
    VerticesVectorT <: AbstractVector{Vec{3, T}},
    CellsVectorT <: AbstractVector{NTuple{4, Int}},
    V2CT <: AbstractArray,
    DTFT <: SpectralTensorCoefficient,
    SetT <: AbstractSet{Int},
} <: AbstractSemidiscreteFunction
    vertices::VerticesVectorT
    cells::CellsVectorT
    vertex_to_cell::V2CT
    activation_points::Vector{Int}
    activation_points_offsets::Vector{Float64}
    diffusion_tensor_field::DTFT
    subdomains::Vector{SetT}
end

solution_size(f::EikonalFunction) = length(f.vertices)

"""
    ReactionEikonalODEFunction(pointwise_function, activation_timings)

The reaction part of the reaction-eikonal split: a [`PointwiseODEFunction`](@ref) of
[`StimulatedCellModel`](@ref)s, one per vertex, whose coordinates are the `activation_timings` the
eikonal solve writes into.
"""
struct ReactionEikonalODEFunction{PF <: PointwiseODEFunction, TV <: AbstractVector}
    pointwise_function::PF
    activation_timings::TV
end

solution_size(f::ReactionEikonalODEFunction) = solution_size(f.pointwise_function)

"""
    ReactionEikonalFunction(ode_function, eikonal_function)

The semidiscrete reaction-eikonal split: the eikonal function, solved once for the activation times,
and the cell dynamics ([`ReactionEikonalODEFunction`](@ref)) they stimulate.
"""
struct ReactionEikonalFunction{ODEFunctionT, EikonalFunctionT}
    ode_function::ODEFunctionT
    eikonal_function::EikonalFunctionT
end

"""
    ReactionEikonalDiffusionFunction(reaction_diffusion_function, eikonal_function, activation_timings)

The semidiscrete reaction-eikonal-diffusion split: a reaction-diffusion split function whose cells get
the foot current at the `activation_timings` of the eikonal function.
"""
struct ReactionEikonalDiffusionFunction{RDFunctionT, EikonalFunctionT, ActivationTimingsT}
    reaction_diffusion_function::RDFunctionT
    eikonal_function::EikonalFunctionT
    activation_timings::ActivationTimingsT
end

# The nodes the protocol activates: the mesh's "endocardium" node set if there is one, otherwise the
# nodes at the endocardium of the coordinate system (transmural coordinate up to 0.05).
function _endocardial_nodes(protocol::UniformEndocardialActivationProtocol, mesh)
    haskey(mesh.grid.nodesets, "endocardium") && return sort!(collect(mesh.grid.nodesets["endocardium"]))
    cs = protocol.cs
    cs === nothing && error("The mesh has no node set \"endocardium\" and the protocol no coordinate system.")
    qrc = NodalQuadratureRuleCollection(LagrangeCollection{1}())
    nodes = Int[]
    for sdh in cs.dh.subdofhandlers
        qr = getquadraturerule(qrc, sdh)
        csc = setup_coefficient_cache(cs, qr, sdh)
        for cell in CellIterator(sdh)
            for qp in QuadratureIterator(qr)
                evaluate_coefficient(csc, cell, qp, 0.0).transmural ≤ 0.05 || continue
                push!(nodes, cell.nodes[qp.i])
            end
        end
    end
    return unique!(nodes)
end

function semidiscretize(
    model::EikonalModel,
    discretization::FastIterativeMethodDiscretization{<:UniformEndocardialActivationProtocol},
    mesh::SimpleMesh,
)
    grid = mesh.grid
    all(c -> c isa Tetrahedron, getcells(grid)) ||
        error("The Fast Iterative Method discretization needs a tetrahedral mesh.")
    protocol = discretization.activation_protocol
    vertices = [Vec{3, Float64}(get_node_coordinate(n)) for n in getnodes(grid)]
    # The prescribed time of every vertex: the offset of the first listed subdomain it belongs to.
    offsets = fill(NaN, length(vertices))
    for subdomain in discretization.subdomains
        haskey(protocol.subdomains_offsets, subdomain) || continue
        for cellidx in getcellset(grid, subdomain), node in getcells(grid, cellidx).nodes
            isnan(offsets[node]) && (offsets[node] = protocol.subdomains_offsets[subdomain])
        end
    end
    isempty(discretization.subdomains) && fill!(offsets, 0.0)
    activation_points = _endocardial_nodes(protocol, mesh)
    cells = [c.nodes for c in getcells(grid)]
    nnodes = getnnodes(grid)
    max_vertices, _, _ = Ferrite._max_nentities_per_cell(getcells(grid))
    vertex_to_cell = Ferrite.build_vertex_to_cell(getcells(grid); max_vertices, nnodes)
    subdomains = [
        OrderedSet{Int}(node for cellidx in getcellset(grid, subdomain) for node in getcells(grid, cellidx).nodes)
        for subdomain in discretization.subdomains
    ]
    return EikonalFunction(
        vertices,
        cells,
        vertex_to_cell,
        activation_points,
        offsets[activation_points],
        model.κ,
        subdomains,
    )
end

"""
    semidiscretize(split::ReactionEikonalSplit{<:MonodomainModel}, (fem, fim), mesh) -> ReactionEikonalFunction

The reaction-eikonal split of a monodomain model: an [`EikonalFunction`](@ref) with the conduction
tensor `κ` of the model, and one [`StimulatedCellModel`](@ref) of the model's ionic model per vertex,
stimulated at the vertex's activation time (initially `Inf`, set by the eikonal solve).
"""
function semidiscretize(
    split::ReactionEikonalSplit{<:MonodomainModel},
    discretizations::Tuple{<:FiniteElementDiscretization, <:FastIterativeMethodDiscretization},
    mesh::AbstractGrid,
)
    epmodel = split.model
    eikonal_function = semidiscretize(EikonalModel(epmodel.κ), discretizations[2], mesh)
    nvertices = solution_size(eikonal_function)
    activation_timings = fill(Inf, nvertices)
    cell_model = StimulatedCellModel(; cell_model = epmodel.ion)
    pointwise = PointwiseODEFunction(
        cell_model,
        activation_timings,
        1:(nvertices*num_states(cell_model)),
        epmodel.internal_state_symbol,
    )
    return ReactionEikonalFunction(ReactionEikonalODEFunction(pointwise, activation_timings), eikonal_function)
end
