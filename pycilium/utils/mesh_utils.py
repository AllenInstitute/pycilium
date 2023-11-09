import numpy

import meshparty
import meshparty.mesh_filters
import trimesh

from pycilium.utils.skeleton_utils import *


def filter_mesh_cdskel_line_distance(mesh, cdskel, radius=600, endcap_buffer=300):
    segs = skelcd_to_linesegments(cdskel)

    orig_mesh = meshparty.trimesh_io.Mesh(
        mesh.vertices, mesh.faces, process=False)
    line_mf = numpy.zeros(orig_mesh.vertices.shape[0], dtype=bool)
    for seg in segs:
        line_mf |= meshparty.mesh_filters.filter_close_to_line(
            orig_mesh, numpy.array(seg), radius,
            endcap_buffer=endcap_buffer)
    line_mesh = meshparty.trimesh_io.Mesh(
        vertices=orig_mesh.vertices, node_mask=line_mf,
        faces=orig_mesh.faces, apply_mask=True, process=False)
    return line_mesh


def gen_filter_meshes_cdskel_line_distance(meshes, *args, **kwargs):
    for mesh in meshes:
        try:
            yield filter_mesh_cdskel_line_distance(mesh, *args, **kwargs)
        # FIXME IndexError when filtering does not yield vertices
        except IndexError:
            continue


def filter_meshes_cdskel_line_distance(*args, **kwargs):
    return [*gen_filter_meshes_cdskel_line_distance(*args, **kwargs)]


def mpskel_with_radius_from_skelcd_meshes(skelcd, meshes):
    vtxs, edges, root_id = vtxs_edges_rootid_read_compact_detail(skelcd)
    
    cp_closest, cp_distance, cp_triangle = None, None, None
    for i, mesh in enumerate(meshes):
        if cp_closest is None:
            cp_closest, cp_distance, cp_triangle = trimesh.proximity.closest_point(mesh, vtxs)
            mesh_idxs = numpy.array([i] * len(vtxs))
            continue
        new_cp_closest, new_cp_distance, new_cp_triangle = trimesh.proximity.closest_point(mesh, vtxs)

        dist_mask = new_cp_distance < cp_distance

        mesh_idxs[dist_mask] = numpy.array([i] * dist_mask.sum())
        cp_distance[dist_mask] = new_cp_distance[dist_mask]
        cp_closest[dist_mask] = new_cp_closest[dist_mask]
        cp_triangle[dist_mask] = new_cp_triangle[dist_mask]
  
    rtdist = numpy.empty_like(cp_distance)
    for i, mesh in enumerate(meshes):
        mesh_mask = (mesh_idxs == i)
   
        # get first vertex of each face from closest triangles
        mesh_vtxs_idxs = mesh.faces[cp_triangle[mesh_mask]][:, 0]
    
        rtdist[mesh_mask] = meshparty.ray_tracing.ray_trace_distance(
            mesh_vtxs_idxs,
            mesh
        )
    
    mpskel = meshparty.skeleton.Skeleton(vtxs, edges, root_id, rtdist)
    return mpskel


def mpskel_with_radius_from_skelcd_meshes_or_None(*args, **kwargs):
    try:
        return mpskel_with_radius_from_skelcd_meshes(*args, **kwargs)
    except IndexError:
        return