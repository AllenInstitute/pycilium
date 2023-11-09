import numpy

import cloudvolume
from cloudvolume.datasource.precomputed.mesh.unsharded import *

from pycilium.utils.skeleton_utils import *


def bb_included(bb1, bb2):
    return (bb1.contains_bbox(bb2) or
            bb2.contains_bbox(bb1) or
            bb1.intersects(bb1, bb2))


def bbox_from_filename(fn, scale=numpy.array([8, 8, 1])):
    bb = cloudvolume.Bbox.from_filename(fn)
    # is there a record for how these are chunked?
    # return bb
    return cloudvolume.Bbox.from_points([
        bb.minpt * scale,
        bb.maxpt * scale
        ])


def get_mesh_manifests_intersecting_bbox(cv, segids, bbox):
    filtered_manifests = {}
    for segid, manifests in cv.mesh._get_manifests(segids).items():
        filtered_manifests[segid] = [
            m for m in manifests if bb_included(bbox, bbox_from_filename(m))]      
    return filtered_manifests


def get_mesh_bbox(
      cv, segids, bbox, 
      remove_duplicate_vertices=True, 
      fuse=True,
      chunk_size=None):
    """
    Merge fragments derived from these segids into a single vertex and face list.
    Why merge multiple segids into one mesh? For example, if you have a set of
    segids that belong to the same neuron.
    segids: (iterable or int) segids to render into a single mesh
    Optional:
      remove_duplicate_vertices: bool, fuse exactly matching vertices
      fuse: bool, merge all downloaded meshes into a single mesh
      chunk_size: [chunk_x, chunk_y, chunk_z] if passed only merge at chunk boundaries
    
    Returns: Mesh object if fused, else { segid: Mesh, ... }
    """
    segids = toiter(segids)
    dne = cv.mesh.exists(segids)
    dne = [ label for label, path in dne.items() if path is None ]

    if dne:
      missing = ', '.join([ str(segid) for segid in dne ])
      raise ValueError(red(
        'Segment ID(s) {} are missing corresponding mesh manifests.\nAborted.' \
        .format(missing)
      ))

    fragments = get_mesh_manifests_intersecting_bbox(cv, segids, bbox)
    fragments = fragments.values()
    fragments = list(itertools.chain.from_iterable(fragments)) # flatten
    fragments = cv.mesh._get_mesh_fragments(fragments)

    # decode all the fragments
    meshdata = defaultdict(list)
    for frag in tqdm(fragments, disable=(not cv.config.progress), desc="Decoding Mesh Buffer"):
      segid = filename_to_segid(frag[0])
      try:
        mesh = Mesh.from_precomputed(frag[1])
      except Exception:
        print(frag[0], 'had a problem.')
        raise
      meshdata[segid].append(mesh)

    if not fuse:
      return { segid: Mesh.concatenate(*meshes) for segid, meshes in six.iteritems(meshdata) }

    meshdata = [ (segid, mesh) for segid, mesh in six.iteritems(meshdata) ]
    meshdata = sorted(meshdata, key=lambda sm: sm[0])
    meshdata = [ mesh for segid, mesh in meshdata ]
    meshdata = list(itertools.chain.from_iterable(meshdata)) # flatten
    mesh = Mesh.concatenate(*meshdata)

    if not remove_duplicate_vertices:
      return mesh 

    if not chunk_size:
      return mesh.consolidate()

    if cv.meta.mip is not None:
      mip = cv.meta.mip
    else:
      # This will usually be wrong, but it's backwards compatible.
      # Throwing an exception instead would probably break too many
      # things.
      mip = cv.config.mip

    if mip not in cv.meta.meta.available_mips:
      raise exceptions.ScaleUnavailableError("mip {} is not available.".format(mip))

    resolution = cv.meta.meta.resolution(mip)
    chunk_offset = cv.meta.meta.voxel_offset(mip)

    return mesh.deduplicate_chunk_boundaries(
      chunk_size * resolution, is_draco=False,
      offset=(chunk_offset * resolution)
    )


# def get_mesh_manifests_intersecting_bbox(cv, segids, bbox):
#     segids = toiter(segids)    
#     paths = [cv.mesh.manifest_path(segid) for segid in segids]
#     paths = [p for p in paths if bb_included(bbox, bbox_from_filename(p))]
#     fragments = cv.cache.download(paths)
# 
#     contents = {}
#     for filename, content in fragments.items():
#       segid = filename_to_segid(filename)
#       if content is None:
#         if allow_missing:
#           contents[segid] = None
#           continue
#         else:
#           raise ValueError(f"manifest is missing for {filename}")
# 
#       content = content.decode('utf8')
#       content = simdjson.loads(content)
#       contents[segid] = content['fragments']
# 
#     return contents
# 
# 
# def get_mesh_bbox(
#         cv, segids, bbox,
#         remove_duplicate_vertices=True, 
#         use=True,
#         hunk_size=None):
#     """
#     Merge fragments derived from these segids into a single vertex and face list.
#     Why merge multiple segids into one mesh? For example, if you have a set of
#     segids that belong to the same neuron.
#     segids: (iterable or int) segids to render into a single mesh
#     Optional:
#       remove_duplicate_vertices: bool, fuse exactly matching vertices
#       fuse: bool, merge all downloaded meshes into a single mesh
#       chunk_size: [chunk_x, chunk_y, chunk_z] if passed only merge at chunk boundaries
#     
#     Returns: Mesh object if fused, else { segid: Mesh, ... }
#     """
#     segids = toiter(segids)
#     dne = cv.mesh.exists(segids)
#     dne = [ label for label, path in dne.items() if path is None ]
# 
#     if dne:
#       missing = ', '.join([ str(segid) for segid in dne ])
#       raise ValueError(red(
#         'Segment ID(s) {} are missing corresponding mesh manifests.\nAborted.' \
#         .format(missing)
#       ))
# 
#     fragments = get_mesh_manifests_intersecting_bbox(cv, segids, bbox)
#     path_id_map = {}
#     for segid, paths in fragments.items():
#       for path in paths:
#         path_id_map[path] = segid
#     fragments = cv._get_mesh_fragments(path_id_map)
# 
#     # decode all the fragments
#     meshdata = defaultdict(list)
#     for filename, contents, segid in tqdm(fragments, disable=(not cv.config.progress), desc="Decoding Mesh Buffer"):
#       try:
#         mesh = Mesh.from_precomputed(contents)
#       except Exception:
#         print(filename, 'had a problem.')
#         raise
#       meshdata[segid].append(mesh)
# 
#     if not fuse:
#       meshdata = { 
#           segid: Mesh.concatenate(*meshes, segid=segid) 
#           for segid, meshes in meshdata.items() 
#       }
#       for mesh in meshdata.values():
#         mesh.vertices = apply_transform(mesh.vertices, cv.transform)
#       return meshdata
# 
#     meshdata = [ (segid, mesh) for segid, mesh in meshdata.items() ]
#     meshdata = sorted(meshdata, key=lambda sm: sm[0])
#     meshdata = [ mesh for segid, mesh in meshdata ]
#     meshdata = list(itertools.chain.from_iterable(meshdata)) # flatten
#     mesh = Mesh.concatenate(*meshdata)
#     mesh.vertices = apply_transform(mesh.vertices, cv.transform)
# 
#     if not remove_duplicate_vertices:
#       return mesh 
# 
#     if not chunk_size:
#       return mesh.consolidate()
# 
#     if cv.meta.mip is not None:
#       mip = cv.meta.mip
#     else:
#       # This will usually be wrong, but it's backwards compatible.
#       # Throwing an exception instead would probably break too many
#       # things.
#       mip = cv.config.mip
# 
#     if mip not in cv.meta.meta.available_mips:
#       raise exceptions.ScaleUnavailableError("mip {} is not available.".format(mip))
# 
#     resolution = cv.meta.meta.resolution(mip)
#     chunk_offset = cv.meta.meta.voxel_offset(mip)
# 
#     return mesh.deduplicate_chunk_boundaries(
#       chunk_size * resolution, is_draco=False,
#       offset=(chunk_offset * resolution)
#     )


def yield_cilobj_segids(cv, cilobj,
                        points_to_consider=[
                            "base_pix",
                            "exit_pix"
                            "soma_valence_pt_pix",
                            "tip_pix"],
                        pt_scale=None):
    previous_seg_ids = []
    for attr in points_to_consider:
        pt = getattr(cilobj, attr, None)
        if pt is None:
            continue
        try:
            if pt_scale is None:
                scaled_pt = cv.point_to_mip(pt, 0, cv.mip)
            else:
                scaled_pt = numpy.array(pt_scale) * numpy.array(pt)
            seg_id = cv[scaled_pt.tolist()].flatten()[0]
        except Exception as e:  # TODO better cv obj
            print(e)
            continue
        if seg_id:
            if seg_id not in previous_seg_ids:
                yield seg_id
                previous_seg_ids.append(seg_id)


def yield_skelcd_segids(cv, skelcd, pt_scale=None, sample_rate=1):
    previous_seg_ids = []
    for swc_vtx in skelcd[0][::sample_rate]:
        pt = numpy.array(swc_vtx[3:6])
        if pt is None:
            continue
        try:
            if pt_scale is None:
                scaled_pt = cv.point_to_mip(pt, 0, cv.mip)
            else:
                scaled_pt = numpy.array(pt_scale) * numpy.array(pt)
            seg_id = cv[scaled_pt.tolist()].flatten()[0]
        except Exception as e:  # TODO better cv obj
            print(e)
            continue
        if seg_id:
            if seg_id not in previous_seg_ids:
                yield seg_id
                previous_seg_ids.append(seg_id)


def get_skelcd_segids(*args, **kwargs):
    return [*yield_skelcd_segids(*args, **kwargs)]


def get_cilobj_segids(*args, **kwargs):
    return [i for i in yield_cilobj_segids(*args, **kwargs)]


def get_cilobj_segid(*args, **kwargs):
    return next(yield_cilobj_segids(*args, **kwargs))


def get_buffered_bbox(bbox, buffer=(10, 10, 1)):
    buffer = numpy.array(buffer)
    return cloudvolume.Bbox.from_points(
        [
            bbox.minpt - buffer,
            bbox.maxpt + buffer
        ])


def cilium_bbox_pix(cilobj, buffer=(0, 0, 0)):
    cilbbox = cloudvolume.Bbox.from_points(
        [cilobj.base_pix,
         cilobj.tip_pix])
    return get_buffered_bbox(cilbbox, buffer=buffer)


def get_segid_to_bboxmesh_map_from_cilobj_buffer(
        mesh_cv, segids, cilobj, bbox_buffer=[100, 100, 10]):
    segid_to_bboxmesh = {}
    cbbox = cilium_bbox_pix(cilobj, bbox_buffer)
    for segid in segids:
        bboxmesh = get_mesh_bbox(
            mesh_cv, [segid], cbbox, remove_duplicate_vertices=False)
        segid_to_bboxmesh[segid] = bboxmesh
    return segid_to_bboxmesh


def get_segid_to_bboxmesh_map_from_bbox(
        mesh_cv, segids, bbox):
    segid_to_bboxmesh = {}
    for segid in segids:
        bboxmesh = get_mesh_bbox(
            mesh_cv, [segid], bbox, remove_duplicate_vertices=False)
        segid_to_bboxmesh[segid] = bboxmesh
    return segid_to_bboxmesh


def get_skelcd_bbox(cdskel):
    vtxs, _, _ = vtxs_edges_rootid_read_compact_detail(cdskel)
    vtxmin = numpy.min(vtxs, axis=0)
    vtxmax = numpy.max(vtxs, axis=0)
    return cloudvolume.Bbox.from_points([
        vtxmin,
        vtxmax
    ])