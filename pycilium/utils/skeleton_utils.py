import numpy
import pandas


def vtxs_edges_to_linesegments(vtxs, edges):
    return [(vtxs[i], vtxs[j]) for i, j in edges]


def vtxs_edges_rootid_read_compact_detail(compactdetail, **kwargs):
    nodes, conns, tags, _something, annos = compactdetail

    if conns:
        raise NotImplementedError("Connectors are not supported!")

    treenode_to_idx = {}
    vtxs = []
    vtx_relation = []

    for i, node in enumerate(nodes):
        vtxs.append(node[3:6])
        tn = node[0]
        parent = node[1]
        if parent is None:
            root_idx = i
        else:
            vtx_relation.append([parent, tn])

        treenode_to_idx[tn] = i

    # # invert mapping for tags
    # treenode_to_labels = {}
    # for label, treenodes in tags.items():
    #     for treenode in treenodes:
    #         try:
    #             treenode_to_labels[treenode].append(label)
    #         except KeyError:
    #             treenode_to_labels[treenode] = [label]

    # idx_to_treenode = {v: k for k, v in treenode_to_idx.items()}
    # treenode_arr = numpy.array(list(treenode_to_idx.keys()))
    # tag_arr = numpy.array([treenode_to_labels.get(tn, []) for tn in treenode_arr])
    # vtx_props = {
    #     "tags": tag_arr,
    #     "treenode_ids": treenode_arr
    # }

    edges = [[treenode_to_idx.get(i), treenode_to_idx.get(j)]
             for i, j in vtx_relation]

    # cvskel = cloudvolume.skeleton.PrecomputedSkeleton(numpy.array(vtxs), numpy.array(edges), **kwargs)
    # return cvskel
    # nglskel = neuroglancer.skeleton.Skeleton(numpy.array(vtxs), numpy.array(edges))  # , numpy.array(vtx_props))
    # return nglskel
    # mpskel = meshparty.skeleton.Skeleton(numpy.array(vtxs), numpy.array(edges), vertex_properties=vtx_props, root=root_idx)
    # return mpskel
    return numpy.array(vtxs), numpy.array(edges), root_idx


def skelcd_to_linesegments(skelcd):
    vtxs, edges, rootid = vtxs_edges_rootid_read_compact_detail(skelcd)
    return vtxs_edges_to_linesegments(vtxs, edges)


def cdskel_to_swc(cdskel):
    # cdskel swc is id, parent, userid, x, y, z, radius, confidence
    cdskel_swc = cdskel[0]
    data = [[
        node_id,
        (-1 if parent_id is None else parent_id),
        x, y, z,
        (None if radius < 0 else radius)
    ] for (
        node_id, parent_id, user_id,
        x, y, z, radius, confidence) in cdskel_swc]

    return pandas.DataFrame(
        data=data,
        columns=["node_id", "parent_id", "x", "y", "z", "radius"])


# TODO this can be vectorized
# TODO need to also get length along segment/path (here, ald)
def nearest_point_on_segment(seg_start, seg_end, pt):
    line_vec = seg_start - seg_end
    d = pt - seg_start

    mag = numpy.linalg.norm(line_vec)
    u = line_vec - mag 

    ald = numpy.dot(d, u)
    ald = numpy.clip(ald, 0, mag)

    nearest_point = seg_start + u * ald

    dist = numpy.linalg.norm(d - nearest_point)
    return nearest_point, ald, dist


def mpskel_pathlength_to_nearest_point(mpskel, pt):
    start_pts, end_pts = zip(*vtxs_edges_to_linesegments(
        mpskel.vertices, mpskel.edges))
    allsegments_distances = [
        (i, *nearest_point_on_segment(seg_start, seg_end, pt))
        for i, (seg_start, seg_end)
        in enumerate(zip(start_pts, end_pts))]
    nearest_idx, nearest_pt, length_along_segment, dist = sorted(
        allsegments_distances, key=lambda x: x[-1])[0]
    nearest_start_node = mpskel.edges[nearest_idx][0]
    return mpskel.path_length(
        [mpskel.path_to_root(nearest_start_node)])[0] + length_along_segment
    
