import copy
import requests
import urllib.parse

import numpy

import pycilium.cilia

expected_annotations_lower_to_limit = {
    "base": 1,
    "tip": 1,
    "exit": 1,
    "d_cent": 1,
    "vesicles": None
}


def state_from_ngllink(link, request_params={}):
    if link is None:
        return None
    parsed_uri = urllib.parse.urlparse(link)
    qparams = urllib.parse.parse_qs(parsed_uri.query)
    jsonservice_url = qparams.get("json_url")

    if jsonservice_url:
        if len(jsonservice_url) > 1:
            raise Exception("too many state values! {}".format(
                jsonservice_url))
        jsonservice_url = jsonservice_url[-1]
    else:
        return

    # TODO implement retries
    r = requests.get(jsonservice_url, **request_params)
    r.raise_for_status()

    state_json = r.json()
    return state_json


def cilanno_from_ngllink(ngllink, anno_base_d=None, request_params={}):
    anno_base_d = {} if anno_base_d is None else copy.deepcopy(anno_base_d)

    anno_dict = {"neuroglancer_link": ngllink}

    state_json = state_from_ngllink(ngllink, request_params=request_params)

    if state_json is not None:
        for p_type, p_limit in expected_annotations_lower_to_limit.items():
            p_lyr = [lyr for lyr in state_json['layers']
                     if lyr['name'].lower() == p_type]
            ptannos = []
            for lyr in p_lyr:
                ptanno = list(filter(None, map(
                    lambda x: x.get("point"), lyr['annotations']))) or []
                ptannos.extend(ptanno)
            if p_limit is not None:
                if len(ptannos) > p_limit:
                    msg = "{} has {} point annotations".format(
                        p_type, len(ptannos))
                    anno_dict["errors"] = (
                        (anno_dict["errors"] + [msg]) if anno_dict.get("errors")
                        else [msg])
            if len(ptannos):
                if p_limit == 1:
                    anno_dict[p_type] = ptannos[0]
                else:
                    anno_dict[p_type] = ptannos[:p_limit]

    return dict(anno_base_d, **anno_dict)


def cilobj_from_ngllink(*args, **kwargs):
    raise NotImplementedError


# janky point equivalence because not all can be trusted to match
def pt_equiv(pt1, pt2):
    if pt1 == pt2:
        return True
    elif numpy.allclose(pt1, pt2):
        return True
    elif list(map(int, pt1)) == list(map(int, pt2)):
        return True
    else:
        return False


def get_nuc_d_for_point(pt, nuc_ds):
    # FIXME one-off correction for bad annotation
    if list(map(int, pt)) == [107004, 73076, 357]:
        pt = [107001, 73075, 358]

    for nuc_d in nuc_ds:
        if pt_equiv(pt, nuc_d["point"]):
            return nuc_d


def add_lineannotation_nuc_ds(linelyr, nuc_ds, annotation):
    for anno_d in linelyr["annotations"]:
        if anno_d["type"] != "line":
            print("{} annotation found!".format(anno_d["type"]))

        # I think pointB is nucleus point, but maybe not.
        anno_pt = "pointA"
        nuc_d = get_nuc_d_for_point(anno_d["pointB"], nuc_ds)
        if nuc_d is None:
            anno_pt = "pointB"
            nuc_d = get_nuc_d_for_point(anno_d["pointA"], nuc_ds)
        if nuc_d is None:
            # print("missing")
            continue

        if annotation in nuc_d.keys():
            print("{} already exists!".format(annotation))

        nuc_d[annotation] = anno_d[anno_pt]


def cilobjs_from_bulk_ngllink(
        ngllink, nucleus_points_layer=None, base_lines_layer=None,
        d_cent_lines_layer=None, tip_lines_layer=None, exit_lines_layer=None,
        pcs_lines_layer=None, request_params={}, cilobj_kwargs={},
        vct_from_annos_func=None, ect_from_annos_func=None, postprocess_func=None):

    if postprocess_func is None:
        postprocess_func = lambda x: x

    vct_from_cilobj = cilobj_kwargs.get("valence_cell_type")
    ect_from_cilobj = cilobj_kwargs.get("extended_cell_type")

    vct_from_annos_func = (
        (lambda x: vct_from_cilobj)
        if vct_from_annos_func is None else vct_from_annos_func)
    ect_from_annos_func = (
        (lambda x: ect_from_cilobj)
        if ect_from_annos_func is None else ect_from_annos_func)

    cilobj_kwargs = {
        k: cilobj_kwargs[k] for k
        in (cilobj_kwargs.keys() - {
            "valence_cell_type", "extended_cell_type"})}
    bulk_state = state_from_ngllink(ngllink, request_params)
    bulk_state_layers = bulk_state["layers"]
    nuc_id_to_tags = {
        d["id"]: d["label"]
        for d in bulk_state_layers[nucleus_points_layer]["annotationTags"]}
    nuc_ds = [{"point": d["point"],
               "description": d.get("description"),
               "tags": [nuc_id_to_tags[tid] for tid in d["tagIds"]]}
              for d in bulk_state_layers[nucleus_points_layer][
                  "annotations"]]
    add_lineannotation_nuc_ds(
        bulk_state_layers[base_lines_layer], nuc_ds, "base")
    add_lineannotation_nuc_ds(
        bulk_state_layers[d_cent_lines_layer], nuc_ds, "d_cent")
    add_lineannotation_nuc_ds(
        bulk_state_layers[tip_lines_layer], nuc_ds, "tip")
    add_lineannotation_nuc_ds(
        bulk_state_layers[exit_lines_layer], nuc_ds, "exit")
    add_lineannotation_nuc_ds(
        bulk_state_layers[pcs_lines_layer], nuc_ds, "pcs")

    cilobjs = [
        pycilium.cilia.CiliaAnalysisAnnotatedCell(
            neuroglancer_pt_annotations=d,
            soma_valence_pt_pix=numpy.array(d["point"]),
            valence_cell_type=vct_from_annos_func(d.get("tags")),
            extended_cell_type=ect_from_annos_func(d.get("tags")),
            metadata={
                "annotated_url": ngllink,
                "notes": d["description"],
                "cilia_pcv": (
                    "V" if d.get("pcs") is not None else (
                        "C" if (d.get("tip") is not None and d.get(
                            "base", d.get("exit")) is not None)
                        else 0))
            },
            **cilobj_kwargs) for d in nuc_ds]

    return postprocess_func(cilobjs)
