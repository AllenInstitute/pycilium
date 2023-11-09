#!/usr/bin/env python
import concurrent.futures
# import warnings

import requests

from pycilium.utils.import_tools.catmaid_utils import CatmaidApiTokenAuth
from pycilium.utils.skeleton_utils import *

# warnings.simplefilter('once')
# # comment the line below out unless you need it.  Seriously.
# catmaid_request_kwargs = {"verify": False}


class CatmaidApiClient:
    def __init__(self, catmaid_base, auth=None,
                 catmaid_request_kwargs={},
                 default_concurrency=2,
                 default_session_producer=requests.Session):
        self.catmaid_base = catmaid_base
        self.auth = auth
        self.catmaid_request_kwargs = catmaid_request_kwargs
        self._default_concurrency = default_concurrency
        self.default_session_producer = default_session_producer

    def get_json(self, *args, session=None, **kwargs):
        session = (self.default_session_producer()
                   if session is None else session)
        r = session.get(*args, **kwargs)
        r.raise_for_status()
        return r.json()

    def post_json(self, *args, session=None, **kwargs):
        session = (self.default_session_producer()
                   if session is None else session)
        r = session.post(*args, **kwargs)
        r.raise_for_status()
        return r.json()

    def get_cdskel(self, project_id, skel_id):
        return self.get_json(
            f"{self.catmaid_base}/{project_id}/skeletons/{skel_id}/compact-detail",
            params={
                "with_connectors": True,
                "with_tags": True,
                "with_annotations": True
            }, auth=self.auth, **self.catmaid_request_kwargs)

    def get_cdskels(self, project_id, skel_ids, concurrency=None):
        concurrency = (self._default_concurrency if concurrency is None
                       else concurrency)
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as e:
            futs = [e.submit(self.get_cdskel, project_id, sk_id)
                    for sk_id in skel_ids]

            skels = [f.result() for f in futs]
        return skels

    def get_annotations(self, project_id):
        d = self.get_json(
            f"{self.catmaid_base}/{project_id}/annotations",
            auth=self.auth, **self.catmaid_request_kwargs)
        return d["annotations"]

    def get_skeleton_ids(self, project_id):
        return self.get_json(
            f"{self.catmaid_base}/{project_id}/skeletons/",
            auth=self.auth, **self.catmaid_request_kwargs)

    def skeletonids_to_neuronnames(self, project_id, skeleton_ids):
        return self.post_json(
            f"{self.catmaid_base}/{project_id}/skeleton/neuronnames",
            auth=self.auth, data={"skids": skeleton_ids},
            **self.catmaid_request_kwargs)

    def nn_to_skelcds(self, project_id, concurrency=None):
        concurrency = (self._default_concurrency if concurrency is None
                       else concurrency)
        # annoid_to_annotation = {
        #     anno["id"]: anno["name"]
        #     for anno in self.get_annotations(project_id)}

        skels = self.get_skeleton_ids(project_id)
        skid_to_skcd = {
            sk_id: cdskel for sk_id, cdskel
            in zip(skels, self.get_cdskels(
                project_id, skels, concurrency=concurrency))}

        skid_to_nn = self.skeletonids_to_neuronnames(
            project_id, [*skid_to_skcd.keys()])

        nn_to_skids = {}
        for skid, nn in skid_to_nn.items():
            try:
                nn_to_skids[nn].append(int(skid))
            except KeyError:
                nn_to_skids[nn] = [int(skid)]

        return {nn: [skid_to_skcd[skid] for skid in skids]
                for nn, skids in nn_to_skids.items()}
