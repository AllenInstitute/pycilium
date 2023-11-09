import json

import numpy


class CiliaAnalysisAnnotatedCell:
    def __init__(self, neuroglancer_pt_annotations=None,
                 static_cilium_seg_id=None, static_soma_seg_id=None,
                 dynamic_cilium_seg_id=None, dynamic_soma_seg_id=None,
                 soma_valence_pt_pix=None,
                 soma_centroid_pix=None, soma_center_mass_pix=None,
                 valence_cell_type=None, extended_cell_type=None,
                 voxel_resolution=None, metadata=None, cellname=None):

        self.neuroglancer_pt_annotations = neuroglancer_pt_annotations
        self.voxel_resolution = numpy.array(voxel_resolution)

        self.static_cilium_seg_id = static_cilium_seg_id
        self.static_soma_seg_id = static_soma_seg_id
        self.dynamic_cilium_seg_id = dynamic_cilium_seg_id
        self.dynamic_soma_seg_id = dynamic_soma_seg_id

        self.valence_cell_type = valence_cell_type
        self.extended_cell_type = extended_cell_type

        self.soma_valence_pt_pix = soma_valence_pt_pix
        self.soma_centroid_pix = soma_centroid_pix
        self.soma_center_mass_pix = soma_center_mass_pix
        self.metadata = ({} if metadata is None else metadata)
        self.cellname = cellname

    # these return None if things don't exist...
    def get_ptannotation(self, anno):
        pt = self.neuroglancer_pt_annotations.get(anno)
        return pt if pt is None else numpy.array(pt)

    def nm_scale(self, pt, allow_none=True):
        return (pt if pt is None and allow_none
                else pt * self.voxel_resolution)

    @property
    def tip_pix(self):
        return self.get_ptannotation("tip")

    @property
    def tip_nm(self):
        return self.nm_scale(self.tip_pix)

    @property
    def base_pix(self):
        return self.get_ptannotation("base")

    @property
    def base_nm(self):
        return self.nm_scale(self.base_pix)

    @property
    def exit_pix(self):
        return self.get_ptannotation("exit")

    @property
    def exit_nm(self):
        return self.nm_scale(self.exit_pix)

    @property
    def exit_or_base_pix(self):
        return self.base_pix if self.exit_pix is None else self.exit_pix

    @property
    def exit_or_base_nm(self):
        return self.nm_scale(self.exit_or_base_pix)

    @property
    def d_cent_pix(self):
        return self.get_ptannotation("d_cent")

    @property
    def d_cent_nm(self):
        return self.nm_scale(self.d_cent_pix)

    @property
    def soma_centroid_nm(self):
        return self.nm_scale(self.soma_centroid_pix)

    @property
    def soma_center_mass_nm(self):
        return self.nm_scale(self.soma_center_mass_pix)

    @property
    def soma_valence_pt_nm(self):
        return self.nm_scale(self.soma_valence_pt_pix)

    @property
    def soma_center_pt_nm(self):
        if self.soma_centroid_nm is None:
            if self.soma_center_mass_nm is None:
                return self.soma_valence_pt_nm
            return self.soma_center_mass_nm
        return self.soma_centroid_nm

    @property
    def prox_nm(self):
        """most proximal point at which cilium exits soma -- base or exit"""
        return self.base_nm if self.exit_nm is None else self.exit_nm

    @staticmethod
    def calculate_theta(disp):
        x, y, z = disp
        return numpy.arctan2(numpy.sqrt(x**2 + y**2), z)

    @staticmethod
    def calculate_phi(disp):
        x, y = disp[:2]
        return numpy.arctan2(y, x)

    @property
    def is_ciliated(self):
        return self.tip_pix is not None and self.prox_nm is not None

    @property
    def has_pocket(self):
        return self.exit_pix is not None

    @property
    def has_mother(self):
        return self.base_pix is not None

    @property
    def has_daughter(self):
        return self.d_cent_pix is not None

    # FIXME this is not sufficiently general
    @property
    def pocket_concealed_surface(self):
        return None if not self.is_ciliated else (
            "pocket" if self.has_pocket else (
                "concealed" if self.extended_cell_type == "OPC" else "surface"
                ))

    def _to_dict(self):
        d = {
            "neuroglancer_pt_annotations": self.neuroglancer_pt_annotations,
            "voxel_resolution": self.voxel_resolution,
            "static_cilium_seg_id": self.static_cilium_seg_id,
            "static_soma_seg_id": self.static_soma_seg_id,
            "dynamic_cilium_seg_id": self.dynamic_cilium_seg_id,
            "dynamic_soma_seg_id": self.dynamic_soma_seg_id,
            "valence_cell_type": self.valence_cell_type,
            "extended_cell_type": self.extended_cell_type,
            "soma_valence_pt_pix": self.soma_valence_pt_pix,
            "soma_centroid_pix": self.soma_centroid_pix,
            "soma_center_mass_pix": self.soma_center_mass_pix,
            "metadata": self.metadata,
            "cellname": self.cellname
        }

        return d

    @classmethod
    def _from_dict(cls, d):
        return cls(**d)

    def __repr__(self):
        return (
            "CiliaAnalysisAnnotatedCell(static_soma_seg_id: {}, "
            "static_cilium_seg_id: {}, dynamic_soma_seg_id: {}, "
            "dynamic_cilium_seg_id: {}, annotations: {})".format(
                self.static_soma_seg_id, self.static_cilium_seg_id,
                self.dynamic_soma_seg_id, self.dynamic_cilium_seg_id,
                json.dumps(self.neuroglancer_pt_annotations)))


def cilobj_ciliated(cilobj):
    return cilobj.tip_pix is not None and cilobj.prox_nm is not None


def cilobj_surface_pocket_concealed(cilobj):
    cell_type = cilobj.extended_cell_type
    ciliated = cilobj_ciliated(cilobj)
    pocket = cilobj.exit_pix is not None
    pocket_concealed_surface = None if not ciliated else (
        "pocket" if pocket else (
            "concealed" if cell_type == "OPC" else "surface"))
    return pocket_concealed_surface