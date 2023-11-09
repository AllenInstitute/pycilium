import dataclasses
import pathlib
import pickle


@dataclasses.dataclass
class CiliaAnalysisDataset:
    cilobjs_array: list
    cn_to_cilobj: dict
    cn_to_skelcd: dict
    cn_to_mpskel: dict
    cn_to_bbox_mesh: dict
        
    @property
    def cellnames(self):
        return (
            self.cn_to_cilobj.keys() | 
            self.cn_to_skelcd.keys() |
            self.cn_to_mpskel.keys() |
            self.cn_to_bbox_mesh.keys()
        )
    
    @property
    def named_cilobjs(self):
        return [*self.cn_to_cilobj.values()]

    @property
    def cilobjs(self):
        return self.cilobjs_array[:]

    @staticmethod
    def _celltype_to_cilobj(cilobjs, ct_attr=None):
        ct_attr = ct_attr or "valence_cell_type"
        ct_to_cilobj = {}
        for cilobj in cilobjs:
            ct = getattr(cilobj, ct_attr)
            try:
                ct_to_cilobj[ct].append(cilobj)
            except KeyError:
                ct_to_cilobj[ct] = [cilobj]
        return ct_to_cilobj

    def celltype_to_cilobj(self, cilobj_filter=None, **kwargs):
        cilobj_filter = cilobj_filter or (lambda x: x)
        return self._celltype_to_cilobj(filter(cilobj_filter, self.cilobjs), **kwargs)
    
    @classmethod
    def from_path(cls, p):
        path = pathlib.Path(p)
        with (path / "cilobjs_array.pkl").open("rb") as f:
            cilobjs_array = pickle.load(f)
        with (path / "cn_to_cilobj.pkl").open("rb") as f:
            cn_to_cilobj = pickle.load(f)
        with (path / "cn_to_skelcd.pkl").open("rb") as f:
            cn_to_skelcd = pickle.load(f)
        try:
            with (path / "cn_to_mpskel_w_radius.pkl").open("rb") as f:
                cn_to_mpskel = pickle.load(f)
        except:
            with (path / "cn_to_mpskel.pkl").open("rb") as f:
                cn_to_mpskel = pickle.load(f)
        try:
            with (path / "cn_to_bbox_mesh.pkl").open("rb") as f:
                cn_to_bbox_mesh = pickle.load(f)
        except Exception as e:
            print(e)
            cn_to_bbox_mesh = {}
        return cls(
            cilobjs_array=cilobjs_array,
            cn_to_cilobj=cn_to_cilobj,
            cn_to_skelcd=cn_to_skelcd,
            cn_to_mpskel=cn_to_mpskel,
            cn_to_bbox_mesh=cn_to_bbox_mesh
        )
