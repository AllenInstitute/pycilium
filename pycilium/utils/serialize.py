import json

import numpy


class CiliaAnalysisJsonEncoder(json.JSONEncoder):
    """json Encoder in the following hierarchy for serialization:
        obj._to_dict()
        dict(obj)
        JsonEncoder.default(obj)
        obj.__dict__
    """
    def default(self, obj):
        """default encoder for that handles Render objects
        Parameters
        ----------
        obj : obj
            any object that implements _to_dict, dict(obj),
            JsonEncoder.default(obj), or __dict__ (in order)
        Returns
        -------
        dict or list
            json encodable datatype
        """
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        if isinstance(obj, numpy.bool_):
            return bool(obj)
        _to_dict = getattr(obj, "_to_dict", None)
        if callable(_to_dict):
            return obj._to_dict()
        else:
            try:
                return dict(obj)
            except TypeError:
                try:
                    return super(CiliaAnalysisJsonEncoder, self).default(obj)
                except TypeError:  # pragma: no cover
                    return obj.__dict__
