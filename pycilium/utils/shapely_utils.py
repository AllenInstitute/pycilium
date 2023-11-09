import numpy
import shapely
import shapely.wkb


def numpy_to_ewkt(arr):
    return shapely.wkb.dumps(shapely.geometry.point.Point(*arr)).hex()


def ewkt_to_numpy(ewkb):
    return numpy.array(
        shapely.wkb.loads(
            (ewkb.hex() if isinstance(ewkb, bytes) else ewkb),
            hex=True))


def linesegments_to_mlsewkt(linesegments):
    return shapely.wkb.dumps(
        shapely.geometry.multilinestring.MultiLineString(
            linesegments)).hex()
