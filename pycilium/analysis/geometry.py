#!/usr/bin/env python
"""
geometry (/trigonometry) functions for cilium analysis
"""
import numpy


def calculate_theta(disp):
    x, y, z = disp
    return numpy.arctan2(numpy.sqrt(x**2 + y**2), z)


def calculate_phi(disp):
    x, y = disp[:2]
    return numpy.arctan2(y, x)


def calculate_dist(disp):
    return numpy.linalg.norm(disp)


def vec_spherical_coords(disp):
    r = calculate_dist(disp)
    theta = calculate_theta(disp)
    phi = calculate_phi(disp)

    return r, theta, phi


def cosine_similarity(a, b):
    return numpy.dot(a, b) / numpy.linalg.norm(a) / numpy.linalg.norm(b)


def mean_radius_arc_distance(v1, v2, center_pt):
    rv1 = v1 - center_pt
    rv2 = v2 - center_pt

    rmean = numpy.mean([numpy.linalg.norm(rv1),
                        numpy.linalg.norm(rv2)])
    psi = numpy.arccos(cosine_similarity(rv1, rv2))
    return rmean * psi


def _null_pt_preprocessor(pt):
    return pt


class PointGeometryProcessor:
    pt_preprocessor_func = _null_pt_preprocessor

    @classmethod
    def preprocess_pt(cls, pt):
        return cls.pt_preprocessor_func(pt)
