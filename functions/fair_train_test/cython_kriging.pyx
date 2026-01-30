cimport numpy
import numpy
cimport cython
from libc.math cimport sqrt, fmax  # exp

@cython.boundscheck(False)
@cython.wraparound(False)

cdef inline numpy.ndarray[numpy.float64_t, ndim=1] ksol_numpy(
        int neq,
        numpy.ndarray[numpy.float64_t, ndim=2] a,
        numpy.ndarray[numpy.float64_t, ndim=1] r
        ):
    """
    Find solution of a system of linear equations.
    :param neq: 
    :param a: 
    :param r: 
    :return: 
    """
    a = a[0: neq * neq]  # trim the array
    a = numpy.reshape(a, (neq, neq))  # reshape to 2D
    s = numpy.linalg.solve(a, r)  # solve the system of equations
    return s

cdef inline double cova2(
        double x1,
        double y1,
        double x2,
        double y2,
        double cc,
        double aa,
        double anis,
        double rotmat1,
        double rotmat2,
        double rotmat3,
        double rotmat4,
        double maxcov
        ):
    """
    Calculate the covariance associated with a variogram model specified by
    a nugget effect and nested variogram structures
    :param x1: 
    :param y1: 
    :param x2: 
    :param y2: 
    :param cc: 
    :param aa: 
    :param anis: 
    :param rotmat1: 
    :param rotmat2: 
    :param rotmat3: 
    :param rotmat4: 
    :param maxcov: 
    :return: 
    """

    cdef double epsilon = 0.000001
    cdef double cova2
    cdef double dx = x2 - x1
    cdef double dy = y2 - y1
    cdef double comparer = dx * dx + dy * dy

    # Non-zero distance, loop over all the structures
    # Compute the appropriate structural distance
    cdef double dx1 = dx * rotmat1 + dy * rotmat2
    cdef double dy1 = (dx * rotmat3 + dy * rotmat4) / anis
    cdef double h = sqrt(fmax(dx1 * dx1 + dy1 * dy1, 0.0))

    # Gaussian model
    # cdef double hh = -3.0 * (h * h) / (aa * aa)
    # cova2_ = cc * exp(hh)

    # Spherical model
    hr = h / aa
    if hr < 1.0:
        cova2_ = cc * (1.0 - hr * (1.5 - 0.5 * hr * hr))

    # compare if you are in the same location
    if comparer < epsilon:
        cova2_ = maxcov
    else:
        pass

    return cova2_

def simple_krig_var(
        int ndata,
        int nest,
        double anis,
        double cc,
        double aa,
        double nug,
        numpy.ndarray[numpy.float64_t, ndim=1] x_train,
        numpy.ndarray[numpy.float64_t, ndim=1] y_train,
        numpy.ndarray[numpy.float64_t, ndim=1] x_test,
        numpy.ndarray[numpy.float64_t, ndim=1] y_test,
        double rotmat1,
        double rotmat2,
        double rotmat3,
        double rotmat4,
        double maxcov
        ):
    """
    Compute the kriging variance only at the testing locations.
    :param ndata:
    :param nest:
    :param anis:
    :param cc:
    :param aa:
    :param nug:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param rotmat1:
    :param rotmat2:
    :param rotmat3:
    :param rotmat4:
    :param maxcov:
    :return:
    """

    cdef int iest, idata, jdata
    cdef double sill = nug + cc
    cdef numpy.ndarray[numpy.float64_t, ndim=2] a = numpy.zeros([ndata, ndata])
    cdef numpy.ndarray[numpy.float64_t, ndim=1] r = numpy.zeros(ndata)
    cdef numpy.ndarray[numpy.float64_t, ndim=1] s = numpy.zeros(ndata)
    cdef numpy.ndarray[numpy.float64_t, ndim=1] rr = numpy.zeros(ndata)
    cdef numpy.ndarray[numpy.float64_t, ndim=1] kriging_variance = numpy.full(nest, sill)

    # Make and solve the kriging matrix, calculate the kriging estimate and variance
    for iest in range(nest):
        for idata in range(ndata):
            for jdata in range(ndata):
                a[idata, jdata] = cova2(
                    x_train[idata],
                    y_train[idata],
                    x_train[jdata],
                    y_train[jdata],
                    cc,
                    aa,
                    anis,
                    rotmat1,
                    rotmat2,
                    rotmat3,
                    rotmat4,
                    maxcov
                )

            r[idata] = cova2(
                x_train[idata],
                y_train[idata],
                x_test[iest],
                y_test[iest],
                cc,
                aa,
                anis,
                rotmat1,
                rotmat2,
                rotmat3,
                rotmat4,
                maxcov
            )
            rr[idata] = r[idata]

        s = ksol_numpy(ndata, a, r)
        for idata in range(0, ndata):
            kriging_variance[iest] = kriging_variance[iest] - s[idata] * rr[idata]
    return kriging_variance
