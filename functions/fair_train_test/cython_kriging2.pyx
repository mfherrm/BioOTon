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
        int nst,
        double nug,
        numpy.ndarray[numpy.float64_t, ndim=1] cc,
        numpy.ndarray[numpy.float64_t, ndim=1] aa,
        numpy.ndarray[numpy.float64_t, ndim=1] it,
        numpy.ndarray[numpy.float64_t, ndim=1] ang,
        numpy.ndarray[numpy.float64_t, ndim=1] anis,
        numpy.ndarray[numpy.float64_t, ndim=2] rotmat,
        double maxcov
):
    """
    Calculate the covariance associated with a variogram model specified by
    a nugget effect and nested variogram structures
    :param ang: 
    :param it: 
    :param nug: 
    :param nst: 
    :param x1: 
    :param y1: 
    :param x2: 
    :param y2: 
    :param cc: 
    :param aa: 
    :param anis: 
    :param rotmat: 
    :param maxcov: 
    :return: 
    """

    cdef double epsilon = 0.000001
    cdef double dx = x2 - x1
    cdef double dy = y2 - y1
    cdef double comparer = dx * dx + dy * dy

    cdef double dx1
    cdef double dy1
    cdef double h
    cdef double cova2_ = 0.0
    # Non-zero distance, loop over all the structures
    # Compute the appropriate structural distance
    if comparer >= epsilon:
        for js in range(nst):
            dx1 = dx * rotmat[0, js] + dy * rotmat[1, js]
            dy1 = (dx * rotmat[2, js] + dy * rotmat[3, js]) / anis[js]
            h = sqrt(fmax(dx1 * dx1 + dy1 * dy1, 0.0))
            if it[js] == 1:
                # spherical model
                hr = h / aa[js]
                if hr < 1.0:
                    cova2_ += cc[js] * (1.0 - hr * (1.5 - 0.5 * hr * hr))
            elif it[js] == 2:
                # Exponential model
                cova2_ += cc[js] * numpy.exp(-3.0 * h / aa[js])
            else:
                # Gaussian model
                hh = -3.0 * (h * h) / (aa[js] * aa[js])
                cova2_ += cc[js] * numpy.exp(hh)
    else:
        cova2_ = maxcov

    return cova2_

def simple_krig_var(
        int nst,
        int ndata,
        int nest,
        numpy.ndarray[numpy.float64_t, ndim=1] anis,
        numpy.ndarray[numpy.float64_t, ndim=1] cc,
        numpy.ndarray[numpy.float64_t, ndim=1] aa,
        numpy.ndarray[numpy.float64_t, ndim=1] it,
        numpy.ndarray[numpy.float64_t, ndim=1] ang,
        double nug,
        numpy.ndarray[numpy.float64_t, ndim=1] x_train,
        numpy.ndarray[numpy.float64_t, ndim=1] y_train,
        numpy.ndarray[numpy.float64_t, ndim=1] x_test,
        numpy.ndarray[numpy.float64_t, ndim=1] y_test,
        numpy.ndarray[numpy.float64_t, ndim=2] rotmat,
        double maxcov
):
    """
    Compute the kriging variance only at the testing locations.
    :param nst:
    :param ndata:
    :param nest:
    :param anis:
    :param cc:
    :param aa:
    :param it:
    :param ang:
    :param nug:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param rotmat:
    :param maxcov:
    :return:
    """

    cdef int iest, idata, jdata
    cdef double sill = nug + numpy.sum(cc)
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
                    nst,
                    nug,
                    cc,
                    aa,
                    it,
                    ang,
                    anis,
                    rotmat,
                    maxcov
                )

            r[idata] = cova2(
                x_train[idata],
                y_train[idata],
                x_test[iest],
                y_test[iest],
                nst,
                nug,
                cc,
                aa,
                it,
                ang,
                anis,
                rotmat,
                maxcov
            )
            rr[idata] = r[idata]

        s = ksol_numpy(ndata, a, r)
        for idata in range(0, ndata):
            kriging_variance[iest] = kriging_variance[iest] - s[idata] * rr[idata]
    return kriging_variance
