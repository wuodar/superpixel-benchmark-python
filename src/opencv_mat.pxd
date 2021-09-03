import numpy as np
cimport numpy as np
np.import_array()

# For cv::Mat usage
cdef extern from "core/core.hpp":
  cdef int CV_WINDOW_AUTOSIZE
  cdef int CV_8UC3
  cdef int CV_8UC1
  cdef int CV_32FC1
  cdef int CV_32SC1
  cdef int CV_8U
  cdef int CV_32F
  cdef int CV_32S

cdef extern from "core/core.hpp" namespace "cv":
  cdef cppclass Mat:
    Mat() except +
    void create(int, int, int)
    void* data
    int rows
    int cols
    int channels()
    int depth()
    size_t elemSize()

# For Buffer usage
cdef extern from "Python.h":
    ctypedef struct PyObject
    object PyMemoryView_FromBuffer(Py_buffer *view)
    int PyBuffer_FillInfo(Py_buffer *view, PyObject *obj, void *buf, Py_ssize_t length, int readonly, int infoflags)
    enum:
        PyBUF_FULL_RO

cdef Mat np2Mat(np.ndarray ary)

cdef object Mat2np(Mat mat)
