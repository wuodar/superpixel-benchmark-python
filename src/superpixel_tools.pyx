# distutils: sources = lib/evaluation.cpp lib/superpixel_tools.cpp
# distutils: libraries = opencv_imgproc
# distutils: language = c++

cimport opencv_mat
import numpy as np
cimport numpy as np
np.import_array()


cdef extern from "../lib/superpixel_tools.h":
    cdef cppclass SuperpixelTools:
        @staticmethod
        int computeRegionSizeFromSuperpixels(const opencv_mat.Mat &image, int superpixels)
        @staticmethod
        void computeHeightWidthFromSuperpixels(const opencv_mat.Mat &image, int superpixels, int &height, int &width)
        @staticmethod
        void computeHeightWidthLevelsFromSuperpixels(const opencv_mat.Mat &image, int superpixels, int &height, int &width, int &levels)
        @staticmethod
        void computeRegionSizeLevels(const opencv_mat.Mat &image, int superpixels, int &region_size, int &levels)
        @staticmethod
        void relabelSuperpixels(opencv_mat.Mat &labels)
        @staticmethod
        void computeLabelsFromBoundaries(const opencv_mat.Mat &image, const opencv_mat.Mat &boundaries, opencv_mat.Mat &labels, int BOUNDARY_VALUE, int INNER_VALUE)
        @staticmethod
        void assignBoundariesToSuperpixels(const opencv_mat.Mat &image, const opencv_mat.Mat &boundaries, opencv_mat.Mat &labels, int BOUNDARY_VALUE)        
        @staticmethod
        int countSuperpixels(const opencv_mat.Mat &labels)
        @staticmethod
        int relabelConnectedSuperpixels(opencv_mat.Mat &labels)
        @staticmethod
        int enforceMinimumSuperpixelSize(const opencv_mat.Mat &image, opencv_mat.Mat &labels, int size)
        @staticmethod
        int enforceMinimumSuperpixelSizeUpTo(const opencv_mat.Mat &image, opencv_mat.Mat &labels, int number)


def count_superpixels(py_labels):
    assert py_labels.dtype == np.intc, "Superpixel array dtype needs to be equal to numpy.intc" 
    
    cppLabels = opencv_mat.np2Mat(py_labels)    
    
    count = SuperpixelTools.countSuperpixels(cppLabels)

    return count


def enforce_minimum_superpixel_size(py_image, py_labels, size):
    assert py_labels.dtype == np.intc, "Superpixel array dtype needs to be equal to numpy.intc" 
    assert py_image.dtype == np.uint8, "Image array dtype needs to be equal to numpy.uint8" 

    cppImage = opencv_mat.np2Mat(py_image)    
    cppLabels = opencv_mat.np2Mat(py_labels)    
    
    new_labels_count = SuperpixelTools.enforceMinimumSuperpixelSize(cppImage, cppLabels, size)
    new_labels = opencv_mat.Mat2np(cppLabels)
    
    return new_labels.copy() 


def enforce_minimum_superpixel_size_up_to(py_image, py_labels, number):
    assert py_labels.dtype == np.intc, "Superpixel array dtype needs to be equal to numpy.intc" 
    assert py_image.dtype == np.uint8, "Image array dtype needs to be equal to numpy.uint8" 

    cppImage = opencv_mat.np2Mat(py_image)    
    cppLabels = opencv_mat.np2Mat(py_labels)    

    new_labels_count = SuperpixelTools.enforceMinimumSuperpixelSizeUpTo(cppImage, cppLabels, number)
    new_labels = opencv_mat.Mat2np(cppLabels)
    
    return new_labels.copy()


def compute_region_size_from_superpixels(py_image, superpixels):
    assert py_image.dtype == np.uint8, "Image array dtype needs to be equal to numpy.uint8" 

    cppImage = opencv_mat.np2Mat(py_image)    

    size = SuperpixelTools.computeRegionSizeFromSuperpixels(cppImage, superpixels)

    return size


def compute_height_width_from_sp(py_image, superpixels, height, width):
    assert py_image.dtype == np.uint8, "Image array dtype needs to be equal to numpy.uint8" 

    cppImage = opencv_mat.np2Mat(py_image)    

    SuperpixelTools.computeHeightWidthFromSuperpixels(cppImage, superpixels, height, width)

    return height, width


def compute_height_width_levels_from_sp(py_image, superpixels, height, width, levels):
    assert py_image.dtype == np.uint8, "Image array dtype needs to be equal to numpy.uint8" 

    cppImage = opencv_mat.np2Mat(py_image)   

    SuperpixelTools.computeHeightWidthLevelsFromSuperpixels(cppImage, superpixels, height, width, levels)

    return height, width, levels


def compute_region_size_levels(py_image, superpixels, region_size, levels):
    assert py_image.dtype == np.uint8, "Image array dtype needs to be equal to numpy.uint8" 

    cppImage = opencv_mat.np2Mat(py_image)

    SuperpixelTools.computeRegionSizeLevels(cppImage, superpixels, region_size, levels)

    return region_size, levels


def relabel_superpixels(py_labels):
    assert py_labels.dtype == np.intc, "Superpixel array dtype needs to be equal to numpy.intc" 
    
    cppLabels = opencv_mat.np2Mat(py_labels)
    
    SuperpixelTools.relabelSuperpixels(cppLabels)
    relabeled_labels = opencv_mat.Mat2np(cppLabels)

    return relabeled_labels.copy()


def compute_labels_from_boundaries(py_image, py_boundaries, boundary_val=-1, inner_val=-2):
    assert py_image.dtype == np.uint8, "Image array dtype needs to be equal to numpy.uint8" 

    cppImage = opencv_mat.np2Mat(py_image)
    cppBoundaries = opencv_mat.np2Mat(py_boundaries)
    cppLabels = opencv_mat.np2Mat(np.empty_like(py_boundaries))

    SuperpixelTools.computeLabelsFromBoundaries(cppImage, cppBoundaries, cppLabels, boundary_val, inner_val)
    computed_labels = opencv_mat.Mat2np(cppLabels)

    return computed_labels.copy()


def assign_boundaries_to_superpixels(py_image, py_boundaries, boundary_val=-1):
    assert py_image.dtype == np.uint8, "Image array dtype needs to be equal to numpy.uint8" 

    cppImage = opencv_mat.np2Mat(py_image)
    cppBoundaries = opencv_mat.np2Mat(py_boundaries)
    cppLabels = opencv_mat.np2Mat(np.empty_like(py_boundaries))

    SuperpixelTools.assignBoundariesToSuperpixels(cppImage, cppBoundaries, cppLabels, boundary_val)        
    assigned_boundaries = opencv_mat.Mat2np(cppLabels)

    return assigned_boundaries.copy()


def relabel_connected_superpixels(py_labels):
    assert py_labels.dtype == np.intc, "Superpixel array dtype needs to be equal to numpy.intc" 

    cppLabels = opencv_mat.np2Mat(py_labels)
    
    new_superpixels_count = SuperpixelTools.relabelConnectedSuperpixels(cppLabels)
    relabeled_labels = opencv_mat.Mat2np(cppLabels)
    
    return relabeled_labels.copy()