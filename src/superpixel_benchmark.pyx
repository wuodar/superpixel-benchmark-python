# distutils: sources = lib/evaluation.cpp lib/superpixel_tools.cpp
# distutils: libraries = opencv_imgproc
# distutils: language = c++

cimport opencv_mat
import numpy as np
cimport numpy as np
np.import_array()


cdef extern from "../lib/evaluation.h":
    cdef cppclass Evaluation:
        @staticmethod
        float computeCompactness(const opencv_mat.Mat &labels)
        @staticmethod
        float computeUndersegmentationError(const opencv_mat.Mat &labels, const opencv_mat.Mat &gt)
        @staticmethod
        float computeNPUndersegmentationError(const opencv_mat.Mat &labels, const opencv_mat.Mat &gt)
        @staticmethod 
        float computeLevinUndersegmentationError(const opencv_mat.Mat &labels, const opencv_mat.Mat &gt) 
        @staticmethod 
        float computeOversegmentationError(const opencv_mat.Mat &labels, const opencv_mat.Mat &gt) 
        @staticmethod 
        float computeBoundaryRecall(const opencv_mat.Mat &labels, const opencv_mat.Mat &gt, float d) 
        @staticmethod 
        float computeBoundaryPrecision(const opencv_mat.Mat &labels, const opencv_mat.Mat &gt, float d) 
        @staticmethod 
        float computeExplainedVariation(const opencv_mat.Mat &labels, const opencv_mat.Mat &image) 
        @staticmethod 
        float computeAchievableSegmentationAccuracy(const opencv_mat.Mat &labels, const opencv_mat.Mat &gt) 
        @staticmethod 
        float computeSumOfSquaredErrorRGB(const opencv_mat.Mat &labels, const opencv_mat.Mat &image) 
        @staticmethod 
        float computeSumOfSquaredErrorXY(const opencv_mat.Mat &labels, const opencv_mat.Mat &image) 
        @staticmethod 
        float computeMeanDistanceToEdge(const opencv_mat.Mat &labels, const opencv_mat.Mat &gt) 
        @staticmethod 
        float computeIntraClusterVariation(const opencv_mat.Mat &labels, const opencv_mat.Mat &image) 
        @staticmethod 
        float computeContourDensity(const opencv_mat.Mat &labels) 
        @staticmethod 
        float computeRegularity(const opencv_mat.Mat &labels) 


def compute_compactness(py_labels):
    assert py_labels.dtype == np.intc, "Superpixel array dtype needs to be equal to numpy.intc" 
    
    cppLabels = opencv_mat.np2Mat(py_labels)
 
    return Evaluation.computeCompactness(cppLabels)


def compute_undersegmentation_error(py_labels, py_gt):
    assert py_labels.dtype == np.intc, "Superpixel array dtype needs to be equal to numpy.intc" 
    assert py_gt.dtype == np.intc, "Ground truth array dtype needs to be equal to numpy.intc" 
    
    cppLabels = opencv_mat.np2Mat(py_labels)
    cppGT= opencv_mat.np2Mat(py_gt)
 
    return Evaluation.computeUndersegmentationError(cppLabels, cppGT)


def compute_np_undersegmentation_error(py_labels, py_gt):
    assert py_labels.dtype == np.intc, "Superpixel array dtype needs to be equal to numpy.intc" 
    assert py_gt.dtype == np.intc, "Ground truth array dtype needs to be equal to numpy.intc" 

    cppLabels = opencv_mat.np2Mat(py_labels)
    cppGT= opencv_mat.np2Mat(py_gt)
 
    return Evaluation.computeNPUndersegmentationError(cppLabels, cppGT)


def compute_oversegmentation_error(py_labels, py_gt):
    assert py_labels.dtype == np.intc, "Superpixel array dtype needs to be equal to numpy.intc" 
    assert py_gt.dtype == np.intc, "Ground truth array dtype needs to be equal to numpy.intc" 

    cppLabels = opencv_mat.np2Mat(py_labels)
    cppGT= opencv_mat.np2Mat(py_gt)
 
    return Evaluation.computeOversegmentationError(cppLabels, cppGT)


def compute_levin_undersegmentation_error(py_labels, py_gt):  
    assert py_labels.dtype == np.intc, "Superpixel array dtype needs to be equal to numpy.intc" 
    assert py_gt.dtype == np.intc, "Ground truth array dtype needs to be equal to numpy.intc" 

    cppLabels = opencv_mat.np2Mat(py_labels)
    cppGT= opencv_mat.np2Mat(py_gt)

    return Evaluation.computeLevinUndersegmentationError(cppLabels, cppGT)

def compute_boundary_recall(py_labels, py_gt, d=0.0025):
    assert py_labels.dtype == np.intc, "Superpixel array dtype needs to be equal to numpy.intc" 
    assert py_gt.dtype == np.intc, "Ground truth array dtype needs to be equal to numpy.intc" 

    cppLabels = opencv_mat.np2Mat(py_labels)
    cppGT= opencv_mat.np2Mat(py_gt)
 
    return Evaluation.computeBoundaryRecall(cppLabels, cppGT, d)
        

def compute_boundary_precision(py_labels, py_gt, d=0.0025): 
    assert py_labels.dtype == np.intc, "Superpixel array dtype needs to be equal to numpy.intc" 
    assert py_gt.dtype == np.intc, "Ground truth array dtype needs to be equal to numpy.intc" 

    cppLabels = opencv_mat.np2Mat(py_labels)
    cppGT= opencv_mat.np2Mat(py_gt)

    return Evaluation.computeBoundaryPrecision(cppLabels, cppGT, d)


def compute_explained_variation(py_labels, py_image): 
    assert py_labels.dtype == np.intc, "Superpixel array dtype needs to be equal to numpy.intc" 
    assert py_image.dtype == np.uint8, "Image array dtype needs to be equal to numpy.uint8" 

    cppLabels = opencv_mat.np2Mat(py_labels)
    cppImage = opencv_mat.np2Mat(py_image)
 
    return Evaluation.computeExplainedVariation(cppLabels, cppImage)


def compute_achievable_segmentation_accuracy(py_labels, py_gt):
    assert py_labels.dtype == np.intc, "Superpixel array dtype needs to be equal to numpy.intc" 
    assert py_gt.dtype == np.intc, "Ground truth array dtype needs to be equal to numpy.intc" 

    cppLabels = opencv_mat.np2Mat(py_labels)
    cppGT = opencv_mat.np2Mat(py_gt)
 
    return Evaluation.computeAchievableSegmentationAccuracy(cppLabels, cppGT)
        

def compute_sum_of_squared_error_RGB(py_labels, py_image):
    assert py_labels.dtype == np.intc, "Superpixel array dtype needs to be equal to numpy.intc" 
    assert py_image.dtype == np.uint8, "Image array dtype needs to be equal to numpy.uint8"  

    cppLabels = opencv_mat.np2Mat(py_labels)
    cppImage = opencv_mat.np2Mat(py_image)
 
    return Evaluation.computeSumOfSquaredErrorRGB(cppLabels, cppImage)


def compute_sum_of_squared_eror_XY(py_labels, py_image): 
    assert py_labels.dtype == np.intc, "Superpixel array dtype needs to be equal to numpy.intc" 
    assert py_image.dtype == np.uint8, "Image array dtype needs to be equal to numpy.uint8" 

    cppLabels = opencv_mat.np2Mat(py_labels)
    cppImage = opencv_mat.np2Mat(py_image)
 
    return Evaluation.computeSumOfSquaredErrorXY(cppLabels, cppImage)


def compute_mean_distance_to_edge(py_labels, py_gt):
    assert py_labels.dtype == np.intc, "Superpixel array dtype needs to be equal to numpy.intc" 
    assert py_gt.dtype == np.intc, "Ground truth array dtype needs to be equal to numpy.intc"  

    cppLabels = opencv_mat.np2Mat(py_labels)
    cppGT = opencv_mat.np2Mat(py_gt)
 
    return Evaluation.computeMeanDistanceToEdge(cppLabels, cppGT)


def compute_intra_cluster_variation(py_labels, py_image):
    assert py_labels.dtype == np.intc, "Superpixel array dtype needs to be equal to numpy.intc" 
    assert py_image.dtype == np.uint8, "Image array dtype needs to be equal to numpy.uint8"  

    cppLabels = opencv_mat.np2Mat(py_labels)
    cppImage = opencv_mat.np2Mat(py_image)
 
    return Evaluation.computeIntraClusterVariation(cppLabels, cppImage)


def compute_contour_density(py_labels): 
    assert py_labels.dtype == np.intc, "Superpixel array dtype needs to be equal to numpy.intc" 

    cppLabels = opencv_mat.np2Mat(py_labels)
 
    return Evaluation.computeContourDensity(cppLabels)

