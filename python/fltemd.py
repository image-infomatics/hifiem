'''
Filename:       fltemd
Created:        25-Jan-2021
Author:         Yuri Kreinin
Description:    This file contains common methods and algorithms used in preprocessing of electron microscopy
                images.
'''

import numpy as np, gc, time, cv2 as cv
from math import floor, sqrt, ceil, radians, exp, sin, cos
from contextlib import nullcontext
from scipy.ndimage import convolve, convolve1d, maximum_filter, minimum_filter, generate_binary_structure, maximum_filter1d, \
    label, binary_dilation, correlate1d, distance_transform_edt
from scipy.special import erfc, expit
from scipy.stats import norm
from skimage.measure import regionprops
# from skimage.morphology

# ---- redirect classes and functions previously exported by this file
from fltlib import struct as _struct, timedelta_to_string as _timedelta_to_string, time_it, safe_sqrt
struct = _struct
timedelta_to_string = _timedelta_to_string

#===========================================================================================================================================
def conv2d(image, kern):
    '''
        a simple helper for 2-dimensional separable convolution, based on openCV method

    Parameters
    ----------
    image : numpy array
        2d or 3d numpy array, as defined by cv2.filter2D
    kern : one dimensional numpy array or object of class kernel, defined below
        one dimensional convolutional filter to be applied in both directions

    Returns
    -------
    numpy array of the same shape as input
        result of 2d convolution
    '''
    if isinstance(kern, kernel): kern = kern.kern

    # applying cv.sepFilter2D is meaningfully slower than cv.filter2D
    # CONV = lambda image, kern: cv.sepFilter2D(image, ddepth = -1, kernelX = kern, kernelY = kern)
    CONV = lambda image: cv.filter2D(cv.filter2D(image, ddepth = -1, kernel = kern.reshape(1, -1)), ddepth = -1, kernel = kern.reshape(-1, 1))
    if np.ndim(image) == 2:
        return CONV(image)
    output = np.zeros_like(image)
    for idx in range(image.shape[0]):
        output[idx, ...] = CONV(image[idx])
    return output

#===========================================================================================================================================
def mean2d(image, size):
    '''
        a simple helper for 2-dimensional block filter based on openCV blur function

    Parameters
    ----------
    image : numpy array
        2d or 3d numpy array, as defined by cv2.blur
    size : int or tuple or list of ints
        defines the size of 2d block kernel. If integer scalar is provided as a parameter
        then the kernel is assumed to be square

    Returns
    -------
    numpy array of the same shape as input
        result of applying 2d block filter

    '''
    return cv.blur(image, (size, size) if np.isscalar(size) else size)

#===========================================================================================================================================
def preprocess_image(image, kern = None, percentile = 2, dtype = None, rescale = None):
    '''
        this method is used to preprocess and normalize image to the positive range starting with zero (usually 0..1).
        It also may clip outlier values and apply a 2d convolution filter prior to other operations

    Parameters
    ----------
    image : numpy array
        2d or 3d numpy array
    kern : numpy array
        The default is None, otherwise contains one-dimensional separable filter.
    percentile : int or None, optional
        The percentile to be used to clip intensity outliers. The default is 2.
    dtype : numpy.dtype, optional
        Specifies a type to convert output. The default is None, which means that result is np.float32 or np.float64
    rescale : float or integer, optional
        Specifies the upper-bound of output intensity range. The default is None, which results in 1

    Returns
    -------
    image : TYPE
        DESCRIPTION.

    '''
    if not kern is None:
        image = conv2d(image, kern)
    if percentile:
        min_image, max_image = np.nanpercentile(image, [percentile, 100 - percentile])
        image = np.clip(image, min_image, max_image)
    else:
        min_image, max_image = np.min(image), np.max(image)

    image = (image - min_image) / (max_image - min_image)
    if rescale and rescale != 1:
        image *= rescale
    if dtype:
        image = image.astype(dtype)

    return image

#===========================================================================================================================================
class kernel:
    '''
    class kernel encapsulates basic convolution operations and supports n-dimensional separable convolutions. The default 7-tap filter is derived
    from Mallat Wavelet Transform and is suitable for dilated convolutions of any order
    '''
    kernHxH = np.array(( 0.015625,  0.09375,   0.234375,  0.3125,    0.234375,   0.09375,  0.015625), dtype = np.float32)
    kernGxK = np.array((-0.015625, -0.09375,  -0.234375,  0.6875,   -0.234375,  -0.09375, -0.015625), dtype = np.float32)

    #---------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self, kern = None, copy = False):
        self.kern = kern.kern if isinstance(kern, kernel) else kernel.kernHxH if kern is None else np.array(kern)
        if copy:
            self.kern = self.kern.copy()

    #---------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def block(size, dtype = np.float32):
        return kernel(np.full((size,), 1 / size, dtype = dtype))

    #---------------------------------------------------------------------------------------------------------------------------------------
    def convn(self, input, mode = "mirror", ndims = None):
        if self.kern.ndim > 1:
            return convolve(input, self.kern, mode = mode)

        ndims = ndims if not ndims is None else input.ndim
        output = input
        for ndim in range(ndims): # apply convolution in reverse dimensions order (columns are in the last subscript)
            output = convolve1d(output, self.kern, axis = -(ndim+1), mode = mode)

        return output

    #---------------------------------------------------------------------------------------------------------------------------------------
    def dilate(self, scaleNo, as_kernel = False):
        step = 1 << scaleNo
        output = np.zeros((np.array(self.kern.shape) - 1) * step + 1, dtype = self.kern.dtype)
        output[::step] = self.kern
        return kernel(output) if as_kernel else output

    #---------------------------------------------------------------------------------------------------------------------------------------
    def complement(self, val = 1, as_kernel = False):
        output = -self.kern
        # output[tuple([x for x in np.round((np.array(x.shape)-1) / 2).astype(int)])] += val
        # output[(*np.round((np.array(x.shape)-1) / 2).astype(int).tolist(),)] += val
        output[(*np.round((np.array(output.shape)-1) / 2).astype(int),)] += val # complement the central tap
        return kernel(output) if as_kernel else output

    #---------------------------------------------------------------------------------------------------------------------------------------
    def __invert__(self):
        return self.complement(as_kernel = True)

    #---------------------------------------------------------------------------------------------------------------------------------------
    def kern_nd(self, ndims = 3):
        output = weights = self.kern.squeeze()
        for _ in range(ndims-1):
            output = output[..., np.newaxis] * weights
        return output

    #---------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def same_shape(kern_1, kern_2):
        kern_1 = kern_1.kern if isinstance(kern_1, kernel) else np.array(kern_1)
        kern_2 = kern_2.kern if isinstance(kern_2, kernel) else np.array(kern_2)
        ndim_1 = kern_1.ndim
        ndim_2 = kern_2.ndim
        if ndim_1 < ndim_2:
            kern_1 = np.expand_dims(kern_1, axis = tuple(range(ndim_2 - ndim_1)))
        elif ndim_1 > ndim_2:
            kern_2 = np.expand_dims(kern_2, axis = tuple(range(ndim_1 - ndim_2)))

        padw = lambda kern_1, kern_2 : tuple([(int(max(x, 0)),) for x in (np.array(kern_1.shape) - np.array(kern_2.shape)) / 2])
        kern_1 = np.pad(kern_1, pad_width = padw(kern_2, kern_1))
        kern_2 = np.pad(kern_2, pad_width = padw(kern_1, kern_2))
        return kern_1, kern_2

    #---------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def add_kernels(kern_1, kern_2):
        kern_1, kern_2 = kernel.same_shape(kern_1, kern_2)
        return kern_1 + kern_2, kern_1, kern_2

    #---------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def sub_kernels(kern_1, kern_2):
        kern_1, kern_2 = kernel.same_shape(kern_1, kern_2)
        return kern_1 - kern_2, kern_1, kern_2

    #---------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def lopass_kernel(scales = 6, as_kernel = False):
        kern = kernel.kernHxH
        for scale_idx in range(1, scales):
            kern = np.correlate(kern, kernel().dilate(scale_idx, as_kernel = False), mode = 'full')
        return kernel(kern) if as_kernel else kern

    #---------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def hipass_kernel(scales = 6, smooth = True):
        kern = kernel.lopass_kernel(scales, as_kernel = True)
        return (kernel() - kern) if smooth else ~kern

    #---------------------------------------------------------------------------------------------------------------------------------------
    def __mul__(self, other):
        if isinstance(other, kernel) or not np.isscalar(other):
            kern_1, kern_2 = kernel.same_shape(self, other)
            return kernel(np.multiply(kern_1, kern_2))
        else:
            return kernel(other * self.kern)

    #---------------------------------------------------------------------------------------------------------------------------------------
    def __rmul__(self, other):
        return self.__mul__(other)

    #---------------------------------------------------------------------------------------------------------------------------------------
    def __imul__(self, other):
        if isinstance(other, kernel) or not np.isscalar(other):
            kern_1, kern_2 = kernel.same_shape(self, other)
            self.kern = np.multiply(kern_1, kern_2)
        else:
            self.kern *= other

        return self

    #---------------------------------------------------------------------------------------------------------------------------------------
    def __neg__(self):
        return kernel(-self.kern)

    #---------------------------------------------------------------------------------------------------------------------------------------
    def __add__(self, kern):
        return kernel(kernel.add_kernels(self, kern)[0])

    #---------------------------------------------------------------------------------------------------------------------------------------
    def __sub__(self, kern):
        return kernel(kernel.sub_kernels(self, kern)[0])

#===========================================================================================================================================
def conndef(ndims, minimal = True):
    return generate_binary_structure(ndims, 1) if minimal else np.ones(ndims * (3,), dtype = bool)

#===========================================================================================================================================
def denoise_v_2_10(image, sigmoid_params: list, ndims = None, scales = 3, wsize = 7, split_hipass = True, propagate_noise = True):
    '''
    The method for volume de-nosing. The noise estimation is done based on covariance between low- and high-pass bands,
    a suppression ratio is attenuated  by sigmoid function with parametrized coefficients

    Parameters
    ----------
    image : n-dimensional array to be processed
        it is recommended to be casted to np.float32
    ndims : integer scalar, optional
        number of convolution dimensions equals to image dimensions by default (None)
    sigmoid_params : a list of dictionaries
        contains parameters to be used for each scale. The number of entries reflects a number of scales. In case of split_hipass parameter
        is True, a number of dictionaries is greater by one than number of scales, because the highest band is splitted
    scales : integer scalar, optional
        number of scales (dilations of low-pass kernel - kernel.kernHxH). The default is None and can be calculated from sigmoid_params
        and split_hipass arguments.
    split_hipass : boolean, optional
        specifies if the first scale should be splitted to high and low bands for more accurate
        The default is True.
    wsize : integer scalar, optional
        number of taps for the one-dimensional separable averaging filter used to calculate local statisitcs. The default is 7.
    propagate_noise : boolean, optional
        The default is True.

    Examples of usage:
        >>> denoised = denoise_v_2_10(rawdata.astype(np.float32), sigmoid_params = [{"gain": 1.0, "offset": -0.5}, {"gain": 1.5, "offset": 0.7}])

    Returns
    -------
    ndarray of np.float32
    '''
    sigmoid_params = np.asarray(sigmoid_params)
    ndims  = image.ndim if ndims is None else ndims
    kernA  = kernel.block(wsize, dtype = kernel.kernHxH.dtype)
    scales = scales if not scales is None else (sigmoid_params.size - 1) if split_hipass else sigmoid_params.size
    kernLP = kernel(copy = True)

    CONVN   = lambda x, kern: (kern if isinstance(kern, kernel) else kernel(kern)).convn(x, ndims = ndims)
    COVAR   = lambda lp, hp, kern: CONVN(np.multiply(lp, hp), kern) - np.multiply(CONVN(lp, kern), CONVN(hp, kern))
    NRATIO  = lambda x, gain, offset: np.nan_to_num(np.reciprocal(1.0 + np.exp(gain * x + offset)), copy = False) # we estimate noise ratio, therefore sigmoid is flipped

    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(kernel.kernHxH.dtype)

    # process the highest frequency component
    lopass  = CONVN(image, kernLP)
    details = image - lopass

    if split_hipass:
        # high-frequency decomposition
        lp_hipass = CONVN(2 * details - CONVN(details, kernLP), kernLP) # lp_hipass contains low frequency part of hipass
        hp_hipass = details - lp_hipass # hp_hipass contains high frequency component
        noise     = hp_hipass * NRATIO(COVAR(hp_hipass, lopass + lp_hipass, kernA), sigmoid_params[0]["gain"], sigmoid_params[0]["offset"])
        if scales > 0: # if scales is 0 than only hiband of the highest scale should be processed
            noise += lp_hipass * NRATIO(COVAR(lp_hipass, lopass, kernA), sigmoid_params[1]["gain"], sigmoid_params[1]["offset"])
        details  -= noise
        del lp_hipass, hp_hipass
    else:
        noise    = details * NRATIO(COVAR(details, lopass, kernA), sigmoid_params[0]["gain"], sigmoid_params[0]["offset"])
        details -= noise

    # process the rest of scales
    for scale_idx in range(1, scales):
        # prev = lopass
        if propagate_noise:
            lopass += CONVN(noise, kernLP.dilate(scale_idx - 1))

        hipass   = lopass # temporary stores previous lopass
        lopass   = CONVN(lopass, kernLP.dilate(scale_idx))
        hipass  -= lopass
        kernA    = kernel.block((wsize - 1) * (1 << scale_idx) + 1, dtype = kernel.kernHxH.dtype)
        params   = sigmoid_params[scale_idx + split_hipass] # adjust parameters index by one if split_hipass is True
        noise    = hipass * NRATIO(COVAR(hipass, lopass, kernA), params["gain"], params["offset"])
        details += hipass - noise

    return lopass + details

#===========================================================================================================================================
def create_edge_mask(hipass, strel):
    strel = conndef(strel) if np.isscalar(strel) else strel
    return np.sign(maximum_filter(hipass, footprint = strel)) != np.sign(minimum_filter(hipass, footprint = strel))

#===========================================================================================================================================
def denoise_v_3_00(image, ndims = None, scales = 3, split_hipass = True, strel = "minimal", expand_window = False,
                   mix_lopass = 0, use_lopass = False, scale_reduction = 1, verbose = False):
    '''
    The method for volume de-nosing, based on the concept similar to nonlocal means averaging. It receives the following arguments:
        image - n-dimensional array to be processed, it is recommended to be casted to np.float32
        ndims = None, number of convolution dimensions equals to image dimensions by default (None)
        scales  = 3, number of scales (dilations of low-pass kernel - kernel.kernHxH)
        split_hipass = True
        strel = "minimal", detect pixels neighboring feature's edge and apply slightly different averaging rule
        expand_window = False, expand the averaging filter with respect to the scale
        mix_lopass = 0, mix lopass and high-pass derived averaging, in proportion (1 - mix_lopass) .* hp_avg + mix_lopass .* lopass,
        use_lopass = False, use lopass component for averaging. This parameter has little sense if mix_lopass != 0
        scale_reduction = 1,  specifies the reduction coefficient of growing band, as multiplier = scale_reduction .^ (scaleIdx - 1)
    Examples of usage:
        denoised = denoise_v_3_00(rawdata) - use the default parameters
        - or -
        denoised = denoise_v_3_00(rawdata, ndims = 3, scales = 2) - use the custom parameters
    '''
    if verbose: print("Start denoising")
    ndims  = image.ndim if ndims is None else ndims
    strel  = strel if (strel is None) or (not isinstance(strel, str)) else conndef(ndims, strel.lower() == "minimal")

    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(kernel.kernHxH.dtype)

    CONVN = lambda x, kern: (kern if isinstance(kern, kernel) else kernel(kern)).convn(x, ndims = ndims)

    # define pixel averaging and accumulation method, w.r.t. algorithm parameters
    def accumulate(hipass, lopass, mul = 1):
        kern = acc.kern
        pixs = prev if use_lopass else image
        mask_p = (hipass >= 0).astype(pixs.dtype)
        if strel is None:
            acc_image  = mask_p * CONVN(pixs * mask_p, kern)
            acc_weight = mask_p * CONVN(mask_p, kern)

            mask_p = 1 - mask_p
            acc_image  += mask_p * CONVN(pixs * mask_p, kern)
            acc_weight += mask_p * CONVN(mask_p, kern)
        else:
            mask_b = create_edge_mask(hipass, strel).astype(pixs.dtype)
            mask_n = 1 - mask_p
            acc_image  = mask_b * (mask_p * CONVN(pixs * mask_p, kern) + mask_n * CONVN(pixs * mask_n, kern))
            acc_weight = mask_b * (mask_p * CONVN(mask_p, kern) + mask_n * CONVN(mask_n, kern))

            mask_b = (1 - mask_b);  mask_p *= mask_b;  mask_n *= mask_b
            acc_image  += mask_p * CONVN(pixs * mask_p, kern) + mask_n * CONVN(pixs * mask_n, kern)
            acc_weight += mask_p * CONVN(mask_p, kern) + mask_n * CONVN(mask_n, kern)

        if mix_lopass != 0:
            acc_image  *= (1 - mix_lopass)
            acc_image  += mix_lopass * lopass
            acc_weight *= (1 - mix_lopass)
            acc_weight += mix_lopass

        if mul != 1:
            acc_image  *= mul
            acc_weight *= mul

        acc.image  += acc_image
        acc.weight += acc_weight
    # end of local function: accumulate

    kernLP = kernel(copy = True)
    lopass = CONVN(image, kernLP)
    hipass = image - lopass

    acc = struct(kern = kernel.kernHxH.copy())
    if split_hipass:
        if verbose: print("Band 0.Hi", end = ">> ")
        hipass -= CONVN(2 * hipass - CONVN(hipass, kernLP), kernLP) # hipass contains high frequency component

        mask    = (hipass >= 0).astype(image.dtype)
        acc.image  = mask * CONVN(image * mask, kernLP)
        acc.weight = mask * CONVN(mask, kernLP)

        mask = 1 - mask
        acc.image  += mask * CONVN(image * mask, kernLP)
        acc.weight += mask * CONVN(mask, kernLP)
        del mask

        if mix_lopass != 0:
            acc.image  *= (1 - mix_lopass)
            acc.image  += mix_lopass * (image - hipass)
            acc.weight *= (1 - mix_lopass)
            acc.weight += mix_lopass

        if verbose: print("Band 0.Lo", end = ">> ")
        if use_lopass:
            prev = image - hipass

        accumulate(image - hipass - lopass, lopass)
    else:
        if verbose: print("Band 0", end = ">> ")
        prev  = image
        acc.image   = np.zeros_like(image)
        acc.weights = np.zeros_like(image)
        accumulate(hipass, lopass)

    for scaleIdx in range(1, scales):
        mul = scale_reduction ** (scaleIdx - 1)
        if verbose: print(f"scale {scaleIdx} (w = {mul:.2%})", end = ">> ")

        prev   = lopass
        kern   = kernLP.dilate(scaleIdx)
        lopass = CONVN(lopass, kern)
        accumulate(prev - lopass, lopass, mul)

        if expand_window:
            acc.kern = np.correlate(acc.kern, kern, mode = "full")

    return acc.image / (acc.weight + np.finfo(np.float32).eps)

#===========================================================================================================================================
def enhance_v_1_00(image, ndims = None, scales = 3, enhance = 1.0, strel = "minimal", expand_window = False, scale_diff = True,
                   method = "samesign", verbose = False):
    '''
    The method for edge enhancement to be applied on image denoised by denoise_v_2_00 function.
    Replicates matlab function EmdLib.Enhance_v_1_00. sample_2.1 was processed by this method

    Parameters
    ----------
    image : n-dimensional array to be processed
        it is recommended to be casted to np.float32

    ndims : scalar, optional, the default is None
        number of convolution dimensions, equals to image dimensions by default

    scales : scalar, optional, the default is 3
        number of scales (dilations of low-pass kernel - kernel.kernHxH).

    strel : optional, the default is "minimal"
        detect pixels neighboring feature's edge and apply slightly different averaging rule

    expand_window : bool, optional, default is False
        expands the averaging filter with respect to the scale.

    scale_diff : boolean, optional, the default is True
        specifies whenever this method enhances differences between scales or difference between lopass (scale) and the input image

    enhance : scalar, optional, the default is 1.0
        multiplier to be used in order to control a magnitude of enhancement

    method : string, optional, the default is "samesign"
        there are three different methods of how high-frequency component is used for edge enhancement

        "simple" - just mutliplies high frequency component of voxels residing near by edge

        "lopass" - applies averaging (lopass) for enhanced high frequency component

        "samesign" - the most sophisitcated method applies averaging (lopass) with regards to the sign of high-frequency component

    Examples of usage:
    ------------------
        denoised = enhance_v_1_00(denoised_image) - use the default parameters
    '''
    if verbose: print("Start enhancement")
    ndims   = image.ndim if ndims is None else ndims
    strel   = conndef(ndims) if strel is None else conndef(ndims, strel.lower() == "minimal") if isinstance(strel, str) else strel
    enhance = enhance if np.isscalar(enhance) else np.array(enhance)

    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(kernel.kernHxH.dtype)

    CONVN = lambda x, kern: (kern if isinstance(kern, kernel) else kernel(kern)).convn(x, ndims = ndims)

    # define run_enhance method with regards to method argument
    if (method is None) or (method.lower() == "simple"):
        run_enhance = lambda hipass, mask, mul: mul * mask * hipass
    elif method.lower() == "lopass":
        run_enhance = lambda hipass, mask, mul: mul * mask * CONVN(hipass, kernLP)
    else: # "samesign" is the default method
        def run_enhance(hipass, mask, mul):
            mask_p = (hipass >= 0).astype(np.float32)
            mask_n = 1 - mask_p
            return mul * mask * (mask_p * CONVN(hipass * mask_p, kernLP) + mask_n * CONVN(hipass * mask_n, kernLP))

    kernLP = kernel.kernHxH.copy()
    lopass = image
    output = image.copy()
    for scaleIdx in range(0, scales):
        hipass  = lopass
        lopass  = CONVN(lopass, kernel().dilate(scaleNo = scaleIdx, as_kernel = True))
        hipass -= lopass
        output += run_enhance(hipass = hipass, \
                              mask   = create_edge_mask(hipass if scale_diff else (image - lopass), strel).astype(np.float32), \
                              mul    = enhance if np.isscalar(enhance) else enhance[min(enhance.size-1, scaleIdx)])
        if expand_window:
            kernLP = np.correlate(kernLP, kernel().dilate(scaleNo = scaleIdx + 1), 'full')

    return output

#===========================================================================================================================================
def enhance_v_2_00(image, ndims = None, scales = 3, enhance = 1.0, strel = "minimal", expand_window = False,
                   scale_diff = True, preserve_max = True, edge_only = False, verbose = False):
    '''
    The method for edge enhancement to be applied on image denoised by denoise_v_2_00 function
        image - n-dimensional array to be processed, it is recommended to be casted to np.float32
        ndims = None, number of convolution dimensions equals to image dimensions by default (None)
        scales  = 3, number of scales (dilations of low-pass kernel - kernel.kernHxH)
        strel = "minimal", detect pixels neighboring feature's edge and apply slightly different averaging rule
        expand_window = False, expand the averaging filter with respect to the scale
        scale_diff = True, specifies whenever this method enhances differences between scales or difference between lopass (scale) and the input image
        enhance = 1.0, multiplier to be used in order to control a magnitude of enhancement
        edge_only = False
    Examples of usage:
        denoised = enhance_v_1_00(denoised_image) - use the default parameters
    '''
    if verbose: print("Start enhancement")
    ndims   = image.ndim if ndims is None else ndims
    strel   = conndef(ndims) if strel is None else conndef(ndims, strel.lower() == "minimal") if isinstance(strel, str) else strel
    enhance = enhance if np.isscalar(enhance) else np.array(enhance)

    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(kernel.kernHxH.dtype)

    CONVN = lambda x, kern: (kern if isinstance(kern, kernel) else kernel(kern)).convn(x, ndims = ndims)

    # define run_enhance method with regards to function's argument
    def run_enhance(hipass, mask):
        EPS = np.finfo(hipass.dtype).eps
        mask_p = (hipass >= 0).astype(np.float32)
        mask_n = 1 - mask_p
        if edge_only:
            mask_p *= mask
            mask_n *= mask
            if preserve_max:
                return mask_p * np.maximum(hipass, CONVN(hipass * mask_p, kernLP) / (CONVN(mask_p, kernLP) + EPS)) \
                     + mask_n * np.minimum(hipass, CONVN(hipass * mask_n, kernLP) / (CONVN(mask_n, kernLP) + EPS))
            else:
                return mask_p * CONVN(hipass * mask_p, kernLP) / (CONVN(mask_p, kernLP) + EPS) + mask_n * CONVN(hipass * mask_n, kernLP) / (CONVN(mask_n, kernLP) + EPS)
        else:
            weight = CONVN(mask_p, kernLP)
            if preserve_max:
                return mask * (mask_p * np.maximum(hipass, CONVN(hipass * mask_p, kernLP) / (weight + EPS)) + mask_n * np.minimum(hipass, CONVN(hipass * mask_n, kernLP) / (1 - weight + EPS)))
            else:
                return mask * (mask_p * CONVN(hipass * mask_p, kernLP) / (weight + EPS) + mask_n * CONVN(hipass * mask_n, kernLP) / (1 - weight + EPS))
    # end of local function: run_enhance

    kernLP  = kernel.kernHxH.copy()
    lopass  = image
    details = np.zeros_like(image)
    for scaleIdx in range(0, scales):
        hipass   = lopass
        lopass   = CONVN(lopass, kernel().dilate(scaleNo = scaleIdx, as_kernel = True))
        hipass  -= lopass

        mask = create_edge_mask(hipass if scale_diff else (image - lopass), strel).astype(details.dtype)
        mul  = enhance if np.isscalar(enhance) else enhance(min(enhance.size, scaleIdx))
        details += (1 - mask) * hipass + mul * run_enhance(hipass, mask)

        if expand_window:
            kernLP = np.correlate(kernLP, kernel().dilate(scaleNo = scaleIdx + 1), 'full')

    return lopass + details

#===========================================================================================================================================
def enhance_v_3_00(image, ndims = None, scales = 3, enhance = 1.0, strel = "minimal", expand_window = False, scale_diff = True,
                   method = "samesign", rangeV = None, sigmaV = None, sigmaR = 0, sizeA = 129, scurve = "gaussian", verbose = False):
    if verbose: print("Start enhancement")
    ndims   = image.ndim if ndims is None else ndims
    strel   = conndef(ndims) if strel is None else conndef(ndims, strel.lower() == "minimal") if isinstance(strel, str) else strel
    params  = struct(method = [method] if np.isscalar(method) else method,
                     rangeV = [rangeV] if (rangeV is None) or np.isscalar(rangeV) else rangeV,
                     sigmaV = [sigmaV] if (sigmaV is None) or np.isscalar(sigmaV) else sigmaV,
                     sigmaR = [sigmaR] if (sigmaR is None) or np.isscalar(sigmaR) else sigmaR,
                     sizeA  = [sizeA]  if np.isscalar(sizeA) else sizeA,
                     enhance = [enhance] if np.isscalar(enhance) else enhance)

    CONVN = lambda x, kern: (kern if isinstance(kern, kernel) else kernel(kern)).convn(x, ndims = ndims)
    # ENHANCE_SIMPLE = lambda hipass, mask, mul: mul * mask * hipass
    # ENHANCE_LOPASS = lambda hipass, mask, mul: mul * mask * CONVN(hipass, kernLP)
    EPS = np.finfo(np.float32).eps
    GAUSSIAN = lambda x, mean, sigma: np.exp(-0.5 * np.square((x - mean) / (sigma + EPS)))
    def GAUSSIAN_SCURVE(x, mu, sigma): # 1 for x < mu and gaussian s-curve for x > mu
        mask = (x <= mu).astype(np.float32)
        return mask + (1 - mask) * GAUSSIAN(x, mu, sigma)
    SIGMOID_SCURVE  = lambda x, mu, sigma: np.reciprocal(1 + np.exp(-(3 * (mu - x) / abs(sigma) + 6))) # 1 for x < mu and inverse sigmoid s-curve for x > mu
    SCURVE = locals()[scurve.upper() + "_SCURVE"]

    kernLP = kernel.kernHxH.copy()
    lopass = image.astype(np.float32, copy = True)
    # output = lopass.copy()
    details = np.zeros(image.shape, dtype = np.float32)
    for scale_idx in range(0, scales):
        hipass  = lopass - CONVN(lopass, kernel().dilate(scaleNo = scale_idx, as_kernel = True))
        lopass -= hipass

        if len(params.method)  > scale_idx: method  = params.method[scale_idx]
        if len(params.sigmaV)  > scale_idx: sigmaV  = params.sigmaV[scale_idx]
        if len(params.rangeV)  > scale_idx: rangeV  = params.rangeV[scale_idx]
        if len(params.sigmaR)  > scale_idx: sigmaR  = params.sigmaR[scale_idx]
        if len(params.enhance) > scale_idx: enhance = params.enhance[scale_idx]
        if (len(params.sizeA)  > scale_idx) and (not params.sizeA[scale_idx] is None):
            kernA = np.full((params.sizeA[scale_idx],), 1 / params.sizeA[scale_idx], dtype=np.float32)

        positive = (hipass > 0).astype(np.float32)
        if verbose:
            print(f"scale {scale_idx}: positive={np.mean(positive):.3f}, std(hp)={np.std(hipass):.3f}, std(lp)={np.std(lopass):.3f}, mean(lp)={np.mean(lopass):.2f}", end = "")

        if sigmaV is None:
            weights = 1
        elif isinstance(rangeV, str):
            if rangeV.lower() == "auto":
                weights = SCURVE(lopass, CONVN(lopass * positive, kernA) / (CONVN(positive, kernA) + EPS), sigmaV) \
                        * SCURVE(CONVN(lopass - lopass * positive, kernA) / (CONVN(1 - positive, kernA) + EPS), lopass, sigmaV)
            elif rangeV.lower() == "left":
                weights = SCURVE(lopass, CONVN(lopass * (1-positive), kernA) / (CONVN(1-positive, kernA) + EPS), sigmaV)
            elif rangeV.lower() == "right": # this option implementation differs from Matlab, because weights decreases for lopass > mu
                weights = SCURVE(lopass, CONVN(lopass * positive, kernA) / (CONVN(positive, kernA) + EPS), sigmaV)
            elif rangeV.lower() == "mean": # the same as "middle" in Matlab
                weights = SCURVE(lopass, CONVN(lopass, kernA), sigmaV)
            else:
                assert False, f"unsupported rangeV value '{rangeV}'"
        elif (not rangeV is None) and not np.isscalar(rangeV):
            weights = SCURVE(lopass, rangeV[1], sigmaV) * SCURVE(rangeV[0], lopass, sigmaV)
        elif SCURVE == GAUSSIAN_SCURVE:
            weights = GAUSSIAN(lopass, CONVN(lopass, kernA) if (rangeV is None) else rangeV, sigmaV)
        else:
            weights = CONVN(lopass, kernA) if (rangeV is None) else rangeV # use this variable in order to reduce memory footprint
            weights = SCURVE(lopass, weights, sigmaV) * SCURVE(weights, lopass, sigmaV)

        if (not sigmaV is None) and (sigmaV < 0): weights = 1 - weights

        if not sigmaR is None:
            bwedge = create_edge_mask(hipass if scale_diff else (image - lopass), strel)
            if verbose:
                print(f", mean(weights)={np.mean(weights):.3f}, mean(edge)={np.mean(bwedge):.3f}", end = "")
            if (sigmaR == 0):
                weights *= bwedge.astype(np.float32)
            else:
                weights *= SCURVE(distance_transform_edt(~bwedge, return_distances = True, return_indices = False) , 0, sigmaR)

        if verbose:
            print(f", mean(w)={np.mean(weights):.3f}", end="")

        weights *= enhance
        if (method is None) or (method.lower() == "simple"):
            hipass += weights * hipass
        elif method.lower() == "lopass":
            hipass += weights * CONVN(hipass, kernLP)
        else: # "samesign" is the default method
            hipass += weights * (positive * CONVN(hipass * positive, kernLP) + (1-positive) * CONVN(hipass * (1-positive), kernLP))
            # the same
            # hipass += weights * ((2 * positive - 1) * CONVN(hipass * positive, kernLP) - (positive - 1) * CONVN(hipass, kernLP))

        details += hipass
        if verbose:
            print(f", std(hipass)={np.std(hipass):.3f}, std(details)={np.std(details):.3f}")

        if expand_window:
            kernLP = np.correlate(kernLP, kernel().dilate(scaleNo = scale_idx + 1), 'full')

    return lopass + details

#===========================================================================================================================================
def enhance_v_3_10(image, ndims = None, scales = 3, enhance = 1.0, hipass = 1.0, strel = "minimal", expand_window = False,
                   scale_diff = True, method = "samesign", rangeV = None, sigmaV = None, sigmaR = 0, sizeA = 129,
                   scurve = "gaussian", distance = None, verbose = False):

    if verbose: print("Start enhancement")
    ndims   = image.ndim if ndims is None else ndims
    strel   = conndef(ndims) if strel is None else conndef(ndims, strel.lower() == "minimal") if isinstance(strel, str) else strel
    to_list = lambda v: [v] if (v is None) or np.isscalar(v) else v
    params  = struct(method = to_list(method), rangeV   = to_list(rangeV), sigmaV  = to_list(sigmaV),
                     sigmaR = to_list(sigmaR), sizeA    = to_list(sizeA),  enhance = to_list(enhance),
                     hipass = to_list(hipass), distance = to_list(distance))

    CONVN = lambda x, kern: (kern if isinstance(kern, kernel) else kernel(kern)).convn(x, ndims = ndims)
    EPS = np.finfo(np.float32).eps
    GAUSSIAN = lambda x, mean, sigma: np.exp(-0.5 * np.square((x - mean) / (sigma + EPS)))
    def GAUSSIAN_SCURVE(x, mu, sigma): # 1 for x < mu and gaussian s-curve for x > mu
        mask = (x <= mu).astype(np.float32)
        return mask + (1 - mask) * GAUSSIAN(x, mu, sigma)
    # SIGMOID_SCURVE  = lambda x, mu, sigma: np.reciprocal(1 + np.exp(-(3 * (mu - x) / abs(sigma) + 6))) # 1 for x < mu and inverse sigmoid s-curve for x > mu
    SIGMOID_SCURVE = lambda x, mu, sigma: expit(3 * (mu - x) / np.abs(sigma) + 6) # 1 for x < mu and inverse sigmoid s-curve for x > mu
    SCURVE = locals()[scurve.upper() + "_SCURVE"]
    EDGE_RATIO = [ 1.42, 1.866, 2.055 ] # constant ratio used to adjust convolution calculated distance weights

    kernLP  = kernel.kernHxH.copy()
    kernHxH = kernel.kernHxH.copy()
    lopass  = image.astype(np.float32, copy = True)
    # output = lopass.copy()
    details = np.zeros(image.shape, dtype = np.float32)
    for scale_idx in range(0, scales):
        hipass  = lopass - CONVN(lopass, kernel().dilate(scaleNo = scale_idx, as_kernel = True))
        lopass -= hipass

        if len(params.method)  > scale_idx: method  = params.method[scale_idx]
        if len(params.sigmaV)  > scale_idx: sigmaV  = params.sigmaV[scale_idx]
        if len(params.rangeV)  > scale_idx: rangeV  = params.rangeV[scale_idx]
        if len(params.sigmaR)  > scale_idx: sigmaR  = params.sigmaR[scale_idx]
        if len(params.hipass)  > scale_idx: alpha   = params.hipass[scale_idx]
        if len(params.enhance) > scale_idx: enhance = params.enhance[scale_idx]
        if len(params.distance) > scale_idx: distance = params.distance[scale_idx]
        if (len(params.sizeA)  > scale_idx) and (not params.sizeA[scale_idx] is None):
            kernA = np.full((params.sizeA[scale_idx],), 1 / params.sizeA[scale_idx], dtype=np.float32)

        positive = (hipass > 0).astype(np.float32)
        if verbose:
            print(f"scale {scale_idx}: positive={np.mean(positive):.3f}, std(hp)={np.std(hipass):.3f}, std(lp)={np.std(lopass):.3f}, mean(lp)={np.mean(lopass):.2f}", end = "")

        if sigmaV is None:
            weightsV = 1
        elif isinstance(rangeV, str):
            if rangeV.lower() == "auto":
                weightsV = SCURVE(lopass, CONVN(lopass * positive, kernA) / (CONVN(positive, kernA) + EPS), sigmaV) \
                        * SCURVE(CONVN(lopass - lopass * positive, kernA) / (CONVN(1 - positive, kernA) + EPS), lopass, sigmaV)
            elif rangeV.lower() == "left":
                weightsV = SCURVE(lopass, CONVN(lopass * (1-positive), kernA) / (CONVN(1-positive, kernA) + EPS), sigmaV)
            elif rangeV.lower() == "right": # this option implementation differs from Matlab, because weights decreases for lopass > mu
                weightsV = SCURVE(lopass, CONVN(lopass * positive, kernA) / (CONVN(positive, kernA) + EPS), sigmaV)
            elif rangeV.lower() == "mean": # the same as "middle" in Matlab
                weightsV = SCURVE(lopass, CONVN(lopass, kernA), sigmaV)
            else:
                assert False, f"unsupported rangeV value '{rangeV}'"
        elif (not rangeV is None) and not np.isscalar(rangeV):
            weightsV = SCURVE(lopass, rangeV[1], sigmaV) * SCURVE(rangeV[0], lopass, sigmaV)
        elif SCURVE == GAUSSIAN_SCURVE:
            weightsV = GAUSSIAN(lopass, CONVN(lopass, kernA) if (rangeV is None) else rangeV, sigmaV)
        else:
            weightsV = CONVN(lopass, kernA) if (rangeV is None) else rangeV # use this variable in order to reduce memory footprint
            weightsV = SCURVE(lopass, weightsV, sigmaV) * SCURVE(weightsV, lopass, sigmaV)

        if (not sigmaV is None) and (sigmaV < 0): weightsV = 1 - weightsV

        if (not distance is None) and (distance.lower() != "ignore"):
            bwedge = create_edge_mask(hipass if scale_diff else (image - lopass), strel)
            if verbose: print(f", mean(weightsV)={np.mean(weightsV):.3f}, mean(edge)={np.mean(bwedge):.3f}", end = "")
            if distance.lower() == "conv":
                weightsV *= EDGE_RATIO[min(len(EDGE_RATIO)-1, scale_idx)] * CONVN(bwedge.astype(np.float32), kernHxH)
            elif (distance.lower() == "dist") and (sigmaR != 0):
                weightsV *= SCURVE(distance_transform_edt(~bwedge, return_distances = True, return_indices = False) , 0, sigmaR)
            else: # "edge"
                weightsV *= bwedge.astype(np.float32)
            del bwedge


        if verbose: print(f", mean(weightsV)={np.mean(weightsV):.3f}", end="")

        weightsV *= enhance
        if (method is None) or (method.lower() == "simple"):
            hipass *= (weightsV + alpha)
        elif method.lower() == "lopass":
            hipass  = alpha * hipass + weightsV * CONVN(hipass, kernLP)
        else: # "samesign" is the default method
            hipass  = alpha * hipass + weightsV * (positive * CONVN(hipass * positive, kernLP) + (1-positive) * CONVN(hipass * (1-positive), kernLP))

        details += hipass
        if verbose:
            print(f", std(hipass)={np.std(hipass):.3f}, std(details)={np.std(details):.3f}")

        if expand_window:
            kernLP = np.correlate(kernLP, kernel().dilate(scaleNo = scale_idx + 1), 'full')

    return lopass + details


#===========================================================================================================================================
def deband_v_1_00(image, direction = "horizontal", scales = 6, kernB = 121, factor = 1, verbose = False):
    if np.isscalar(kernB):
        kernB = np.ones((kernB,), dtype = kernel.kernHxH.dtype) / kernB
        kernB = np.correlate(kernB, kernB, mode = 'full')

    ndimA = -2; ndimB = -1
    if not (direction.lower() in ["horz", "horizontal"]):
        ndimA, ndimB = ndimB, ndimA

    CONV = lambda x, w, axis: convolve1d(x, w, axis = axis, mode = "mirror")

    kernG  = kernel(kernel.kernGxK)
    output = image.astype(kernel.kernHxH.dtype) if np.issubdtype(image.dtype, np.integer) else np.copy(image)
    lopass = np.copy(output)
    kernS  = np.ones((3,), dtype = np.float32) / 3
    # mask   = (CONV(CONV(np.square(output), kernS, -1), kernS, -2) - np.square(CONV(CONV(output, kernS, -1), kernS, -2)) != 0).astype(np.float32)
    mask   = (CONV(CONV(np.square(output), kernS, -1), kernS, -2) - np.square(CONV(CONV(output, kernS, -1), kernS, -2)) != 0)

    for scaleIdx in range(scales):
        hipass  = CONV(lopass, kernG.dilate(scaleIdx), ndimA)
        hipass2 = np.square(hipass)
        thresh  = np.mean(hipass2[mask]) - np.mean(hipass[mask])**2
        if verbose:
            print(f"Scale {scaleIdx}: thresh = {np.sqrt(thresh):.4}", end = " ")
        mapV    = np.exp(-0.5 * np.maximum(0, (hipass2 / (factor * thresh)) - 1))
        lopass -= hipass

        output[mask] -= CONV(hipass * mapV, kernB, ndimB)[mask]

    if verbose:
        print()

    return output

#===========================================================================================================================================
def deband_v_1_01(image, direction = "horizontal", scales = 6, kernB = 251, weight_ratio = 1, weight_thresh = 1,
                  weight_method = "NPDF", bwmask = None, verbose = False):
    '''
    improved version of deband_v_1_00. Code replicates modified matlab function EmdLib.Deband_v_2_00

    Parameters
    ----------
    image : 2D or 3D ndarray.
        Contains image slices of any numeric type
    direction : string, optional.
        Specifies bands direction ('horizontal' / 'vertical'). The default is "horizontal".
    scales : integer, optional.
        Number of scales to be processed. The default is 6.
    kernB : integer, optional.
        Length of debanding average kernel or kernel vector if this parameter os not scalar. The default is 251.
    weight_ratio : numeric, optional.
        Used for calculating band contribution weights. The default is 1.
    weight_thresh : numeric, optional.
        Used for calculating band contribution weights. The default is 1.
    weight_method : string, optional.
        Method (function) to be used in calculation of band contribution weights ("NPDF", "SIGMOID", "ERFC"), The default is "NPDF"
    bwmask : boolean ndarray of the same size as image or None
        Specifies bitmask of active area to be used in calculation. If None (default) the mask is calculated internally
    verbose : boolean, optional
        The default is True.

    Returns
    -------
    output : np.float32, ndarray of the same size as image
        Contains debanded image
    '''
    EPS = np.finfo(np.float32).eps
    if np.isscalar(kernB):
        kernB = np.ones((kernB,), dtype = kernel.kernHxH.dtype) / kernB
        kernB = np.correlate(kernB, kernB, mode = 'full')

    ndimA = -2; ndimB = -1
    if not (direction.lower() in ["horz", "horizontal"]):
        ndimA, ndimB = ndimB, ndimA

    CONV = lambda x, w, axis: convolve1d(x, w, axis = axis, mode = "mirror")
    ERFC = lambda x, thresh = weight_thresh, ratio = weight_ratio: 0.5 * erfc((ratio / sqrt(2)) * np.abs(x) - (ratio * thresh + 3) / sqrt(2))
    SIGMOID = lambda x, thresh = weight_thresh, ratio = weight_ratio: 1 / (1 + np.exp(ratio * np.abs(x) - (ratio * thresh + 3)))
    NPDF = lambda x, thresh = weight_thresh, ratio = weight_ratio: np.exp(-ratio * np.square(np.maximum(np.abs(x) - thresh, 0)))

    if not callable(weight_method):
        weight_method = NPDF if weight_method is None else locals()[weight_method]

    kernG  = kernel(kernel.kernGxK)
    output = image.astype(kernel.kernHxH.dtype) if np.issubdtype(image.dtype, np.integer) else np.copy(image)
    lopass = np.copy(output)
    if bwmask is None:
        kernS  = np.ones((3,), dtype = np.float32) / 3
        bwmask = minimum_filter(CONV(CONV(np.square(output), kernS, -1), kernS, -2) - np.square(CONV(CONV(output, kernS, -1), kernS, -2)),
                                size = (1, 3, 3) if image.ndim == 3 else (3, 3)) > 0

    for scale_idx in range(scales):
        hipass  = CONV(lopass, kernG.dilate(scale_idx), ndimA)
        power   = sqrt(np.mean(np.square(hipass[bwmask])))
        weights = weight_method(hipass / power, weight_thresh, weight_ratio)
        bands   = CONV(hipass * weights * bwmask, kernB, ndimB) / (CONV(bwmask.astype(np.float32), kernB, ndimB) + EPS)
        lopass -= hipass
        output[bwmask] -= bands[bwmask]

        if verbose:
            print(f"Scale {scale_idx}: stdev = {power:.2f}", end = " ")

    if verbose:
        print()

    return output

#===========================================================================================================================================
def deband_v_2_00(image, direction = "horizontal", scales = 6, bands_len = 121, bwmask = None,
                  adjust_beta = False, use_nearest = False, details = struct(start = 1), verbose = False):
    '''
    version of de-banding introduced by matlab function EmdLib.Deband_v_4_00, used to process sample_1.2

    Parameters
    ----------
    image : 2D or 3D ndarray.
        Contains image slices of any numeric type
    direction : string, optional.
        Specifies bands direction ('horizontal' / 'vertical'). The default is "horizontal".
    scales : integer, optional.
        Number of scales to be processed. The default is 6.
    bands_len : integer scalar, list of integers or numpy array of integers
        Specifies the length of the stripe to be used for debanding of each scale. If number of elements in bands_len is less
        than number of scales bands_len[-1] is used for the last scales. The default is 121.
    bwmask : boolean ndarray of the same size as image or None
        Specifies bitmask of active area to be used in calculation. If None (default) the mask is calculated internally
    adjust_beta : boolean, optional
        Specifies that the scale debanding should be adjusted for the loss of signal strength due to low pass filtering. The default is False.
    use_nearest : boolen, optional
        Specifies method of value substitution for pixels, which exceeds detail-detection threshold. By default their value is bounded to
        threshold, but it may be also replaced by the nearest pixel with value bellow threshold. The default is False.
    details : dictionary or struct, optional
        Specifies parameters used to calculate stripe-limiting threshold. The default is struct(start = 1).

    Returns
    -------
     output : np.float32, ndarray of the same size as image
        Contains debanded image
    '''

    EPS = np.finfo(np.float32).eps

    # if isinstance(bands_len, str): bands_len = ast.literal_eval(bands_len)
    bands_len = np.array([bands_len]) if np.isscalar(bands_len) else np.array(bands_len)
    ndimA = -2; ndimB = -1
    if not (direction.lower() in ["horz", "horizontal"]):
        ndimA, ndimB = ndimB, ndimA

    # if isinstance(details, str): details = ast.literal_eval(details)
    if details is None:
        calc_threshold = lambda v: v
        kernB = None
    else:
        if np.isscalar(details):
            X = details
            kernB = None
        else:
            shape = np.ones((image.ndim,), dtype = int); shape[ndimB] = image.shape[ndimB]
            details = struct(**details) if isinstance(details, dict) else details
            if not details.isfield("start"): details.start = 1
            if (not details.isfield("end")) or details.end == details.start:
                X = details.start
            elif not details.isfield("middle"): # linear function
                X = details.start + (details.end - details.start) * np.arange(image.shape[ndimB], dtype = np.float32).reshape(shape) / (image.shape[ndimB]-1)
            else:
                if not details.isfield("mul"): details.middle = 20
                X = details.end - (details.end - details.start) \
                     / (1 + np.exp(details.mul * (np.arange(image.shape[ndimB], dtype = np.float32).reshape(shape) / (image.shape[ndimB]-1) - details.middle)))

        calc_threshold = lambda v: X * v
        kernB = np.ones((details.len,), dtype = np.float32) if details.isfield("len") else None

    # try: kernB = np.ones((details.len,), dtype = np.float32) / details.len
    # except: kernB = None

    if bwmask is None:
        CONV = lambda x, w, axis = ndimB: correlate1d(x, w, axis = axis, mode = "mirror")
        if not kernB is None: countX = details.len * image.shape[ndimA]
    elif not bwmask.any():
        return image.astype(np.float32, copy = False)
    else:
        bwfloat = bwmask.astype(np.float32)
        CONV = lambda x, w, axis = ndimB: bwfloat * correlate1d(x * bwfloat, w, axis = axis, mode = "mirror") / (correlate1d(bwfloat, w, axis = axis, mode = "mirror") + EPS)
        if not kernB is None: countX = correlate1d(bwfloat.sum(axis = ndimA, keepdims = True), kernB, axis = ndimB, mode = "mirror") + EPS

    kernHxH  = kernel(kernel.kernHxH)
    output = image.astype(np.float32) if np.issubdtype(image.dtype, np.integer) else np.copy(image)
    lopass = np.copy(output)

    for scale_idx in range(scales):
        hipass  = CONV(lopass, kernHxH.dilate(scale_idx))
        lopass  = CONV(hipass, kernHxH.dilate(scale_idx), ndimA)
        hipass -= lopass

        threshold = ((hipass**2).mean() if bwmask is None else (hipass[bwmask]**2).mean())
        if kernB is None:
            threshold = np.sqrt(threshold)
        else:
            threshold = np.sqrt(np.maximum(0.5 * threshold, correlate1d((hipass**2).sum(axis = ndimA, keepdims = True), kernB, axis = ndimB, mode = "mirror") / countX))

        threshold = calc_threshold(threshold)
        if use_nearest:
            bands = hipass[tuple(distance_transform_edt(((abs(hipass) >= threshold) | ~bwmask) if not bwmask is None else (abs(hipass) >= threshold),
                                                        return_distances = False, return_indices = True))]
        else:
            bands = np.maximum(-threshold, np.minimum(threshold, hipass))

        if scale_idx < bands_len.size:
            kernA = np.ones((bands_len[scale_idx],), dtype = np.float32) / bands_len[scale_idx]
            kernA = np.correlate(kernA, kernA, mode = 'full')
            if adjust_beta: beta = sqrt((kernel(kernA).complement()**2).sum())

        bands   = CONV(bands, kernA)
        output -= ((beta - 1) * hipass + bands) / beta if adjust_beta else bands

    return output


#===================================================================================================================
def performance_time(text, start):
    call = time.time() - start
    return f"{text} - {call // 60 :.0f} mins {call - (call // 60)*60 :.2f} secs"

#===========================================================================================================================================
def deband_v_3_00(image, direction = "horizontal", bwmask = None, scales = 6, adjust_beta = False, details = struct(start = 1),
                  destripe = [{"method" : "limit", "bands_len" : 121}], verbose = False):
    '''
    version of de-striping introduced by matlab function EmdLib.Deband_v_5_00

    Parameters
    ----------
    image : 2D or 3D ndarray.
        Contains image slices of any numeric type
    direction : string, optional.
        Specifies bands direction ('horizontal' / 'vertical'). The default is "horizontal".
    scales : integer, optional.
        Number of scales to be processed. The default is 6.
    bwmask : boolean ndarray of the same size as image or None
        Specifies bitmask of active area to be used in calculation. If None (default) the mask is calculated internally
    adjust_beta : boolean, optional
        Specifies that the scale debanding should be adjusted for the loss of signal strength due to low pass filtering. The default is False.
    details : dictionary or struct, optional
        Specifies parameters used to calculate stripe-limiting threshold. The default is struct(start = 1).
    destripe : list or tuple of dictionaries or structs
        for each frequency band (scale) defines de-striping method and its parameters. The number of elements is supposed to meet value
        specified by scales parameter. If scales parameter is larger than len(destripe), the last element destripe[-1] is used for the rest
        of scales. Each element of the list contains dictionary (or struct) with following items:
            bands_len : integer scalar
                specifies the length of the stripe to be used for debanding of each scale.
            mask_len : integer scalar, optional
                specifies the size of two dimensional window used for averaging of weight matrix
            thresh_mul : float scalar, optional
                if provided, specifies the multiplier to be used for adjusting threshold values
            method : str
                specifies method of de-striping technique to be applied
                "limit" - clips values to the range [-threshold ... threshold]
                "nearest" - replaces values that are out of range [-threshold ... threshold] by the values of the nearest pixels in the range
                "ratio" - weights pixels that are out of range [-threshold ... threshold] with the weight specified by "weight" item
                "predict" - for connected component formed by pixels, which are out of threshold range, calculates probability of being stripe

    Returns
    -------
     output : np.float32, ndarray of the same size as image
        Contains debanded image
    '''

    EPS = np.finfo(np.float32).eps

    # if isinstance(bands_len, str): bands_len = ast.literal_eval(bands_len)
    # bands_len = np.array([bands_len]) if np.isscalar(bands_len) else np.array(bands_len)
    ndimA = -2; ndimB = -1
    if not (direction.lower() in ["horz", "horizontal"]):
        ndimA, ndimB = ndimB, ndimA

    if details is None:
        calc_threshold = lambda v: v
        kernB = None
    else:
        if np.isscalar(details):
            X = details
            kernB = None
        else:
            shape = np.ones((image.ndim,), dtype = int); shape[ndimB] = image.shape[ndimB]
            details = struct(**details) if isinstance(details, dict) else details
            if not details.isfield("start"): details.start = 1
            if (not details.isfield("end")) or details.end == details.start:
                X = details.start
            elif not details.isfield("middle"): # linear function
                X = details.start + (details.end - details.start) * np.arange(image.shape[ndimB], dtype = np.float32).reshape(shape) / (image.shape[ndimB]-1)
            else:
                if not details.isfield("mul"): details.middle = 20
                X = details.end - (details.end - details.start) \
                     / (1 + np.exp(details.mul * (np.arange(image.shape[ndimB], dtype = np.float32).reshape(shape) / (image.shape[ndimB]-1) - details.middle)))

        calc_threshold = lambda v: X * v
        kernB = np.ones((details.len,), dtype = np.float32) if details.isfield("len") else None

    if bwmask is None:
        CONV = lambda x, w, axis = ndimB: correlate1d(x, w, axis = axis, mode = "mirror")
        MASK = lambda x, size: kernel.block(size).convn(x.astype(np.float32), ndims = 2)
        if not kernB is None: countX = details.len * image.shape[ndimA]
    elif not bwmask.any():
        return image.astype(np.float32, copy = False)
    else:
        bwfloat = bwmask.astype(np.float32)
        CONV = lambda x, w, axis = ndimB: bwfloat * correlate1d(x * bwfloat, w, axis = axis, mode = "mirror") / (correlate1d(bwfloat, w, axis = axis, mode = "mirror") + EPS)
        MASK = lambda x, size: kernel.block(size).convn((x | ~bwmask).astype(np.float32), ndims = 2)
        if not kernB is None: countX = correlate1d(bwfloat.sum(axis = ndimA, keepdims = True), kernB, axis = ndimB, mode = "mirror") + EPS

    AVERAGE = lambda x, size: kernel.block(size).convn(x, ndims = 2)

    # if any([method.lower() == "predict" ])

    kernHxH  = kernel(kernel.kernHxH)
    output = image.astype(np.float32) if np.issubdtype(image.dtype, np.integer) else np.copy(image)
    lopass = np.copy(output)

    for scale_idx in range(scales):
        if verbose: start = time.time()
        hipass  = CONV(lopass, kernHxH.dilate(scale_idx))
        lopass  = CONV(hipass, kernHxH.dilate(scale_idx), ndimA)
        hipass -= lopass

        threshold = ((hipass**2).mean() if bwmask is None else (hipass[bwmask]**2).mean())
        if kernB is None:
            threshold = np.sqrt(threshold)
        else:
            threshold = np.sqrt(np.maximum(0.5 * threshold, correlate1d((hipass**2).sum(axis = ndimA, keepdims = True), kernB, axis = ndimB, mode = "mirror") / countX))

        threshold = calc_threshold(threshold)

        if scale_idx < len(destripe):
            params = struct(**destripe[scale_idx]) if isinstance(destripe[scale_idx], dict) else destripe[scale_idx]
            kernA  = np.full((params.bands_len,), 1 / params.bands_len, dtype = np.float32)
            kernA  = np.correlate(kernA, kernA, mode = 'full')
            params.method = params.method.lower()
            if adjust_beta: beta = sqrt((kernel(kernA).complement()**2).sum())

        if params.isfield("thresh_mul", True) and params.thresh_mul != 1:
            threshold *= params.thresh_mul


        if params.method == "limit":
            bands = np.maximum(-threshold, np.minimum(threshold, hipass))
            if params.isfield("mask_len"):
                bwtemp = MASK(np.fabs(hipass) <= threshold, params.mask_len)
                bands  = bwtemp * CONV(hipass * bwtemp, kernA)
            else:
                bands  = CONV(bands, kernA)

        elif params.method == "nearest":
            bwtemp = np.fabs(hipass) <= threshold
            if not bwmask is None: bwtemp |= ~bwmask
            bands = hipass[tuple(distance_transform_edt(bwtemp, return_distances = False, return_indices = True))]
            if params.isfield("mask_len"):
                bwtemp = MASK(~bwtemp, params.mask_len)
                bands  = bwtemp * CONV(bands * bwtemp, kernA)
            else:
                bands  = CONV(bands, kernA)

        elif params.method == "ratio":
            bwtemp = np.ones(hipass.shape, dtype = np.float32)
            bwtemp[np.fabs(hipass) > threshold] = params.weight
            if params.isfield("mask_len"):
                bwtemp = AVERAGE(bwtemp, params.mask_len)
                bands  = bwtemp * CONV(hipass * bwtemp, kernA)
            else:
                bands  = CONV(hipass * bwtemp, kernA)

        elif params.method == "predict":

            strel = conndef(2, minimal = False)
            if image.ndim == 3: strel = strel[np.newaxis, ...]

            if verbose: print(performance_time(f"scale {scale_idx} start prediction", start))
            bwtemp = np.fabs(hipass) > threshold
            if not bwmask is None: bwtemp &= bwmask
            # props  = regionprops(label(bwtemp, return_num = False), cache = False)
            bwlabel, num_of_labels = label(bwtemp, structure = strel, output = np.int32)
            props  = regionprops(bwlabel, cache = False)
            if verbose: print(performance_time(f"detecting {len(props)} objects", start))

            weights = np.ones((num_of_labels + 1,), dtype = np.float32)
            # bwtemp = 1 - bwtemp.astype(np.float32)
            bwtemp = np.ones(hipass.shape, dtype = np.float32)
            max_angle = sin(radians(params.max_angle))
            max_ratio = params.max_ratio
            # if verbose:
            #     props_array = np.recarray((len(props),), dtype = np.dtype([('area', np.int32), ('majorAxis', np.float32), ('minorAxis', np.float32),
            #                                                                 ('orientation', np.float32), ('bandness', np.float32)]))
            for prop_idx, prop in enumerate(props):
                majorAxis = prop.major_axis_length
                bandness  = 0
                # if majorAxis <= 0.10 * params.bands_len:
                    # bwtemp[prop.slice] = 1
                    # bandness = 0
                # else:
                if majorAxis > 0.10 * params.bands_len:
                    minorAxis = prop.minor_axis_length
                    angle = prop.orientation
                    if ndimB == -2: # vertical
                        bandness = max_angle * cos(angle) * minorAxis / max(majorAxis * max_ratio * (max_angle - abs(sin(angle))), EPS)
                    else:
                        bandness = max_angle * sin(angle) * minorAxis / max(majorAxis * max_ratio * (max_angle - cos(angle)), EPS)

                    weights[prop.label] = exp(-0.5 * (bandness ** 2))
                    # bwtemp[prop.slice] = bandness

                # if verbose:
                #     props_array[prop_idx].area = prop.area
                #     props_array[prop_idx].majorAxis = prop.major_axis_length
                #     props_array[prop_idx].minorAxis = prop.minor_axis_length
                #     props_array[prop_idx].orientation = degrees(prop.orientation)
                #     props_array[prop_idx].bandness = bandness

            bwtemp[bwlabel != 0] = weights[bwlabel[bwlabel != 0]]
            if verbose: print(performance_time(f"predict bands {scale_idx} total time", start))
            if params.isfield("mask_len"):
                bwtemp = AVERAGE(bwtemp, params.mask_len)
                bands  = bwtemp * CONV(hipass * bwtemp, kernA)
            else:
                bands  = CONV(hipass * bwtemp, kernA)

        output -= ((beta - 1) * hipass + bands) / beta if adjust_beta else bands
        if verbose: print(performance_time(f"scale {scale_idx} total time", start))

    return output

#===========================================================================================================================================
def find_flat_background(image, threshold = 0, min_area = None):
    '''
    find large flat 2d regions, which are most likely to be artificial background

    Parameters
    ----------
    image : 2d or 3d ndarray

    threshold : int or float scalar, optional. The default is 0.
        specifies the difference between maximal and minimal pixel (voxel) in 3x3 neighborhood to be considered as flat
    min_area : integer scalar, optional, the default is None.
        small regions below this parameter are removed from the output bitmap

    Returns
    -------
    bwflat : boolean ndarray of the same shape as image
    '''
    min_area = ceil(0.0001 * image.shape[-1] * image.shape[-2]) if min_area is None else min_area
    strel = conndef(2, minimal = False)
    if image.ndim == 3: strel = strel[np.newaxis, ...]
    bwflat = binary_dilation((maximum_filter(image, footprint = strel) - minimum_filter(image, footprint = strel)) <= threshold,
                             structure = strel, border_value = 0)
    labels, num_of_labels = label(bwflat, structure = strel)
    for label_idx in range(1, num_of_labels + 1):
        index, = np.nonzero((labels.flat == label_idx))
        if index.size <= min_area:
            bwflat.flat[index] = False
    return bwflat

#===========================================================================================================================================
def adjust_histogram_v_0_00(image, outfit = 0.01, drange = (0, 255), nbins = 1024):
    '''
    adjust_histogram_v_0_00 - simple histogram adjustment method that compresses histogram outfiters and preserves the internal part
        similar to "eliminate outfitters", except it doesn't assign the same value for values that are below the low and above the high percentile,
        but applies the histogram equalization for this values
    parameters:
        image : n-dimensional array to be processed

        outfit : float, optional. The default is 0.01.
            percentage of pixels / voxels to be compressed from the right and left side of histogram. The same percentage is applied if this
            value is scalar, otherwise it has to contain 2-element tuple
        drange : scalar or two-elements tuple, optional. The default is (0, 255)
            data range of output image. If this parameter is scalar otherwise it has to contain 2-element tuple,
            the 1st element is for the minimal value and the 2nd element is for maximal
        nbins : int, optional. The default is 1024.
            specifies number of bins to be used in histogram, if None or 0 - np.unique sill be used for the maximal accurate transformation

    Examples of usage:
        >>> adjusted = adjust_histogram_v_0_00(denoised) % use the default parameters
        or use custom parameters, compress 5% of left outfitters, ignore right one and normalize output values to the range from -1 to 1
        >>> adjusted = adjust_histogram_v_0_00(denoised, outfit = (0.05, 0), drange = (-1, 1))
    '''
    outfit = (outfit, outfit) if np.isscalar(outfit) else outfit
    drange = (0, drange) if np.isscalar(drange) else drange

    if not nbins:
        vals, valsIdx, cdf = np.unique(image, return_inverse = True, return_counts = True)
    else:
        vals = nbins * (1 - np.finfo(np.float32).eps) * (image - image.min()) / (image.max() - image.min())
        valsIdx = np.trunc(vals).astype(np.int64)
        cdf, vals = np.histogram(vals, bins = nbins, range = (0, nbins))

    cdf = np.cumsum(cdf.astype(np.float32)) / image.size
    preserve = np.nonzero((outfit[0] <= cdf) & (cdf <= (1 - outfit[1])))[0]
    leftIdx  = preserve[0]
    rightIdx = preserve[-1]

    lut = np.zeros(cdf.shape, dtype = cdf.dtype)
    lut[0:leftIdx] = cdf[0:leftIdx]
    lut[-1:rightIdx:-1] = cdf[-1:rightIdx:-1]
    lut[preserve] = cdf[leftIdx] + (cdf[rightIdx] - cdf[leftIdx]) / (vals[rightIdx] - vals[leftIdx]) * (vals[preserve] - vals[leftIdx])
    lut = (drange[1] - drange[0]) * (lut - lut[0]) / (lut[-1] - lut[0])

    return lut[valsIdx].reshape(image.shape)

#===========================================================================================================================================
def adjust_histogram_v_2_00(image, flatten = 0.25, blend = 0.0, clip_bins = 0, mean = None, drange = 255, std_size = 9, dilate_hist = 9, scales = 6, dtype = np.uint8):
    drange = (0, drange) if (drange is None) or np.isscalar(drange) else drange
    mean = (0.5 * (drange[1] - drange[0])) if mean is None else mean
    eps = np.finfo(np.float32).eps

    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(kernel.kernHxH.dtype)
    else:
        image = np.nan_to_num(image)

    kernA = kernel.kernHxH
    for scaleIdx in range(1, scales + 1):
        kernA = np.correlate(kernel().dilate(scaleIdx), kernA, mode = 'full')

    lopass = kernel(kernA).convn(image, ndims = 2)
    if (std_size is None) or std_size < 3:
        img_mean = np.mean(lopass)
        std_image = None
    else:
        kernA = kernel(np.ones((std_size,), dtype = np.float32) / std_size)
        std_image = kernA.convn(np.square(image)) - np.square(kernA.convn(image))
        std_image = np.sqrt(std_image * (std_image > 0))
        std_image = np.minimum(1, std_image / np.mean(std_image))
        img_mean  = np.sum(lopass * std_image) / np.sum(std_image)

    nimage = mean - flatten * img_mean + image - (1 - flatten) * lopass
    if blend >= 1:
        return nimage

    bins = np.concatenate(((-np.inf, ), np.arange(0.5, 254.5 + np.finfo(np.float32).eps, 0.25, dtype = np.float32), (np.inf, )), axis = -1)
    pdf  = np.histogram(nimage, bins, weights = std_image, density = False)[0]
    if (not dilate_hist is None) and (dilate_hist > 0):
        pdf = maximum_filter1d(pdf, size = dilate_hist, output = pdf, mode = 'constant', cval = 0)
    pdf /= np.sum(pdf)

    if clip_bins > 0:
        if clip_bins < 1:
            redist = (1 - clip_bins) / np.count_nonzero(pdf)
            pdf = clip_bins * pdf + redist * (pdf > 0)
        else:
            redist = np.sum(np.maximum(0, pdf - clip_bins / pdf.size)) / np.count_nonzero(pdf)
            pdf = pdf if redist <= eps else np.minimum(pdf, clip_bins / pdf.size) + redist * (pdf > 0)

    if np.sum(pdf) != 1:
        print(f"1 - sum(pdf) = {1 - np.sum(pdf)}")

    lut = drange[0] + (drange[1] - drange[0]) * (np.cumsum(pdf) - pdf[0]) / (1 - pdf[0])
    index = np.digitize(nimage, bins = bins) - 1
    return np.clip((1 - blend) * lut[index] + blend * nimage, drange[0], drange[1], dtype = dtype, casting = "unsafe")

#===========================================================================================================================================
def adjust_histogram_v_3_00(image, flatten = 0.10, mean = None, crop = (61, 199), drange = 255, std_size = 9, sigma_ratio = 2.0,
                            minmax_size = 9, method = "middle", ratio = 3, scales = 6, dtype = np.uint8,
                            adaptive = True, clip = None, redistribute = None, sigma = None):

    drange = (0, 255) if drange is None else (0, drange) if np.isscalar(drange) else drange
    mean   = (0.5 * (drange[1] - drange[0])) if mean is None else mean
    ratio  = 3.0 if ratio is None else ratio
    method = "middle" if method is None else method.lower()
    dtype  = image.dtype if dtype is None else dtype
    eps    = np.finfo(np.float32).eps

    sigma_ratio = 2.0 if sigma_ratio is None else sigma_ratio
    minmax_size = ([1] * (image.ndim - 2) + [minmax_size, minmax_size]) if np.isscalar(minmax_size) else minmax_size

    adaptive = False if adaptive is None else adaptive
    clip = clip if (clip is None) or (not np.isscalar(clip)) else None if clip == 0 else (-abs(clip), abs(clip))
    redistribute = None if clip is None else redistribute if not (redistribute is None) else 'full'
    scales = 6 if scales is None else max(6, scales)


    CONV = lambda x, kern: (kern if isinstance(kern, kernel) else kernel(kern)).convn(x, ndims = 2)
    # NORMCDF = lambda x, center, sigma: 1 - 0.5 * erfc((x - center) / (sqrt(2) * sigma))
    # VAR     = lambda x, kern: CONVN(np.square(x), kern) - np.square(CONVN(x, kern))

    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(kernel.kernHxH.dtype)
    else:
        image = np.nan_to_num(image)


    kernHxH = kernel.kernHxH
    for scaleIdx in (1, 2): kernHxH = np.correlate(kernel().dilate(scaleIdx), kernHxH, mode = 'full')
    actmap = np.sqrt(CONV(np.square(image - CONV(image, kernHxH)), np.ones((std_size,), dtype = np.float32) / std_size))
    if not crop is None:
        mask = (crop[0] <= image) & (image <= crop[1])
        actmap = actmap[mask]

    if ((not flatten is None) and (flatten < 1)) or adaptive:
        for scaleIdx in range(3, scales): kernHxH = np.correlate(kernel().dilate(scaleIdx), kernHxH, mode = 'full')

    if flatten == 0:
        image -= CONV(image, kernHxH) - mean
    elif (not flatten is None) and (flatten < 1):
        image -= (1 - flatten) * CONV(image, kernHxH) - (1 - flatten) * mean

    if crop is None:
        vals = image
    elif actmap.size >= 0.001 * image.size:
        vals = image[mask]; del mask
    else:
        return np.clip(image, drange[0], drange[1], dtype = dtype, casting = "unsafe")

    actmap = np.minimum(actmap / np.mean(actmap), 1, out = actmap)
    maxval = maximum_filter(image, size=minmax_size, mode="mirror")
    minval = minimum_filter(image, size=minmax_size, mode="mirror")
    middle = 0.5 * (maxval + minval)

    sum_amap = np.sum(actmap)
    center = np.sum(actmap * vals) / sum_amap
    sigma  = sigma if (not sigma is None) and (sigma > 0) else np.sqrt(np.sum(actmap * np.square(vals)) / sum_amap - (center**2))
    print(f"center = {center:.2f} sigma = {sigma:.2f}")

    multi  = np.maximum(ratio - ratio * norm.cdf(locals()[method], loc = center, scale = sigma), 1)
    if adaptive:
        lopass = CONV(image, kernHxH)
        sigma  = np.maximum(sigma, sigma_ratio * np.sqrt(CONV(np.square(image - lopass), kernHxH)))
    else:
        lopass = center
        sigma *= sigma_ratio

    image *= multi; image -= (multi - 1) * middle

    ncdf = norm.cdf(image, loc = lopass, scale = sigma)
    if not clip is None:
        cdf   = norm.cdf(clip); # left and right normal cdf for clipped boundaries
        pdf   = norm.pdf(clip); # left and right normal pdf for clipped boundaries
        delta = (cdf[1] - cdf[0]) - 0.5 * (pdf[0] + pdf[1]) * (clip[1] - clip[0])

        def midcdf():
            hipass  = (image - lopass) / sigma
            hipass  = np.clip(hipass - clip[0], 0, clip[1] - clip[0], out = hipass)
            return 0.5 * hipass * (2 * pdf[0] + (pdf[1] - pdf[0]) * hipass / (clip[1] - clip[0]))

        if redistribute.lower() == 'leftmost':
            mul  = delta / cdf[0] + 1
            ncdf = mul * np.minimum(ncdf, cdf[0]) + midcdf() + np.maximum(ncdf - cdf[1], 0)
        elif redistribute.lower() == 'left':
            mul  = delta / (cdf[0] + 0.5 * (pdf[0] + pdf[1]) * (clip[1] - clip[0])) + 1
            ncdf = mul * (np.minimum(ncdf, cdf[0]) + midcdf()) + np.maximum(ncdf - cdf[1], 0)
        else: # redistribute.lower() == 'full':
            mul  = 1 / (1 - delta)
            ncdf = mul * (np.minimum(ncdf, cdf[0]) + midcdf() + np.maximum(ncdf - cdf[1], 0))

    return np.clip(lopass + (drange[1] - drange[0]) * (ncdf - 0.5), drange[0], drange[1]).astype(dtype)
    # return (drange[0] + drange[1] * NORMCDF(multi * image - (multi - 1) * middle, center, sigma)).astype(dtype)

#===========================================================================================================================================
def adjust_histogram_v_3_01(image, mean = 132, drange = 255, sigma_ratio = 2.0,
                            minmax_size = 9, method = "middle", ratio = 3, dtype = np.uint8,
                            clip = None, redistribute = None, sigma = 10, save_memory = True):

    drange = (0, 255) if drange is None else (0, drange) if np.isscalar(drange) else drange
    sigma_ratio = 2.0 if sigma_ratio is None else sigma_ratio
    # minmax_size = ([1] * (image.ndim - 2) + [minmax_size, minmax_size]) if np.isscalar(minmax_size) else minmax_size
    ratio  = 3.0 if ratio is None else ratio
    method = "middle" if method is None else method.lower()
    dtype  = image.dtype if dtype is None else dtype
    eps    = np.finfo(np.float32).eps

    clip = clip if (clip is None) or (not np.isscalar(clip)) else None if clip == 0 else (-abs(clip), abs(clip))
    redistribute = None if clip is None else redistribute if not (redistribute is None) else 'full'

    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(kernel.kernHxH.dtype)
    else:
        image = np.nan_to_num(image)

    maxval = maximum_filter(image, size=minmax_size, mode="mirror")
    minval = minimum_filter(image, size=minmax_size, mode="mirror")
    middle = 0.5 * (maxval + minval)

    multi  = np.maximum(ratio - ratio * norm.cdf(locals()[method], loc = mean, scale = sigma), 1)

    if save_memory:
        del minval, maxval
        gc.collect()

    image *= multi; image -= (multi - 1) * middle
    sigma *= sigma_ratio

    if save_memory:
        del middle, multi
        gc.collect()

    ncdf = norm.cdf(image, loc = mean, scale = sigma)
    if not clip is None:
        cdf   = norm.cdf(clip); # left and right normal cdf for clipped boundaries
        pdf   = norm.pdf(clip); # left and right normal pdf for clipped boundaries
        delta = (cdf[1] - cdf[0]) - 0.5 * (pdf[0] + pdf[1]) * (clip[1] - clip[0])

        def midcdf():
            hipass  = (image - mean) / sigma
            hipass  = np.clip(hipass - clip[0], 0, clip[1] - clip[0], out = hipass)
            return 0.5 * hipass * (2 * pdf[0] + (pdf[1] - pdf[0]) * hipass / (clip[1] - clip[0]))

        if redistribute.lower() == 'leftmost':
            mul  = delta / cdf[0] + 1
            ncdf = mul * np.minimum(ncdf, cdf[0]) + midcdf() + np.maximum(ncdf - cdf[1], 0)
        elif redistribute.lower() == 'left':
            mul  = delta / (cdf[0] + 0.5 * (pdf[0] + pdf[1]) * (clip[1] - clip[0])) + 1
            ncdf = mul * (np.minimum(ncdf, cdf[0]) + midcdf()) + np.maximum(ncdf - cdf[1], 0)
        else: # redistribute.lower() == 'full':
            mul  = 1 / (1 - delta)
            ncdf = mul * (np.minimum(ncdf, cdf[0]) + midcdf() + np.maximum(ncdf - cdf[1], 0))

    return np.clip(mean + (drange[1] - drange[0]) * (ncdf - 0.5), drange[0], drange[1]).astype(dtype)

#===========================================================================================================================================
def adjust_histogram_v_3_02(image, drange = 255, dtype = np.uint8, bwmask = None, smoothing = 501,
                            method = "image", sigma = 7.5, offset = -15, ratio = 2, minmax_size = 9,
                            compress_range = None, compress_ratio = 0.1, clip = [-0.8, 0.8],
                            redistribute = "left", hp_sigma = 15, mean = None, save_memory = True):
    '''
    adjust_histogram_v_3_02 implements histogram adjustment and enhancement method introduced by Matlab function EmdLib.HistogramAdjustment_v_3_02

    Parameters
    ----------
    image : 2 or 3 dimensional ndarray
        the processing is done in xy-plane, therefore the entire section or the large patch of the section should be submitted
        NOTE: if image contains floating point values and bwmask is not None, values of image[~bwmask] are changed by this function to middle of drange,
            no copy is made in order to reduce memory consumption. The caller should save and restore the original values under image[~bwmask] if they
            have meaning for further processing
    drange : scalar or two-elements tuple, optional. The default is (0, 255)
        data range of output image. If this parameter is scalar otherwise it has to contain 2-element tuple,
        the 1st element is for the minimal value and the 2nd element is for maximal
    dtype : numpy dtype, optional
        specifies data-type to convert the function output. The default is np.uint8.
    bwmask: boolean ndarray of the same shape as image, optional
        provides binary mask to mark ROI pixels. The default is None, which states for entire image to be included
    smoothing: integer, optional
        length of smoothing filter used to retrieve lopass component of the image, default is 501
    method : string, optional
        specifies method to be used for auto-level enhancement, the default is "image".
    sigma : float, optional
        used in auto-level step of enhancement. the default is 7.5.
    offset : float, optional
        used in auto-level step of enhancement. the default is -15.
    ratio : float, optional
        specifies enancement magnitude applied by auto-level step, the default is 2.
    minmax_size : integer, optional
        size of structure element used for morhological dilation and erosion. The default is 9.
    compress_range : list on np.array of 2 values, optional
        specifies when the specific range of lopass component has to be compressed. the default is None.
    compress_ratio : float value in range [0..1], optional
        specifies the compression ratio to be applied to the lopass pixels in range provided by <compress_range> parameter. the default is 0.1.
    clip : scalar or array-like variable of 2, optional
        values specifying the left and right limits for histogram clippling, the default is [-0.8, 0.8].
    redistribute : string, optional
        specifies how the clipped part of histogram is redestributed between bins. Possible values are 'leftmost', 'left' and 'full' the default is "left".
        'leftmost' makes the clipped area to be redistributed to the values below clip[0]
        'left' makes the clipped area to be redistributed to the values upto clip[1]
        'full' uniformly redistributes clipped area between the entire range
    hp_sigma : numeric, optional
        specifies standard deviation to be used for clip-limited normal distributed histogram adjustment
    mean : numeric, optional. The default is None
        specifies mean value to be used in order to normalize image, as it centred at mean and has standard deviation of hp_sigma
    save_memory : boolean, optional
        deletes the unused variables and invokes garbage collector in order to reduce memory footprint, the default is True.

    Returns
    -------
    ndarray of type <dtype>
        the processed image

    '''
    eps    = np.finfo(np.float32).eps
    drange = (0, 255) if drange is None else (0, drange) if np.isscalar(drange) else drange
    mean   = np.mean(drange) if mean is None else mean
    method = "image" if method is None else method.lower()
    dtype  = image.dtype if dtype is None else dtype
    clip = clip if (clip is None) or (not np.isscalar(clip)) else None if clip == 0 else (-abs(clip), abs(clip))
    redistribute = None if clip is None else redistribute if not (redistribute is None) else 'full'

    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32)
    elif not bwmask is None:
        bwmask = bwmask & ~np.isnan(image)
    else:
        bwmask = ~np.isnan(image)

    if (bwmask is None) or np.all(bwmask):
        bwmask = None # we later multiply some results by bwmask
    else:
        image[~bwmask] = mean

    if (not ratio is None) and (ratio > 0):
        kernA = np.ones((smoothing,), dtype = np.float32) / np.float32(smoothing)
        kernA = kernel(np.correlate(kernA, kernA, mode = 'full'))
        if bwmask is None:
            lopass = kernA.convn(image, ndims = 2)
            hipass = image - lopass
            MULTI  = lambda vals, mean: ratio * (1 - norm.cdf(vals, loc = mean + offset, scale = sigma))
        else:
            bwmask = bwmask.astype(np.float32)
            hipass = bwmask * (image - kernA.convn(image * bwmask, ndims = 2) / (kernA.convn(bwmask, ndims = 2) + eps))
            lopass = image - hipass
            MULTI  = lambda vals, mean: ratio * bwmask * (1 - norm.cdf(vals, loc = mean + offset, scale = sigma))

        if method != "image":
            maxval = maximum_filter(image, size = minmax_size, mode = "mirror")
            minval = minimum_filter(image, size = minmax_size, mode = "mirror")
            middle = 0.5 * (maxval + minval)

        if compress_range is None:
            multi = MULTI(locals()[method], lopass)
            if method == "image":
                middle = 0.5 * (maximum_filter(image, size = minmax_size, mode = "mirror") + minimum_filter(image, size = minmax_size, mode = "mirror"))
        else:
            multi  = MULTI(locals()[method], np.maximum(compress_range[0], np.minimum(compress_range[1], lopass)))
            lopass = 0.5 * (1 - compress_ratio) * (compress_range[1] - compress_range[0]) + np.minimum(lopass, compress_range[0]) + np.maximum(0, lopass - compress_range[1]) \
                + compress_ratio * np.minimum(compress_range[1] - compress_range[0], np.maximum(0, lopass - compress_range[0]))
            image  = lopass + hipass
            middle = 0.5 * (maximum_filter(image, size = minmax_size, mode = "mirror") + minimum_filter(image, size = minmax_size, mode = "mirror"))

        if save_memory:
            if method != "image":
                del minval, maxval
            gc.collect()

        image  = image + multi * (image - middle)

    hipass = (image - mean) / hp_sigma
    ncdf = norm.cdf(hipass, 0, 1)
    # if save_memory: gc.collect()
    if not clip is None:
        cdf = norm.cdf(clip); # left and right normal cdf for clipped boundaries
        pdf = norm.pdf(clip); # left and right normal pdf for clipped boundaries

        delta   = (cdf[1] - cdf[0]) - 0.5 * (pdf[0] + pdf[1]) * (clip[1] - clip[0])
        hipass  = np.clip(hipass - clip[0], 0, clip[1] - clip[0], out = hipass)
        hipass *= 0.5 * (2 * pdf[0] + (pdf[1] - pdf[0]) * hipass / (clip[1] - clip[0]))

        if redistribute.lower() == 'leftmost':
            mul  = delta / cdf[0] + 1
            ncdf = mul * np.minimum(ncdf, cdf[0]) + hipass + np.maximum(ncdf - cdf[1], 0)
        elif redistribute.lower() == 'left':
            mul  = delta / (cdf[0] + 0.5 * (pdf[0] + pdf[1]) * (clip[1] - clip[0])) + 1
            ncdf = mul * (np.minimum(ncdf, cdf[0]) + hipass) + np.maximum(ncdf - cdf[1], 0)
        else: # redistribute.lower() == 'full':
            mul  = 1 / (1 - delta)
            ncdf = mul * (np.minimum(ncdf, cdf[0]) + hipass + np.maximum(ncdf - cdf[1], 0))

    # return np.clip(middle + (drange[1] - drange[0]) * (ncdf - 0.5), drange[0], drange[1]).astype(dtype)
    return np.clip(drange[0] + (drange[1] - drange[0] + 1) * ncdf, drange[0], drange[1]).astype(dtype, copy = False);


#==================================================================================================================================
# def denoise_nlm_xcorr(sections, template, reference = None, clip = None, store_section = False, BLOCK = 7, NBHOOD = 11, WEIGHT_COUNT = 10, print_fn = print):
def denoise_nlm_xcorr(sections, template, reference = None, source = None, clip = None, section_weight = None, source_weight = 1.0,
                      adjust_xcorr = False, BLOCK = 7, NBHOOD = 11, WEIGHT_COUNT = 10, print_fn = print):
    """
    applies nlm-like denoising based on cross-correlation similiarity to the large image

    Parameters
    ----------
    sections : numpy.ndarray
        3d stack of images contains pixels used for denoising. The stack axes are YXZ, so Z can be treated by opencv filtering
        function as channel
    template : numpy.ndarray (of float32)
        2d image used for similarity check vs reference sections
    reference: numpy.ndarry (otional)
        3d stack of images used for similarity estimation vs template image. <sections> parameter is used in case of
        <reference> is None
    source : numpy.ndarray, optional
        source image, used as unconditional part of averaging with a weight specified by source_weight parameter
    clip: float or None
        if this parameter is specified, clips similarity score not to exceed the specified value: 0 < clip < 1.0
    **kwargs :
        named arguments to be passed to calc_xcorr_weights function
    Returns
    -------
    two dimensional numpy.ndarray of the same shape as two first dimensions of template
    """
    EPS = np.finfo(np.float32).eps

    # pixels contains sections in [y, x, z] order
    _time_it = lambda message: nullcontext() if print_fn is None else time_it(message, print_fn = print_fn)
    # _conv2d  = lambda x, kern: cv.filter2D(cv.filter2D(x, ddepth = -1, kernel = kern.reshape(1, -1)), ddepth = -1, kernel = kern.reshape(-1, 1))

    IMAGE_CY = sections.shape[0]
    IMAGE_CX = sections.shape[1]
    SECTIONS = sections.shape[2]
    NBHOOD_MARGIN = (NBHOOD - 1) // 2

    def update_weights(section_idx, dx = 0, dy = 0):
        nonlocal min_index

        if (not section_weight is None) and (abs(section_weight[section_idx]) <= EPS):
            return

        x = struct(templ = slice( max(dx, 0), IMAGE_CX + min(0, dx)), image = slice(-min(dx, 0), IMAGE_CX - max(0, dx)))
        y = struct(templ = slice( max(dy, 0), IMAGE_CY + min(0, dy)), image = slice(-min(dy, 0), IMAGE_CY - max(0, dy)))

        index = struct(y = data.y[y.templ, x.templ], x = data.x[y.templ, x.templ], min = min_index[y.templ, x.templ])

        # data.mean = mean2d(data.image[y.image, x.image, section_idx], BLOCK)
        # data.std  = safe_sqrt(mean2d(np.square(data.image[y.image, x.image, section_idx]), BLOCK) - np.square(data.mean))
        # templ.mean = mean2d(templ.image[y.templ, x.templ], BLOCK)
        # templ.std  = safe_sqrt(mean2d(np.square(templ.image[y.templ, x.templ]), BLOCK) - np.square(templ.mean))

        # xcorr = (mean2d(data.image[y.image, x.image, section_idx] * templ.image[y.templ, x.templ], BLOCK) - data.mean * templ.mean) / (data.std * templ.std + EPS)
        # bwmap = xcorr > (nlm.weight[index.y, index.x, index.min] + np.finfo(np.float32).resolution)

        xcorr = (mean2d(data.image[y.image, x.image, section_idx] * templ.image[y.templ, x.templ], BLOCK) \
                - data.mean[y.image, x.image, section_idx] * templ.mean[y.templ, x.templ]) / (data.std[y.image, x.image, section_idx] * templ.std[y.templ, x.templ] + EPS)
        # xcorr[xcorr > 0] = np.square(xcorr[xcorr > 0])
        # xcorr = safe_sqrt(xcorr)
        if adjust_xcorr: xcorr *= section_weight[section_idx]
        bwmap = (xcorr > (nlm.weight[index.y, index.x, index.min] + np.finfo(np.float32).resolution)) & (np.fabs(xcorr) <= 1.0)

        index.x = index.x[bwmap]; index.y = index.y[bwmap]; index.min = index.min[bwmap]
        nlm.weight[index.y, index.x, index.min] = xcorr[bwmap]
        # nlm.pixels[index.y, index.x, index.min] = sections[y.templ, x.templ, section_idx][bwmap]
        nlm.pixels[index.y, index.x, index.min] = sections[y.image, x.image, section_idx][bwmap]
        if (not section_weight is None) and (not adjust_xcorr): nlm.section[index.y, index.x, index.min] = section_idx
        min_index = np.argmin(nlm.weight, axis = -1)
        if print_fn:
            print_fn(f"{np.count_nonzero(bwmap)} pixels were stored")
        return

    data  = struct(image = sections if reference is None else reference)
    templ = struct(image = template)

    with _time_it("prepare correlation data"):
        data.mean = mean2d(data.image, BLOCK)
        data.std  = safe_sqrt(mean2d(np.square(data.image), BLOCK) - np.square(data.mean))
        templ.mean = mean2d(templ.image, BLOCK)
        templ.std  = safe_sqrt(mean2d(np.square(templ.image), BLOCK) - np.square(templ.mean))

    nlm = struct(pixels = np.zeros((*template.shape, WEIGHT_COUNT), dtype = np.float32),
                 weight = np.zeros((*template.shape, WEIGHT_COUNT), dtype = np.float32))

    if section_weight is None:
        adjust_xcorr = False
    elif not adjust_xcorr:
        nlm.section = np.zeros((*template.shape, WEIGHT_COUNT), dtype = np.int16)
        section_weight = np.array(section_weight, dtype = np.float32)

    data.y, data.x = np.indices((IMAGE_CY, IMAGE_CX), dtype = np.int16)
    if source is None or not source_weight:
        min_index = np.zeros((IMAGE_CY, IMAGE_CX), dtype = np.int16)
    else:
        nlm.pixels[..., 0] = source
        nlm.weight[..., 0] = source_weight
        min_index = np.ones((IMAGE_CY, IMAGE_CX), dtype = np.int16)
        if (not section_weight is None) and (not adjust_xcorr):
            nlm.section[..., 0] = -1

    for dx in range(0, NBHOOD_MARGIN + 1):
        for dy in range(0, NBHOOD_MARGIN + 1):
            for section_idx in range(SECTIONS):
                with _time_it(f"process {section_idx = } {dx = } {dy = }"):
                    update_weights(section_idx,  dx,  dy)
                    if dx != 0 or dy != 0:
                        update_weights(section_idx, -dx,  dy)
                        update_weights(section_idx,  dx, -dy)
                        update_weights(section_idx, -dx, -dy)


    if clip: nlm.weight[nlm.weight > clip] = clip
    if (not section_weight is None) and (not adjust_xcorr):
        if source is None:
            nlm.weight *= section_weight[nlm.section]
        else:
            bwmap = nlm.section >= 0
            nlm.weight[bwmap] *= section_weight[nlm.section[bwmap]]

        # output = np.zeros(template.shape, dtype = np.float32)
        # for section_idx in range(len(section_weight)):
        #     if section_weight[section_idx] != 0:
        #         bwmap = (nlm.section == section_idx).astype(np.float32)
        #         output += section_weight[section_idx] * np.sum(nlm.pixels * nlm.weight * bwmap, axis = -1) / (np.sum(nlm.weight * bwmap, axis = -1) + EPS)
        # return output

    return np.sum(nlm.pixels * nlm.weight, axis = -1) / (np.sum(nlm.weight, axis = -1) + EPS)

#==================================================================================================================================
def denoise_nlm_patches(sections, template, reference = None, output = None, patch_shape = (1024, 1024), overlap = (64, 64), print_fn = print, **kwargs):
    """
    applies nlm-like denoising based on cross-correlation similiarity to the large image, splitting it to smaller overlapping sections

    Parameters
    ----------
    sections : numpy.ndarray
        3d stack of images contains pixels used for denoising. The stack axes are YXZ, so Z can be treated by opencv filtering
        function as channel
    template : numpy.ndarray (of float32)
        2d image used for similarity check vs reference sections
    reference: numpy.ndarry (otional)
        3d stack of images used for similarity estimation vs template image. <sections> parameter is used in case of
        <reference> is None
    patch_shape : tuple of 2 integer values or scalar. The default is (1024, 1024).
        specifies the size of the patch.
    overlap : tuple of 2 integer values or scalar. The default is (64, 64).
        specifies size of patches overlapping region
    **kwargs :
        named arguments to be passed to calc_xcorr_weights function
    Returns
    -------
    two dimensional numpy.ndarray of the same shape as two first dimensions of template
    """

    _time_it = lambda message: nullcontext() if print_fn is None else time_it(message, print_fn = print_fn)

    def create_slices(image_start, patch_cx, overlap, image_cx):
        """
        creates 2 slices used to copy information from patch to image
        """
        image_end   = image_start + patch_cx
        patch_start = None
        patch_end   = None
        if image_start > 0:
            image_start += overlap // 2
            patch_start  = overlap // 2
        if image_end > image_cx:
            patch_end = patch_cx - (image_end - image_cx)
            image_end = None
        return slice(image_start, image_end), slice(patch_start, patch_end)

    if np.isscalar(patch_shape): patch_shape = (patch_shape, patch_shape)
    if np.isscalar(overlap): overlap = (overlap, overlap)

    CY = template.shape[0]
    CX = template.shape[1]

    if output is None: output = np.zeros(template.shape, dtype = np.float32)

    STRIDE = (patch_shape[0] - overlap[0], patch_shape[1] - overlap[1])
    grid = struct(y = np.arange(0, CY, STRIDE[0], dtype = int), x = np.arange(0, CX, STRIDE[1], dtype = int))

    for y in grid.y:
        image_y, patch_y = create_slices(y, patch_shape[0], overlap[0], CY)
        slice_y = slice(y, min(CY, y + patch_shape[0]))
        for x in grid.x:
            slice_x = slice(x, min(CX, x + patch_shape[1]))
            with _time_it(f"denoise patch {x = :05d} {y = :05d}"):
                patch = denoise_nlm_xcorr(sections  = sections[slice_y, slice_x], template = template[slice_y, slice_x],
                                          reference = None if reference is None else reference[slice_y, slice_x],
                                          print_fn = None, **kwargs)
                image_x, patch_x = create_slices(x, patch_shape[1], overlap[1], CX)
                output[image_y, image_x] = patch[patch_y, patch_x]

    return output

#==================================================================================================================================
def process_patches(process, inputs : dict, output = None, shape = None, patch_shape = (1024, 1024), overlap = (64, 64), print_fn = print, **kwargs):
    """
    applies nlm-like denoising based on cross-correlation similiarity to the large image, splitting it to smaller overlapping sections

    Parameters
    ----------
    process : function
        processing function to be applied for each patch
    inputs: dict or numpy.ndarray
        dictionary containing input images to be splitted to patches passed to processing function
    output : np.ndarray, optional
        array to be used for storing results. If output is None then this array is created inside the function
        with shape specified by <shape> parameter
    shape : tuple, optional
        the shape of the <output>. This parameter is ignored if <output> is provided. If no <output> is provided and
        <shape> is None then inputs must be np.ndarray (not dictionary) and output has the same shape and type as input
    patch_shape : tuple of 2 integer values or scalar. The default is (1024, 1024).
        specifies the size of the patch.
    overlap : tuple of 2 integer values or scalar. The default is (64, 64).
        specifies size of patches overlapping region
    **kwargs :
        named arguments to be passed to calc_xcorr_weights function
    Returns
    -------
    two dimensional numpy.ndarray of the same shape as two first dimensions of template
    """

    _time_it = lambda message: nullcontext() if print_fn is None else time_it(message, print_fn = print_fn)

    def create_slices(image_start, patch_cx, overlap, image_cx):
        """
        creates 2 slices used to copy information from patch to image
        """
        image_end   = image_start + patch_cx
        patch_start = None
        patch_end   = None
        if image_start > 0:
            image_start += overlap // 2
            patch_start  = overlap // 2
        if image_end > image_cx:
            patch_end = patch_cx - (image_end - image_cx)
            image_end = None
        return slice(image_start, image_end), slice(patch_start, patch_end)

    if np.isscalar(patch_shape): patch_shape = (patch_shape, patch_shape)
    if np.isscalar(overlap): overlap = (overlap, overlap)

    if not output is None:
        shape = output.shape
    elif shape is None:
        shape = inputs.shape # inputs must be np.ndarray
        output = np.zeros(shape, dtype = inputs.dtype)
    else:
        output = np.zeros(shape, dtype = np.float32)

    CY = shape[0]
    CX = shape[1]

    STRIDE = (patch_shape[0] - overlap[0], patch_shape[1] - overlap[1])
    grid = struct(y = np.arange(0, CY, STRIDE[0], dtype = int), x = np.arange(0, CX, STRIDE[1], dtype = int))

    for y in grid.y:
        image_y, patch_y = create_slices(y, patch_shape[0], overlap[0], CY)
        slice_y = slice(y, min(CY, y + patch_shape[0]))
        for x in grid.x:
            slice_x = slice(x, min(CX, x + patch_shape[1]))
            with _time_it(f"process patch {x = :05d} {y = :05d}"):
                image_x, patch_x = create_slices(x, patch_shape[1], overlap[1], CX)
                if isinstance(inputs, dict):
                    # patch = dict()
                    # for key, val in inputs.items():
                    #     patch[key] = val[slice_y, slice_x]
                    patch = { key : val[slice_y, slice_x] for key, val in inputs.items() if not val is None }
                    output[image_y, image_x] = process(**patch, **kwargs)
                else:
                    output[image_y, image_x] = process(inputs[slice_y, slice_x], **kwargs)

    return output

#===========================================================================================================================================
def milling_correction(source, residual = None, sigma = 0.25, nbhood = 97, method = "xcorr"):
    if residual is None: residual = source

    EPS = np.finfo(np.float32).eps

    GAUSSIAN = lambda x, sigma: np.exp(-0.5 * np.square(x / sigma))
    # SQRT     = lambda x: np.sqrt(np.clip(x, a_min = 0))
    MEAN2D   = lambda x: mean2d(x, nbhood)
    # VAR2D    = lambda x: np.clip(MEAN2D(np.square(x)) - np.square(MEAN2D(x)), 0)
    # COV2D    = lambda x, y: MEAN2D(x * y) - MEAN2D(x) * MEAN2D(y)

    data = struct(mean = MEAN2D(residual))
    data.variance = np.clip(MEAN2D(np.square(residual)) - np.square(data.mean), a_min = 0.0, a_max = None)
    # data.before = MEAN2D(residual[..., 1:-1] * residual[...,  :1]) - data.mean[..., 1:-1] * data.mean[...,  :1]
    # data.after  = MEAN2D(residual[..., 1:-1] * residual[..., -1:]) - data.mean[..., 1:-1] * data.mean[..., -1:]
    data.before = MEAN2D(residual * residual[...,  :1]) - data.mean * data.mean[...,  :1]
    data.after  = MEAN2D(residual * residual[..., -1:]) - data.mean * data.mean[..., -1:]

    # if method.lower() == "var":
    #     diff = np.log(np.clip(data.variance[..., 1:-1] + data.variance[..., -1:] - 2 * data.after,  a_min = 0, a_max = None) \
    #                 / np.clip(data.variance[..., 1:-1] + data.variance[...,  :1] - 2 * data.before, a_min = EPS, a_max = None))
    # else:
    #     if method.lower() == "xcorr":
    #         data.before /= np.sqrt(data.variance[..., 1:-1] * data.variance[...,  :1]) + EPS
    #         data.after  /= np.sqrt(data.variance[..., 1:-1] * data.variance[..., -1:]) + EPS

    if method.lower() == "var":
        diff = np.log(np.clip(data.variance + data.variance[..., -1:] - 2 * data.after,  a_min = 0, a_max = None) \
                    / np.clip(data.variance + data.variance[...,  :1] - 2 * data.before, a_min = EPS, a_max = None))
    else:
        if method.lower() == "xcorr":
            data.before /= np.sqrt(data.variance * data.variance[...,  :1]) + EPS
            data.after  /= np.sqrt(data.variance * data.variance[..., -1:]) + EPS
        diff = data.before - data.after # for "xcorr" and "covar"
    # end if method.lower() == "var"
    del data

    center = (diff.shape[-1] - 1) / 2.0
    z = np.zeros(diff.shape[0:-1], dtype = np.int16)
    for idx in range(0, diff.shape[-1] - 1):
        z[(diff[..., idx] >= 0) & (diff[..., idx+1] < 0) & (np.abs(idx - center) < np.abs(z - center))] = idx

    y, x = np.indices(diff.shape[:-1], dtype = np.int16)

    middle = np.expand_dims(z + np.reciprocal(1.0 - diff[y, x, z + 1] / (diff[y, x, z] + EPS), dtype = np.float32), axis = -1)
    weight = GAUSSIAN(np.arange(0, diff.shape[-1], dtype = np.float32).reshape((1, 1, -1)) - middle, sigma * np.sqrt(1 + np.abs(middle - center)))
    # weight = GAUSSIAN(np.arange(0, diff.shape[-1], dtype = np.float32).reshape((1, 1, -1)) - middle, sigma)
    # weight = GAUSSIAN(middle - np.arange(0, diff.shape[-1], dtype = np.float32).reshape((1, 1, -1)), sigma)
    return np.sum(weight * source, axis = -1) / np.sum(weight, axis = -1)


#===========================================================================================================================================
def rescale_drange(image, drange = 255):
    drange = (0, drange) if np.isscalar(drange) else drange
    return (drange[1] - drange[0]) * (image - image.min()) / (image.max() - image.min())
