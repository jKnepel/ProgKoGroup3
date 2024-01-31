#pragma once

#include <opencv2/opencv.hpp>
#include <omp.h>

class ImageFilter
{
public:
    /// <summary>
    /// Calculate a grayscale using weighted channels based on the perceived luminosity (0.21 R + 0.72 G + 0.07 B)
    /// https://do-marlay-ka-moonh.medium.com/converting-color-images-to-grayscale-ab0120ea2c1e
    /// </summary>
    /// <param name="image"></param>
    /// <param name="useOpenMP"></param>
    static void GrayscaleImage(cv::Mat& image, bool useOpenMP = true);

    /// <summary>
    /// Calculate a grayscale using weighted channels based on the perceived luminosity (0.21 R + 0.72 G + 0.07 B)
    /// https://do-marlay-ka-moonh.medium.com/converting-color-images-to-grayscale-ab0120ea2c1e
    /// This grayscale filter is done within a single collapsed loop.
    /// </summary>
    /// <param name="image"></param>
    /// <param name="useOpenMP"></param>
    static void GrayscaleImageCollapsed(cv::Mat& image, bool useOpenMP = true);

    /// <summary>
    /// Turn an RGB colorspace image to HSV colorspace
    /// https://en.wikipedia.org/wiki/HSL_and_HSV#From_RGB
    /// </summary>
    /// <param name="image"></param>
    /// <param name="useOpenMP"></param>
    static void HSVImage(cv::Mat& image, bool useOpenMP = true);

    /// <summary>
    /// Turn an RGB colorspace image to HSV colorspace
    /// https://en.wikipedia.org/wiki/HSL_and_HSV#From_RGB
    /// This hsv filter is done within a single collapsed loop.
    /// </summary>
    /// <param name="image"></param>
    /// <param name="useOpenMP"></param>
    static void HSVImageCollapsed(cv::Mat& image, bool useOpenMP = true);

    /// <summary>
    /// Emboss an RGB colorspace image using a comparison of neighboring pixels
    /// </summary>
    /// <param name="image"></param>
    /// <param name="useOpenMP"></param>
    static void EmbossImage(cv::Mat& image, bool useOpenMP = true);

    /// <summary>
    /// Emboss an RGB colorspace image using a comparison of neighboring pixels
    /// This embossing filter is done within a single collapsed loop.
    /// </summary>
    /// <param name="image"></param>
    /// <param name="useOpenMP"></param>
    static void EmbossImageCollapsed(cv::Mat& image, bool useOpenMP = true);
};
