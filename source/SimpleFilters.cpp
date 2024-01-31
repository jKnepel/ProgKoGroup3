#include <iostream>
#include <opencv2/opencv.hpp>

/// <summary>
/// Applies the specified filter methods on an image located at the specified file path.
/// </summary>
/// <param name="imagePath">The file path of the to be filtered image</param>
/// <param name="outputDir">The output directory path of the to be filtered image after saving</param>
/// <param name="filterMethods">The image filter methods that should be run</param>
/// <param name="useOpenMP">Wether to use OpenMP for the image filter or not</param>
/// <param name="showImage">Wether to show the resulting image or not</param>
/// <param name="saveImage">Wether to save the resulting image or not</param>
static void SimpleFilters(
    const std::string& imagePath,
    const std::string& outputDir,
    const std::vector<std::function<void(cv::Mat&, bool)>>& filterMethods,
    bool useOpenMP = true,
    bool showImage = false,
    bool saveImage = false
) {
    cv::Mat image = cv::imread(imagePath);

    if (image.empty())
        throw std::invalid_argument("Could not open or find the image!");

#ifdef _DEBUG
    // NOTE : only print image values to console in debug mode, since it will skew the benchmark result
    std::cout << "Image width: " << image.cols << std::endl;
    std::cout << "Image height: " << image.rows << std::endl;
    std::cout << "Image pixels: " << image.cols * image.rows << std::endl;
#endif

    // apply the filters
    for (const auto& filter : filterMethods) {
        filter(image, useOpenMP);
    }

    if (showImage) {
        cv::imshow("Final Image SimpleFilters", image);
    }

    if (saveImage) {
        cv::imwrite(outputDir + "/resulting_image_simple.png", image);
    }
}
