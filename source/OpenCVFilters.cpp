#include <iostream>
#include <opencv2/opencv.hpp>

/// <summary>
/// Applies the specified filter methods on an image located at the specified file path.
/// </summary>
/// <param name="imagePath">The file path of the to be filtered image</param>
/// <param name="outputDir">The output directory path of the to be filtered image after saving</param>
/// <param name="showImage">Wether to show the resulting image or not</param>
/// <param name="saveImage">Wether to save the resulting image or not</param>
/// <param name="doHSV">Wether to apply a HSV filter on the image or not</param>
/// <param name="doGrayscale">Wether to apply a grayscale filter on the image or not</param>
/// <param name="doEmboss">Wether to apply an embossing filter on the image or not</param>
static void OpenCVFilters(
    const std::string& imagePath,
    const std::string& outputDir,
    bool showImage = false,
    bool saveImage = false,
    bool doHSV = true,
    bool doGrayscale = true,
    bool doEmboss = true
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
    if (doHSV) {
        cv::cvtColor(image, image, cv::COLOR_RGB2HSV);
    }

    if (doGrayscale) {
        cv::cvtColor(image, image, cv::COLOR_RGB2GRAY);
    }

    if (doEmboss) {
        cv::Mat kernel = (cv::Mat_<float>(3, 3) << 
            -1, 0, 0, 
             0, 0, 0, 
             0, 0, 1
        );
        cv::filter2D(image, image, -1, kernel);
        image.convertTo(image, -1, 1, 128);
    }

    if (showImage) {
        cv::imshow("Final Image OpenCVFilters", image);
    }

    if (saveImage) {
        cv::imwrite(outputDir + "/resulting_image_opencv.png", image);
    }
}
