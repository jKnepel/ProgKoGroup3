#include "ImageFilter.h"

void ImageFilter::GrayscaleImage(cv::Mat& image, bool useOpenMP)
{
    #pragma omp parallel for schedule(dynamic) if(useOpenMP)
    for (int x = 0; x < image.rows; x++) {
        for (int y = 0; y < image.cols; y++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(x, y);
            uchar gray = static_cast<uchar>(0.21 * pixel[2] + 0.72 * pixel[1] + 0.07 * pixel[0]);
            image.at<cv::Vec3b>(x, y) = cv::Vec3b(gray, gray, gray);
        }
    }
}

void ImageFilter::GrayscaleImageCollapsed(cv::Mat& image, bool useOpenMP)
{
    #pragma omp parallel for schedule(static) if(useOpenMP)
    for (int xy = 0; xy < image.rows * image.cols; xy++) {
        int x = xy / image.cols;
        int y = xy % image.cols;
        cv::Vec3b pixel = image.at<cv::Vec3b>(x, y);
        uchar gray = static_cast<uchar>(0.21 * pixel[2] + 0.72 * pixel[1] + 0.07 * pixel[0]);
        image.at<cv::Vec3b>(x, y) = cv::Vec3b(gray, gray, gray);
    }
}

void ImageFilter::HSVImage(cv::Mat& image, bool useOpenMP)
{
    #pragma omp parallel for schedule(dynamic) if(useOpenMP)
    for (int x = 0; x < image.rows; ++x) {
        for (int y = 0; y < image.cols; ++y) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(x, y);

            float r = pixel[2] / 255.0f;
            float g = pixel[1] / 255.0f;
            float b = pixel[0] / 255.0f;

            float cmax = fmax(fmax(r, g), b);
            float cmin = fmin(fmin(r, g), b);
            float delta = cmax - cmin;

            float hue = 0.0;
            if (delta != 0.0) {
                if (cmax == r)
                    hue = 60 * fmod((g - b) / delta, 6.0f);
                else if (cmax == g)
                    hue = 60 * (((b - r) / delta) + 2.0f);
                else if (cmax == b)
                    hue = 60 * (((r - g) / delta) + 4.0f);
            }
            if (hue < 0) hue += 360;
            float saturation = (cmax == 0.0f) ? 0 : (delta / cmax);
            float value = cmax;

            // multiply saturation and value by 255 to switch from normalized HSV space to pseudo-RGB space
            image.at<cv::Vec3b>(x, y) = cv::Vec3b(
                static_cast<uchar>(hue),
                static_cast<uchar>(saturation * 255),
                static_cast<uchar>(value * 255));
        }
    }
}

void ImageFilter::HSVImageCollapsed(cv::Mat& image, bool useOpenMP)
{
    #pragma omp parallel for schedule(static) if(useOpenMP)
    for (int xy = 0; xy < image.rows * image.cols; xy++) {
        int x = xy / image.cols;
        int y = xy % image.cols;
        cv::Vec3b pixel = image.at<cv::Vec3b>(x, y);

        float r = pixel[2] / 255.0f;
        float g = pixel[1] / 255.0f;
        float b = pixel[0] / 255.0f;

        float cmax = fmax(fmax(r, g), b);
        float cmin = fmin(fmin(r, g), b);
        float delta = cmax - cmin;

        float hue = 0.0;
        if (delta != 0.0) {
            if (cmax == r)
                hue = 60 * fmod((g - b) / delta, 6.0f);
            else if (cmax == g)
                hue = 60 * (((b - r) / delta) + 2.0f);
            else if (cmax == b)
                hue = 60 * (((r - g) / delta) + 4.0f);
        }
        if (hue < 0) hue += 360;
        float saturation = (cmax == 0.0f) ? 0 : (delta / cmax);
        float value = cmax;

        // multiply saturation and value by 255 to switch from normalized HSV space to pseudo-RGB space
        image.at<cv::Vec3b>(x, y) = cv::Vec3b(
            static_cast<uchar>(hue),
            static_cast<uchar>(saturation * 255),
            static_cast<uchar>(value * 255));
    }
}

void ImageFilter::EmbossImage(cv::Mat& image, bool useOpenMP)
{
    // since embossing compares with other pixels, that might already have been written
    // changes need to be made on a clone of the original image and then applied once the filter is complete 
    cv::Mat embossedImage = image.clone();

    #pragma omp parallel for schedule(dynamic) if(useOpenMP)
    for (int x = 0; x < image.rows; x++) {
        for (int y = 0; y < image.cols; y++) {
            if (x - 1 < 0 || y - 1 < 0) {
                // initialize pixels without top-left neighbor as gray
                embossedImage.at<cv::Vec3b>(x, y) = cv::Vec3b(128, 128, 128);
                continue;
            }

            cv::Vec3b pixel = image.at<cv::Vec3b>(x, y);
            cv::Vec3b compPixel = image.at<cv::Vec3b>(x - 1, y - 1);

            double diffR = fabs(compPixel[2] - pixel[2]);
            double diffG = fabs(compPixel[1] - pixel[1]);
            double diffB = fabs(compPixel[0] - pixel[0]);
            uchar diff = static_cast<uchar>(fmax(fmax(diffR, diffG), diffB));
            uchar gray = static_cast<uchar>(fmin(diff + 128, 255));
            embossedImage.at<cv::Vec3b>(x, y) = cv::Vec3b(gray, gray, gray);
        }
    }

    image = embossedImage;
}

void ImageFilter::EmbossImageCollapsed(cv::Mat& image, bool useOpenMP)
{
    // since embossing compares with other pixels, that might already have been written
    // changes need to be made on a clone of the original image and then applied once the filter is complete 
    cv::Mat embossedImage = image.clone();

    #pragma omp parallel for schedule(static) if(useOpenMP)
    for (int xy = 0; xy < image.rows * image.cols; xy++) {
        int x = xy / image.cols;
        int y = xy % image.cols;
        if (x - 1 < 0 || y - 1 < 0) {
            // initialize pixels without top-left neighbor as gray
            embossedImage.at<cv::Vec3b>(x, y) = cv::Vec3b(128, 128, 128);
            continue;
        }

        cv::Vec3b pixel = image.at<cv::Vec3b>(x, y);
        cv::Vec3b compPixel = image.at<cv::Vec3b>(x - 1, y - 1);

        double diffR = fabs(compPixel[2] - pixel[2]);
        double diffG = fabs(compPixel[1] - pixel[1]);
        double diffB = fabs(compPixel[0] - pixel[0]);
        uchar diff = static_cast<uchar>(fmax(fmax(diffR, diffG), diffB));
        uchar gray = static_cast<uchar>(fmin(fmax(diff + 128, 0), 255));
        embossedImage.at<cv::Vec3b>(x, y) = cv::Vec3b(gray, gray, gray);
    }

    image = embossedImage;
}
