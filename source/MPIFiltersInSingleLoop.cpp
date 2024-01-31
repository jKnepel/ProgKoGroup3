#include <iostream>
#include <opencv2/opencv.hpp>
#include <mpi.h>
#include "ImageFilter.h"

/// <summary>
/// Early attempt of running all three filters within a single loop. Because the embossing filter has to be applied on 
/// neighboring pixels that were already filtered by the two other filters and with parallelisation race-conditions 
/// lead to pixels filtered in random order, the embossing filter has to be done in an additional loop. This
/// makes the performance gains negligible.
/// </summary>
/// <param name="rank">The rank of the current MPI process</param>
/// <param name="size">The size of the MPI processes</param>
/// <param name="imagePath">The file path of the to be filtered image</param>
/// <param name="outputDir">The output directory path of the to be filtered image after saving</param>
/// <param name="useOpenMP">Wether to use OpenMP for the image filter or not</param>
/// <param name="showImage">Wether to show the resulting image or not</param>
/// <param name="saveImage">Wether to save the resulting image or not</param>
/// <param name="doHSV">Wether to apply a HSV filter on the image or not</param>
/// <param name="doGrayscale">Wether to apply a grayscale filter on the image or not</param>
/// <param name="doEmboss">Wether to apply an embossing filter on the image or not</param>
static void MPIFiltersInSingleLoop(int& rank, int& size,
    const std::string& imagePath,
    const std::string& outputDir,
    bool useOpenMP = true,
    bool showImage = false,
    bool saveImage = false,
    bool doHSV = true,
    bool doGrayscale = true,
    bool doEmboss = true
) {
    int imageProperties[4];
    cv::Mat image;
    cv::Mat partialImage;

    if (rank == 0) {
        // load the image on the host process
        image = cv::imread(imagePath);

        if (image.empty())
            throw std::invalid_argument("Could not open or find the image!");

        imageProperties[0] = image.rows;
        imageProperties[1] = image.cols;
        imageProperties[2] = image.type();
        imageProperties[3] = image.channels();

#ifdef _DEBUG
        // NOTE : only print image values to console in debug mode, since it will skew the benchmark result
        std::cout << "Image width: " << image.cols << std::endl;
        std::cout << "Image height: " << image.rows << std::endl;
        std::cout << "Image pixels: " << image.cols * image.rows << std::endl;
        std::cout << "Rank: " << rank << ", Size: " << size << std::endl;
#endif
    }

    // distribute the image from the host between the processes
    MPI_Bcast(imageProperties, 4, MPI_INT, 0, MPI_COMM_WORLD);

    int* sendcounts = new int[size];
    int* displs = new int[size];
    int sizePerProcess = imageProperties[0] / size;
    int rest = imageProperties[0] % size;

    int increment = 0;
    for (int i = 0; i < size; i++) {
        displs[i] = increment;
        sendcounts[i] = (i == size - 1) ? sizePerProcess + rest : sizePerProcess;
        if (i == rank) partialImage = cv::Mat(sendcounts[i], imageProperties[1], imageProperties[2]);
        sendcounts[i] *= imageProperties[1] * imageProperties[3];
        increment += sendcounts[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Scatterv(image.data, sendcounts, displs, MPI_UNSIGNED_CHAR,
        partialImage.data, sendcounts[rank], MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    // apply the hsv and grayscale filter
    #pragma omp parallel for schedule(dynamic) if(useOpenMP)
    for (int x = 0; x < partialImage.rows; x++) {
        for (int y = 0; y < partialImage.cols; y++) {
            cv::Vec3b pixel = partialImage.at<cv::Vec3b>(x, y);

            if (doHSV) {
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
                partialImage.at<cv::Vec3b>(x, y) = pixel = cv::Vec3b(
                    static_cast<uchar>(hue),
                    static_cast<uchar>(saturation * 255),
                    static_cast<uchar>(value * 255));
            }

            if (doGrayscale) {
                uchar gray = static_cast<uchar>(0.21 * pixel[2] + 0.72 * pixel[1] + 0.07 * pixel[0]);
                partialImage.at<cv::Vec3b>(x, y) = pixel = cv::Vec3b(gray, gray, gray);
            }
        }
    }

    if (doEmboss) {
        // apply the embossing filter individually
        ImageFilter::EmbossImage(partialImage, useOpenMP);
    }

    // gather the partial image back to the full image on the host process
    MPI_Gatherv(partialImage.data, sendcounts[rank], MPI_UNSIGNED_CHAR,
        image.data, sendcounts, displs, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (showImage) {
            cv::imshow("Final Image MPIFiltersInSingleLoop", image);
        }

        if (saveImage) {
            cv::imwrite(outputDir + "/resulting_image_mpi_single_loop.png", image);
        }
    }
}
