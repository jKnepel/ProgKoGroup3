#include <iostream>
#include <opencv2/opencv.hpp>
#include <mpi.h>

/// <summary>
/// Applies the specified filter methods on an image located at the specified file path inside the current MPI process.
/// </summary>
/// <param name="rank">The rank of the current MPI process</param>
/// <param name="size">The size of the MPI processes</param>
/// <param name="imagePath">The file path of the to be filtered image</param>
/// <param name="outputDir">The output directory path of the to be filtered image after saving</param>
/// <param name="filterMethods">The image filter methods that should be run</param>
/// <param name="useOpenMP">Wether to use OpenMP for the image filter or not</param>
/// <param name="showImage">Wether to show the resulting image or not</param>
/// <param name="saveImage">Wether to save the resulting image or not</param>
static void MPIFilters(int& rank, int& size,
    const std::string& imagePath,
    const std::string& outputDir,
    const std::vector<std::function<void(cv::Mat&, bool)>>& filterMethods,
    bool useOpenMP = true,
    bool showImage = false,
    bool saveImage = false
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

    // apply the filters
    for (const auto& filter : filterMethods) {
        filter(partialImage, useOpenMP);
    }

    // gather the partial image back to the full image on the host process
    MPI_Gatherv(partialImage.data, sendcounts[rank], MPI_UNSIGNED_CHAR,
        image.data, sendcounts, displs, MPI_UNSIGNED_CHAR,
        0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (showImage) {
            cv::imshow("Final Image MPIFilters", image);
        }

        if (saveImage) {
            cv::imwrite(outputDir + "/resulting_image_mpi.png", image);
        }
    }
}
