#define RUN_SIMPLE
#undef RUN_MPI
#undef RUN_OPENCV

#include <iostream>
#include "AlgorithmBenchmark.h"
#include "ImageFilter.h"
#include "SimpleFilters.cpp"
#include "MPIFilters.cpp"
#include "MPIFiltersInSingleLoop.cpp"
#include "OpenCVFilters.cpp"

#ifdef RUN_MPI
#include <mpi.h>
#include <chrono>
using std::chrono::duration;
using std::chrono::high_resolution_clock;
#endif

int main(int argc, char** argv) {
    const std::string imagePath = "";
    const std::string outputDir = "";
    const int numberOfRepetitions = 100;

    const bool useOpenMP = true;
    const bool showImage = false;
    const bool saveImage = false;

    if (useOpenMP)
        omp_set_num_threads(omp_get_num_procs());

    // defines what filters will be run in what order
    std::vector<std::function<void(cv::Mat&, bool)>> filterMethods = {
        &ImageFilter::HSVImage,
        &ImageFilter::GrayscaleImage,
        &ImageFilter::EmbossImage,
    };

    AlgorithmBenchmark benchmark{};

#ifdef RUN_SIMPLE
    // run a benchmark of the defined filter methods
    std::cout << "Simple Filters: " << std::endl;
    auto simpleAlgorithm = std::bind(SimpleFilters, imagePath, outputDir, filterMethods, useOpenMP, showImage, saveImage);
    benchmark.RunBenchmark(simpleAlgorithm, numberOfRepetitions);
    std::cout << "Total Duration: " << benchmark.GetTotalDuration() << "ms" << std::endl;
    std::cout << "Average Duration: " << benchmark.GetAvgDuration() << "ms" << std::endl;
    std::cout << std::endl;
    benchmark.ResetBenchmark();
#endif

#ifdef RUN_MPI
    // run a benchmark of the defined filter methods divided between mpi processes
    auto startTimeMPI = high_resolution_clock::now();
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        std::cout << "MPI Filters: " << std::endl;
    }

    auto mpiAlgorithm = std::bind(MPIFilters, rank, size, imagePath, outputDir, filterMethods, useOpenMP, showImage, saveImage);
    benchmark.RunBenchmark(mpiAlgorithm, numberOfRepetitions);
    if (rank == 0) {
        auto endTimeMPI = high_resolution_clock::now();
        duration<double, std::milli> duration = endTimeMPI - startTimeMPI;
        std::cout << "Total Duration: " << duration.count() << "ms" << std::endl;
        std::cout << "Total Duration Algorithm: " << benchmark.GetTotalDuration() << "ms" << std::endl;
        std::cout << "Average Duration Algorithm: " << benchmark.GetAvgDuration() << "ms" << std::endl;
        std::cout << std::endl;
    }
    benchmark.ResetBenchmark();

    MPI_Finalize();
#endif

#ifdef RUN_OPENCV
    // run a benchmark of all filter methods using opencv
    const bool doHSV = true;
    const bool doGrayscale = true;
    const bool doEmboss = true;
    std::cout << "OpenCV Filters: " << std::endl;
    auto openCVAlgorithm = std::bind(OpenCVFilters, imagePath, outputDir, showImage, saveImage, doHSV, doGrayscale, doEmboss);
    benchmark.RunBenchmark(openCVAlgorithm, numberOfRepetitions);
    std::cout << "Total Duration: " << benchmark.GetTotalDuration() << "ms" << std::endl;
    std::cout << "Average Duration: " << benchmark.GetAvgDuration() << "ms" << std::endl;
    std::cout << std::endl;
    benchmark.ResetBenchmark();
#endif

    cv::waitKey(0);

    return 0;
}
