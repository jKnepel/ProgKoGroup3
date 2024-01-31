# ProgKoGroup3
## Julian Knepel - Group 3
## Grayscale, Embossing, HSV - OpenMP, MPI

This is a university project for Module Programmierkonzepte und Algorithmen WiSe23/24. It uses C++, Visual Studio, OpenCV, OpenMPI and MPI
to implement 3 different image filters and parallelize them. 

### Installation
- To run this, clone the project and open it using a supported Visual Studio version with all C++ for Windows dependencies. 
- OpenCV ([4.9.0](https://sourceforge.net/projects/opencvlibrary/files/4.9.0/)) and [MS-MPI](https://www.microsoft.com/en-us/download/details.aspx?id=100305) also need to be installed. 
- Add the OpenCV binary folder ...\opencv\build\x64\vc16\bin and MPI binary folder ...\MPI\Bin to the Windows path 
- Create the Windows environment variables 
- - OPENCV_INC pointing to ...\opencv\build\include
- - OPENCV_LIB64 pointing to ...\opencv\build\x64\vc16\lib
- - MSMPI_INC pointing to ...\MPI\Include 
- - MSMPI_LIB64 pointing to ...\MPI\Lib\x64 

If another OpenCV version, other than 4.9.0, should be used, the included OpenCV lib also needs to be adjusted in Visual Studio. This can be done in Project > ProgKoGroup3 Properties > Linker > Input > Additional Dependencies, where opencv_world490.lib must be changed for the correct library version.

### Run the project
Under Source Files > Main.cpp the correct benchmark and filters can be chosen. in line 21 a path to the desired image needs to be added. In lines 1-3 the directives can be used to switch between filters, MPI filters and OpenCV filters. The lines 21-37 can be used to add various options to the benchmarks, like the required filters, threadnumber, use of OpenMP, image saving etc. The project can then be run using Visual Studio. If MPI should use multiple processes, the MPI directive needs to be enabled and the project build. The command `mpiexec.exe -n N ProgKoGroup3.exe` then needs to be run in the \ProgKoGroup3\x64\Release folder with N representing the number of processes.