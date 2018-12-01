GPU based optical flow extraction from videos
=================================================

### Features
Forked from https://github.com/dl-container-registry/furnari-flow
This tool allows to extract optical flow from image sequences. The tool creates a video where x and y optical flow images are stored side by side. Optical flow is obtained by clipping large displacement. Other options are available, such as dilatation (how much to skip for calculating each optical flow).

### Dependencies:
 * [OpenCV 2.4](http://opencv.org/downloads.html)
 * [cmake](https://cmake.org/)

### Installation
First, build opencv 2.4.x with gpu support. To do so, download opencv 2.4.x sources from https://opencv.org/releases.html. Unzip the downloaded archive, then enter the opencv folder and enter the following commands:

 * `mkdir build`
 * `cd build`
 * `cmake -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF ..`
 * `make`

Then clone the current repository. Type:

 * `export OpenCV_DIR=path_to_opencv_build_directory` (not required if you `make install` the opencv)
 * `mkdir build`
 * `cd build`
 * `cmake -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF ..`
 * `make`

### Usage
The software assumes that all video frames have been extracted in a directory. Files should be named according to some pattern, e.g., `img_%07d.jpg`.
```
./compute_flow /z/dat list_dir_1.txt 0
```
The above command will read /z/dat/list_dir_1.txt file line by line. Then, it will read an image sequence of /z/dat/Kinetics-400-frames/line1 and save optical flow /z/dat/Kinetics-400-flow/line1/u and /z/dat/Kinetics-400-flow/line1/v. You can modify `fsrc` and `ftrg` of the source file to change Kinetics-400-frames and Kinetics-400-flow.
