//************************************************************************
// compute_flow.cpp
// Computes OpenCV GPU Zach et al. [1] TVL1 Optical Flow
// Dependencies: OpenCV (compiled for GPU)
// Author: Christoph Feichtenhofer -> Forked by Antonino Furnari -> again by Kyle Min
// Institution: Graz University of Technology
// Email: feichtenhofer@tugraz
// Date: Nov. 2015
// [1] C. Zach, T. Pock, H. Bischof: A duality based approach for realtime TV-L 1 optical flow. DAGM 2007.
//************************************************************************

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cstdlib>
#include <string>
#include <vector>
#include <math.h>
#include <iostream>
#include <fstream>
#include <sys/time.h>
#include <time.h>
#include <queue>

#include <opencv2/core/core.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;
using namespace cv;

static void convertFlowToImage(const Mat &flowIn, Mat &flowOut, float lowerBound, float higherBound){
    #define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
    for (int i = 0; i < flowIn.rows; ++i) {
        for (int j = 0; j < flowIn.cols; ++j) {
            float x = flowIn.at<float>(i,j);
            flowOut.at<uchar>(i,j) = CAST(x, lowerBound, higherBound);
        }
    }
    #undef CAST
}

int main( int argc, char *argv[] ){
    cv::gpu::GpuMat frame0GPU, frame1GPU, uGPU, vGPU;
    Mat currentFrame;
    Mat frame0_rgb_, frame1_rgb_, frame0_rgb, frame1_rgb, frame0, frame1, rgb_out;
    Mat frame0_32, frame1_32, imgU, imgV;
    Mat motion_flow, flow_rgb;

    string img_format = "img_%07d.jpg", flow_format = "flow_%07d.jpg";

    const char* keys = {
                "{ h  | help     | false | print help message }"
                "{ g  | gpuID    |  0    | use this gpu}"
                "{ d  | dilation |  1    | temporal dilation (1: use neighbouring frames, 2: skip one, 3: skip two)}"
                "{ b  | bound    |  20   | maximum optical flow for clipping}"
    };

    CommandLineParser cmd(argc, argv, keys);

    if (cmd.get<bool>("help"))
    {
        cout << "Usage: compute_flow input_folder gpu_id [options]" << endl;
        cout << "Available options:" << endl;
        cmd.printParams();
        cout << "Example: compute_flow /z/dat/Kinetics list_dir_1.txt 0 [options]" << endl;
        return 0;
    }

    string input_folder, input_file;
    int gpuID = 0, dilation = 1, bound = 20;

    if (argc > 3) {
        input_folder = argv[1];
        input_file = argv[2];
        gpuID = atoi(argv[3]);
        dilation = cmd.get<int>("dilation");
        bound = cmd.get<float>("bound");

    }
    else {
        cout << "Not enough parameters!"<< endl;
        cout << "Usage: compute_flow input_folder gpu_id [options]" << endl;
        cout << "Available options:" << endl;
        cmd.printParams();
        cout << "Example: compute_flow /z/dat/Kinetics list_dir_1.txt 0 [options]" << endl;
        return 0;
    }

    int gpuCounts = cv::gpu::getCudaEnabledDeviceCount();
    cout << "Number of GPUs present " << gpuCounts << endl;

    cv::gpu::setDevice(gpuID);
    cv::gpu::printShortCudaDeviceInfo(cv::gpu::getDevice());
    cv::gpu::OpticalFlowDual_TVL1_GPU alg_tvl1;

    string fsrc = "Kinetics-400-frames/", ftrg = "Kinetics-400-flow/";
    string pf;
    ifstream file((input_folder+"/"+input_file).c_str());
    while(getline(file, pf)){
        if (pf[pf.length()-1] == '\n') {
            pf.erase(pf.length()-1);
        }

        cout << "Start processing: " << pf << endl;

        queue<Mat> framequeue;
        VideoCapture cap; //open video
        try{
            cap.open(input_folder+"/"+fsrc+pf+"/"+img_format);
        }
        catch (exception& e){
            cout << e.what() << endl; //print exception
        }

        int width = 0, height = 0;
        if( cap.isOpened() == 0 ){
            cout << "Unable to capture" << endl;
            return -1; //exit with a -1 error
        }

        int frame_to_write = 1;

        //NOW THE VIDEO IS OPEN AND WE CAN START GRABBING FRAMES

        //fill the frame queue
        for (int ii = 0; ii < dilation; ii++){
            cap >> currentFrame;
            width = currentFrame.cols;
            height = currentFrame.rows;
            framequeue.push(currentFrame.clone());
        }

        cap >> currentFrame;
        while (!currentFrame.empty()){
            //extract frame 0 from the front of the queue and pop
            frame0_rgb = framequeue.front().clone();
            framequeue.pop();

            // Allocate memory for the images
            flow_rgb = Mat(Size(width,height),CV_8UC3);
            motion_flow = Mat(Size(width,height),CV_8UC3);
            frame0 = Mat(Size(width,height),CV_8UC1);
            frame1 = Mat(Size(width,height),CV_8UC1);
            frame0_32 = Mat(Size(width,height),CV_32FC1);
            frame1_32 = Mat(Size(width,height),CV_32FC1);

            // Convert the image to grey and float
            cvtColor(currentFrame,frame1,CV_BGR2GRAY);
            frame1.convertTo(frame1_32,CV_32FC1,1.0/255.0,0);

            cvtColor(frame0_rgb,frame0,CV_BGR2GRAY);
            frame0.convertTo(frame0_32,CV_32FC1,1.0/255.0,0);

            frame1GPU.upload(frame1);
            frame0GPU.upload(frame0);
            alg_tvl1(frame0GPU,frame1GPU,uGPU,vGPU);

            uGPU.download(imgU);
            vGPU.download(imgV);

            float min_u_f = -bound;
            float max_u_f = bound;

            float min_v_f = -bound;
            float max_v_f = bound;

            Mat img_u(imgU.rows, imgU.cols, CV_8UC1);
            Mat img_v(imgV.rows, imgV.cols, CV_8UC1);

            convertFlowToImage(imgU, img_u, min_u_f, max_u_f);
            convertFlowToImage(imgV, img_v, min_v_f, max_v_f);

            char *flowname = new char[255];
            sprintf(flowname, flow_format.c_str(), frame_to_write);

            imwrite(input_folder+"/"+ftrg+pf+"/u/"+string(flowname),img_u);
            imwrite(input_folder+"/"+ftrg+pf+"/v/"+string(flowname),img_v);

            frame_to_write++;
            framequeue.push(currentFrame.clone());
            cap >> currentFrame;
        }
    }
    return 0;
}


