// The functions contained in this file are pretty dummy
// and are included only as a placeholder. Nevertheless,
// they *will* get included in the shared library if you
// don't remove them :)
//
// Obviously, you 'll have to write yourself the super-duper
// functions to include in the resulting library...
// Also, it's not necessary to write every function in this file.
// Feel free to add more files in this project. They will be
// included in the resulting library.

#include <cstdio>
#include <cmath>
#include "/home/qiuchao/PycharmProjects/inferMINC/denseCRF/examples/util.h"
#include "/home/qiuchao/PycharmProjects/inferMINC/denseCRF/densecrf.h"
#include <fstream>
#include <stdlib.h>
#include<iostream>
#include "opencv2/opencv.hpp"

extern "C"
{
    //template<class T>
    void matToArray(const cv::Mat& srcMat, unsigned char* dstAr)
    {
        int h,w,c,k,kt;
        h = srcMat.rows;
        w = srcMat.cols;
        c = 3;
        k = 0;
        const unsigned char* srcPtr;

        for (int i=0;i<h;i++)
        {
            srcPtr = srcMat.ptr<unsigned char>(i);

            for (int j=0;j<w*c;j+=3)
            {
                dstAr[k] = srcPtr[j];
                dstAr[k+1] = srcPtr[j+1];
                dstAr[k+2] = srcPtr[j+2];
                k += 3;
            }
        }
    }

    /*------------------------------------------------------*/

    int doCRF(const char* img_in,const char* bin_in,const char* bin_out,
                    const int imgH,const int imgW,const int M,const int lp1,
                    const int cp1,const int wp1,const int lp2, const int wp2,
                    const int cp3, const int wp3)
    {

        //char* a = "examples/1.ppm";
        //char* c = "examples/output.ppm";
        //const char* a = img_in;
        //const char* c = img_out;

        // Number of labels
        //const int M = 6;

        // Load the color image and some crude annotations (which are used in a simple classifier)
        //int W, H;
        /*
        unsigned char * im = readPPM( img_in, W, H );

        if (!im){
            printf("Failed to load image!\n");
            return 1;
        }
        */

        /////////// Put your own unary classifier here! ///////////
        std::ifstream fin;
        //fin.open("examples/test.bin",std::ios::in|std::ios::binary);
        fin.open(bin_in,std::ios::in|std::ios::binary);

        if(!fin){
            std::cout<<"open error!"<<std::endl;
            return 1;
        }

        int predMapSize = imgW*imgH*M;

        float* buffer = new float[predMapSize];//30042600//1165500
        fin.read((char*)buffer,sizeof(float)*predMapSize);
        fin.close();

        for (int i=0;i<predMapSize;i++)
        {
            float tmp = -log(*(buffer+i));
            *(buffer+i) = tmp;
        }

        ///////////////////////////////////////////////////////////
        // Setup the CRF model
        DenseCRF2D crf(imgW, imgH, M);
        // Specify the unary potential as an array of size W*H*(#classes)
        // packing order: x0y0l0 x0y0l1 x0y0l2 .. x1y0l0 x1y0l1 ...
        crf.setUnaryEnergy( buffer );

        cv::Mat imCV = cv::imread(img_in);
        cv::Mat imLab;
        cv::cvtColor(imCV,imLab,cv::COLOR_BGR2Lab);
        int tmpR = imLab.rows;
        int tmpC = imLab.cols;


        unsigned char * imAr = new unsigned char[tmpR*tmpC*3];
        matToArray(imLab, imAr);


        crf.addPairwiseBilateral( lp1, lp1, cp1, cp1, cp1, imAr, wp1 );
        crf.addPairwiseGaussian( lp2, lp2, wp2 );
        crf.addPairwiseColorGaussian(cp3, cp3, cp3, imAr, wp3);
        //crf.addPairwiseBilateral( 3, 3, 20, 20, 20, imAr, 10 );
        //crf.addPairwiseBilateral( 0.2, 0.2, 20, 2.5, 2.5, imAr, 8 );
        //crf.addPairwiseBilateral( 60, 60, 20, 100, 100, imAr, 10 );
        //crf.addPairwiseGaussian( 6, 6, 10 );


        // Do map inference
        short * map = new short[imgW*imgH];
        crf.map(10, map);

        // Store the result
        //unsigned char *res = colorize( map, W, H );
        //writePPM( c, W, H, res );
        std::ofstream fout;
        //fout.open("/home/qc/PycharmProjects/inferMINC/result.bin",std::ios::binary|std::ios::out);
        fout.open(bin_out,std::ios::binary|std::ios::out);
        fout.write((char*)map,sizeof(short)*(predMapSize/M));//166500
        fout.close();

        //delete[] im;
        delete[] map;
        delete[] buffer;
        delete[] imAr;

        return 0;
    }

}
