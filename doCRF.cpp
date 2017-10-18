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
        std::ifstream fin;
        fin.open(bin_in,std::ios::in|std::ios::binary);

        if(!fin){
            std::cout<<"open error!"<<std::endl;
            return 1;
        }

        int predMapSize = imgW*imgH*M;

        float* buffer = new float[predMapSize];
        fin.read((char*)buffer,sizeof(float)*predMapSize);
        fin.close();

        for (int i=0;i<predMapSize;i++)
        {
            float tmp = -log(*(buffer+i));
            *(buffer+i) = tmp;
        }
        
        // Setup the CRF model
        DenseCRF2D crf(imgW, imgH, M);
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

        // Do map inference
        short * map = new short[imgW*imgH];
        crf.map(10, map);

        // Store the result
        std::ofstream fout;
        fout.open(bin_out,std::ios::binary|std::ios::out);
        fout.write((char*)map,sizeof(short)*(predMapSize/M));
        fout.close();

        delete[] map;
        delete[] buffer;
        delete[] imAr;
        return 0;
    }

}
