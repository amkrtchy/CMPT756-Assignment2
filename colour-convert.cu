#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "colour-convert.h"


// Empy kernel to fire up GPU
__global__ void emptyKernel(void) {

}

// RGB 2 YUV kernel
__global__ void rgb2yuvKernel(unsigned char *r, unsigned char *g, unsigned char *b, unsigned char *y, 
                              unsigned char *u, unsigned char *v, int *n) {

        int ind = threadIdx.x + blockIdx.x*blockDim.x;
        unsigned char ny, cb, cr;
    
        if (ind < *n) {
            ny  = (unsigned char)( 0.299*r[ind] + 0.587*g[ind] +  0.114*b[ind]);
            cb = (unsigned char)(-0.169*r[ind] - 0.331*g[ind] +  0.499*b[ind] + 128);
            cr = (unsigned char)( 0.499*r[ind] - 0.418*g[ind] - 0.0813*b[ind] + 128);

            y[ind]  = ny;
            u[ind] = cb;
            v[ind] = cr;
        }

}

// YUV 2 RGB kernel
__global__ void yuv2rgbKernel(unsigned char *r, unsigned char *g, unsigned char *b, 
                              unsigned char *y, unsigned char *u, unsigned char *v, int *n) {

         
        int ind = threadIdx.x + blockIdx.x*blockDim.x;

        if (ind < *n) {
            int ny  = (int)y[ind];
            int cb = (int)u[ind] - 128;
            int cr = (int)v[ind] - 128;
            
            int rt  = (int)(ny + 1.402*cr); 
            int gt = (int)(ny - 0.344*cb - 0.714*cr);
            int bt  = (int)(ny + 1.772*cb);


            rt = (rt < 255) ? rt: 255;
            rt  = (rt > 0) ? rt: 0;

            gt = (gt < 255) ? gt: 255;
            gt = (gt > 0) ? gt : 0;

            bt = (bt < 255 )? bt: 255;
            bt = (bt > 0) ? bt : 0; 

            r[ind] = rt;
            g[ind] = gt;
            b[ind] = bt;
        }

}


void launchEmptyKernel() {

    emptyKernel<<<1, 1>>>();
}


void copyToDevice(PPM_IMG img_in) { 
    
    // Allocate memory for the PPM_IMG
    unsigned char * img_r_d;
    unsigned char * img_g_d;
    unsigned char * img_b_d;

    int size = img_in.w * img_in.h * sizeof(unsigned char);
   
    cudaMalloc((void **) &img_r_d, size);
    cudaMalloc((void **) &img_g_d, size);
    cudaMalloc((void **) &img_b_d, size);


    // Copy PPM to device                           
    cudaMemcpy(img_r_d, img_in.img_r, size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_g_d, img_in.img_g, size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_b_d, img_in.img_b, size, cudaMemcpyHostToDevice);

    cudaFree(img_r_d);
    cudaFree(img_g_d);
    cudaFree(img_b_d);


}

void copyToDeviceAndBack(PPM_IMG img_in) { 

    // Allocate memory for the PPM_IMG on the device
    unsigned char * img_r_d;
    unsigned char * img_g_d;
    unsigned char * img_b_d;


    int size = img_in.w * img_in.h * sizeof(unsigned char);
   
    cudaMalloc((void **) &img_r_d, size);
    cudaMalloc((void **) &img_g_d, size);
    cudaMalloc((void **) &img_b_d, size);


    // Copy PPM to device                           
    cudaMemcpy(img_r_d, img_in.img_r, size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_g_d, img_in.img_g, size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_b_d, img_in.img_b, size, cudaMemcpyHostToDevice);

    // Copy from device to host                      
    cudaMemcpy(img_in.img_r, img_r_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_in.img_g, img_g_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_in.img_b, img_b_d, size, cudaMemcpyDeviceToHost);

    cudaFree(img_r_d);
    cudaFree(img_g_d);
    cudaFree(img_b_d);

}

YUV_IMG rgb2yuvGPU(PPM_IMG img_in)
{

    YUV_IMG img_out;


    img_out.w = img_in.w;
    img_out.h = img_in.h;

    img_out.img_y = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_u = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_v = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);

    // Allocate memory for the PPM_IMG & YUV_IMG on the device
    unsigned char * img_r_d;
    unsigned char * img_g_d;
    unsigned char * img_b_d;

    unsigned char * img_y_d;
    unsigned char * img_u_d;
    unsigned char * img_v_d;

    int * N_d;

    int size = img_in.w * img_in.h * sizeof(unsigned char);
   
    cudaMalloc((void **) &img_r_d, size);
    cudaMalloc((void **) &img_g_d, size);
    cudaMalloc((void **) &img_b_d, size);

    cudaMalloc((void **) &img_y_d, size);
    cudaMalloc((void **) &img_u_d, size);
    cudaMalloc((void **) &img_v_d, size);
    cudaMalloc((void **) &N_d, sizeof(int));


    // Copy PPM to device                           
    cudaMemcpy(img_r_d, img_in.img_r, size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_g_d, img_in.img_g, size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_b_d, img_in.img_b, size, cudaMemcpyHostToDevice);

    int N = img_in.w*img_in.h;
    int M = 512; // number of threads

    cudaMemcpy(N_d, &N, sizeof(int), cudaMemcpyHostToDevice);


    rgb2yuvKernel<<<(N+M-1)/M,M>>>(img_r_d, img_g_d, img_b_d, img_y_d, img_u_d, img_v_d, N_d);//Launch the Kernel

    // Copy from device to host                      
    cudaMemcpy(img_out.img_y, img_y_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_u, img_u_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_v, img_v_d, size, cudaMemcpyDeviceToHost);

    cudaFree(img_r_d);
    cudaFree(img_g_d);
    cudaFree(img_b_d);

    cudaFree(img_y_d);
    cudaFree(img_u_d);
    cudaFree(img_v_d);


    return img_out;
}



PPM_IMG yuv2rgbGPU(YUV_IMG img_in)
{
    PPM_IMG img_out;
    
    img_out.w = img_in.w;
    img_out.h = img_in.h;

    img_out.img_r = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_g = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_b = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);

    // Allocate memory for the PPM_IMG & YUV_IMG on the device
    unsigned char * img_r_d;
    unsigned char * img_g_d;
    unsigned char * img_b_d;

    unsigned char * img_y_d;
    unsigned char * img_u_d;
    unsigned char * img_v_d;

    int * N_d;

    int size = img_in.w * img_in.h * sizeof(unsigned char);
   
    cudaMalloc((void **) &img_r_d, size);
    cudaMalloc((void **) &img_g_d, size);
    cudaMalloc((void **) &img_b_d, size);

    cudaMalloc((void **) &img_y_d, size);
    cudaMalloc((void **) &img_u_d, size);
    cudaMalloc((void **) &img_v_d, size);
    cudaMalloc((void **) &N_d, sizeof(int));


    // Copy YUV to device                           
    cudaMemcpy(img_y_d, img_in.img_y, size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_u_d, img_in.img_u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_v_d, img_in.img_v, size, cudaMemcpyHostToDevice);


    int N = img_in.w*img_in.h;
    int M = 512;

    cudaMemcpy(N_d, &N, sizeof(int), cudaMemcpyHostToDevice);

    yuv2rgbKernel<<<(N+M-1)/M,M>>>(img_r_d, img_g_d, img_b_d, img_y_d, img_u_d, img_v_d, N_d);//Launch the Kernel
    

    // Copy from device to host                      
    cudaMemcpy(img_out.img_r, img_r_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_g, img_g_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_b, img_b_d, size, cudaMemcpyDeviceToHost);

    cudaFree(img_r_d);
    cudaFree(img_g_d);
    cudaFree(img_b_d);

    cudaFree(img_y_d);
    cudaFree(img_u_d);
    cudaFree(img_v_d);


    return img_out;
}

//Convert RGB to YUV444, all components in [0, 255]
YUV_IMG rgb2yuv(PPM_IMG img_in)
{
    YUV_IMG img_out;
    int i;//, j;
    unsigned char r, g, b;
    unsigned char y, cb, cr;
    
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_y = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_u = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_v = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
 
    for(i = 0; i < img_out.w*img_out.h; i ++){
        r = img_in.img_r[i];
        g = img_in.img_g[i];
        b = img_in.img_b[i];
        
        y  = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
        cb = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
        cr = (unsigned char)( 0.499*r - 0.418*g - 0.0813*b + 128);
        
        img_out.img_y[i] = y;
        img_out.img_u[i] = cb;
        img_out.img_v[i] = cr;
    }

    
    return img_out;
}

unsigned char clip_rgb(int x)
{
    if(x > 255)
        return 255;
    if(x < 0)
        return 0;

    return (unsigned char)x;
}

//Convert YUV to RGB, all components in [0, 255]
PPM_IMG yuv2rgb(YUV_IMG img_in)
{
    PPM_IMG img_out;
    int i;
    int  rt,gt,bt;
    int y, cb, cr;
    
    
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_r = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_g = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_b = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);

    for(i = 0; i < img_out.w*img_out.h; i ++){
        y  = (int)img_in.img_y[i];
        cb = (int)img_in.img_u[i] - 128;
        cr = (int)img_in.img_v[i] - 128;
        
        rt  = (int)( y + 1.402*cr);
        gt  = (int)( y - 0.344*cb - 0.714*cr); 
        bt  = (int)( y + 1.772*cb);

        img_out.img_r[i] = clip_rgb(rt);
        img_out.img_g[i] = clip_rgb(gt);
        img_out.img_b[i] = clip_rgb(bt);
    }

    
    return img_out;
}
