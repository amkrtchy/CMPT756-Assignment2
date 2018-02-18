#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "colour-convert.h"

// AM: empy kernel to fire up GPU
__global__ void empyKernel(void) {

}

__global__ void rgb2yuvKernel(unsigned char *r, unsigned char *g, unsigned char *b, unsigned char *y, unsigned char *u, unsigned char *v) {

        unsigned char ny, cb, cr;

        // ny  = (unsigned char)(__fadd_rn(__fadd_rn(__fmul_rn(0.299, r[blockIdx.x]),
        //                        __fmul_rn(0.587, g[blockIdx.x])), 
        //                        __fmul_rn(0.114, b[blockIdx.x])));

        // cb = (unsigned char)(__fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(-0.169, r[blockIdx.x]),
        //                        __fmul_rn(-0.331, g[blockIdx.x])), 
        //                        __fmul_rn(0.499, b[blockIdx.x])), 128));

        // cr = (unsigned char) (__fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(0.499, r[blockIdx.x]),
        //                        __fmul_rn(-0.418, g[blockIdx.x])), 
        //                        __fmul_rn(-0.0813, b[blockIdx.x])), 128));

        
        
        ny  = (unsigned char)( 0.299*r[blockIdx.x] + 0.587*g[blockIdx.x] +  0.114*b[blockIdx.x]);
        cb = (unsigned char)(-0.169*r[blockIdx.x] - 0.331*g[blockIdx.x] +  0.499*b[blockIdx.x] + 128);
        cr = (unsigned char)( 0.499*r[blockIdx.x] - 0.418*g[blockIdx.x] - 0.0813*b[blockIdx.x] + 128);

        y[blockIdx.x]  = ny;
        u[blockIdx.x] = cb;
        v[blockIdx.x] = cr;

}

__global__ void yuv2rgbKernel(unsigned char *r, unsigned char *g, unsigned char *b, unsigned char *y, unsigned char *u, unsigned char *v) {


        int ny  = (int)y[blockIdx.x];
        int cb = (int)u[blockIdx.x] - 128;
        int cr = (int)v[blockIdx.x] - 128;
        
        int rt  = (int)(ny + 1.402*cr); 
        int gt = (int)(ny - 0.344*cb - 0.714*cr);
        int bt  = (int)(ny + 1.772*cb);

        r[blockIdx.x] = (rt < 255) ? rt: 255;
        r[blockIdx.x]  = (rt > 0) ? rt: 0;

        //g[blockIdx.x] = (int)( y[blockIdx.x] - 0.344*u[blockIdx.x] - 0.714*v[blockIdx.x]); 
        g[blockIdx.x] = (gt < 255) ? gt: 255;
        g[blockIdx.x] = (gt > 0) ? gt : 0;

        //b[blockIdx.x] = (int)(y[blockIdx.x]+ 1.772*u[blockIdx.x]);
        b[blockIdx.x] = (bt < 255 )? bt: 255;
        b[blockIdx.x] = (bt > 0) ? bt : 0;

}


// AM copy image to device

void copyToDevice(PPM_IMG img_in) { 
    


}

void copyToHost(PPM_IMG img_in) { 
    
    unsigned char * img_r_d;
    unsigned char * img_g_d;
    unsigned char * img_b_d;

    // AM: Allocate memory for the PPM_IMG struct on the device
    int size = img_in.w * img_in.h * sizeof(unsigned char);
   
    cudaMalloc((void **) &img_r_d, size);
    cudaMalloc((void **) &img_g_d, size);
    cudaMalloc((void **) &img_b_d, size);



}

YUV_IMG rgb2yuvGPU(PPM_IMG img_in)
{

    YUV_IMG img_out;


    img_out.w = img_in.w;
    img_out.h = img_in.h;

    img_out.img_y = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_u = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_v = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);

    // AM: Allocate memory for the PPM_IMG & YUV_IMG on the device
    unsigned char * img_r_d;
    unsigned char * img_g_d;
    unsigned char * img_b_d;

    unsigned char * img_y_d;
    unsigned char * img_u_d;
    unsigned char * img_v_d;

    int size = img_in.w * img_in.h * sizeof(unsigned char);
   
    cudaMalloc((void **) &img_r_d, size);
    cudaMalloc((void **) &img_g_d, size);
    cudaMalloc((void **) &img_b_d, size);

    cudaMalloc((void **) &img_y_d, size);
    cudaMalloc((void **) &img_u_d, size);
    cudaMalloc((void **) &img_v_d, size);


    // Copy PPM to device                           
    cudaMemcpy(img_r_d, img_in.img_r, size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_g_d, img_in.img_g, size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_b_d, img_in.img_b, size, cudaMemcpyHostToDevice);


    rgb2yuvKernel<<<img_in.w*img_in.h, 1>>>(img_r_d, img_g_d, img_b_d, img_y_d, img_u_d, img_v_d);//Launch the Kernel

    // Copy from device to host                      
    cudaMemcpy(img_out.img_y, img_y_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_u, img_u_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_v, img_v_d, size, cudaMemcpyDeviceToHost);


    return img_out;
}



PPM_IMG yuv2rgbGPU(YUV_IMG img_in)
{
    PPM_IMG img_out;
    //Put you CUDA setup code here.
    img_out.w = img_in.w;
    img_out.h = img_in.h;

    img_out.img_r = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_g = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_b = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);

    // AM: Allocate memory for the PPM_IMG & YUV_IMG on the device
    unsigned char * img_r_d;
    unsigned char * img_g_d;
    unsigned char * img_b_d;

    unsigned char * img_y_d;
    unsigned char * img_u_d;
    unsigned char * img_v_d;

    int size = img_in.w * img_in.h * sizeof(unsigned char);
   
    cudaMalloc((void **) &img_r_d, size);
    cudaMalloc((void **) &img_g_d, size);
    cudaMalloc((void **) &img_b_d, size);

    cudaMalloc((void **) &img_y_d, size);
    cudaMalloc((void **) &img_u_d, size);
    cudaMalloc((void **) &img_v_d, size);


    // Copy YUV to device                           
    cudaMemcpy(img_y_d, img_in.img_y, size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_u_d, img_in.img_u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(img_v_d, img_in.img_v, size, cudaMemcpyHostToDevice);

    yuv2rgbKernel<<<img_in.w*img_in.h, 1>>>(img_r_d, img_g_d, img_b_d, img_y_d, img_u_d, img_v_d);//Launch the Kernel


    // Copy from device to host                      
    cudaMemcpy(img_out.img_r, img_r_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_g, img_g_d, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(img_out.img_b, img_b_d, size, cudaMemcpyDeviceToHost);


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
