#include <stdio.h>
#include <string.h>
#include <stdlib.h>
// CUDA Runtime
#include <cuda_runtime.h>
// Utility and system includes
#include <helper_cuda.h>
// helper for shared that are common to CUDA Samples
#include <helper_functions.h>
#include <helper_timer.h>

#include "colour-convert.h"

void run_cpu_color_test(PPM_IMG img_in);
void run_gpu_color_test(PPM_IMG img_in);

int main(){
    PPM_IMG img_ibuf_c;

    printf("Running colour space converter .\n");
    img_ibuf_c = read_ppm("in.ppm");

    run_cpu_color_test(img_ibuf_c);
    run_gpu_color_test(img_ibuf_c);
    free_ppm(img_ibuf_c);
    
    return 0;
}


void run_gpu_color_test(PPM_IMG img_in)
{
    StopWatchInterface *timer=NULL;
    PPM_IMG img_obuf_rgb;
    YUV_IMG img_obuf_yuv;


  
    printf("Starting GPU processing...\n");


    // Test for the copy to device and back
    launchEmptyKernel();
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    copyToDevice(img_in); 
    sdkStopTimer(&timer);

    printf("Copy PPM to device: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);


    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    copyToDeviceAndBack(img_in); //copy to device
    sdkStopTimer(&timer);

    printf("Copy PPM to device and back: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);
    
 
    // Test for RGB 2 YUV GPU conversion
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    img_obuf_yuv = rgb2yuvGPU(img_in); //Start RGB 2 YUV
    sdkStopTimer(&timer);

    printf("RGB to YUV conversion time(GPU): %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    // Is Conversion ok? Note, there can be small differencein pixel values due to different rounding mechanisms
    printf("Is my GPU to YUV correct? %d\n", confirm_gpu_rgb2yuv(img_obuf_yuv, img_in));


    // Test for RGB 2 YUV GPU conversion
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);
	
    img_obuf_rgb = yuv2rgbGPU(img_obuf_yuv); //Start YUV 2 RGB
    sdkStopTimer(&timer);

    printf("YUV to RGB conversion time(GPU): %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);  

    // Is Conversion ok? Note, there can be small differencein pixel values due to different rounding mechanisms
    printf("Is my YUV to GPU correct? %d\n", confirm_gpu_yuv2rgb(img_obuf_rgb, img_in));  


    write_yuv(img_obuf_yuv, "out_yuv_gpu.yuv");
    write_ppm(img_obuf_rgb, "out_rgb_gpu.ppm");
    
    free_ppm(img_obuf_rgb); //Uncomment these when the images exist
    free_yuv(img_obuf_yuv);

}


void run_cpu_color_test(PPM_IMG img_in)
{
    StopWatchInterface *timer=NULL;
    PPM_IMG img_obuf_rgb;
    YUV_IMG img_obuf_yuv;
    
    
    printf("Starting CPU processing...\n");
  
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    img_obuf_yuv = rgb2yuv(img_in); //Start RGB 2 YUV

    sdkStopTimer(&timer);
    printf("RGB to YUV conversion time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

   

    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    img_obuf_rgb = yuv2rgb(img_obuf_yuv); //Start YUV 2 RGB

    sdkStopTimer(&timer);
    printf("YUV to RGB conversion time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);    


    write_yuv(img_obuf_yuv, "out_yuv_cpu.yuv");
    write_ppm(img_obuf_rgb, "out_rgb_cpu.ppm");
    
    free_ppm(img_obuf_rgb);
    free_yuv(img_obuf_yuv);
    
}

// There could be a minor differente and it is OK
//https://stackoverflow.com/questions/14406364/different-results-for-cuda-addition-on-host-and-on-gpu
bool confirm_gpu_rgb2yuv(YUV_IMG img_gpu, PPM_IMG img) //Place code here that verifies your conversion
{
    YUV_IMG img_cpu = rgb2yuv(img);
    int max_diff = 0;
    
    for(int i=0; i<img_gpu.w*img_gpu.h; i++){
    	 if((int)img_gpu.img_y[i] != (int)img_cpu.img_y[i]){
            	if(abs((int)img_gpu.img_y[i] - (int)img_cpu.img_y[i]) > max_diff) {
            		max_diff = abs((int)img_gpu.img_y[i] - (int)img_cpu.img_y[i]);
            	}
             
            }
         if((int)img_gpu.img_u[i] != (int)img_cpu.img_u[i]){
            	if(abs((int)img_gpu.img_u[i] - (int)img_cpu.img_u[i]) > max_diff) {
            		max_diff = abs((int)img_gpu.img_u[i] - (int)img_cpu.img_u[i]);
            	}
             
            }
         if((int)img_gpu.img_v[i] != (int)img_cpu.img_v[i]){
            	if(abs((int)img_gpu.img_v[i] - (int)img_cpu.img_v[i]) > max_diff) {
            		max_diff = abs((int)img_gpu.img_v[i] - (int)img_cpu.img_v[i]);
            	}
             
            }
    }
    if (max_diff > 0) {
    	printf("Maximum difference between pixels is %i\n", max_diff);
    	return false;
    }
    return true;
        
}

// There could be a minor differente and it is OK
//https://stackoverflow.com/questions/14406364/different-results-for-cuda-addition-on-host-and-on-gpu

bool confirm_gpu_yuv2rgb(PPM_IMG img_gpu, PPM_IMG img_in) //Place code here that verifies your conversion
{
	int max_diff = 0;

    for(int i=0; i<img_gpu.w*img_gpu.h; i++){
    	if((int)img_gpu.img_r[i] != (int)img_in.img_r[i]){
            	if(abs((int)img_gpu.img_r[i] - (int)img_in.img_r[i]) >= max_diff) {
            		max_diff = abs((int)img_gpu.img_r[i] - (int)img_in.img_r[i]);
            	}
             
            }
             if((int)img_gpu.img_g[i] != (int)img_in.img_g[i]){
            	if(abs((int)img_gpu.img_g[i] - (int)img_in.img_g[i]) >= max_diff) {
            		max_diff = abs((int)img_gpu.img_g[i] - (int)img_in.img_g[i]);
             }
            }
             if((int)img_gpu.img_b[i] != (int)img_in.img_b[i]){
            	if(abs((int)img_gpu.img_b[i] - (int)img_in.img_b[i]) >= max_diff) {
            		max_diff = abs((int)img_gpu.img_b[i] - (int)img_in.img_b[i]);
             }
           }
    }
    if (max_diff > 0)
    {
    	printf("Maximum difference between pixels is %i\n", max_diff);
    	return false;
    }
    return true;
}




PPM_IMG read_ppm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    char *ibuf;
    PPM_IMG result;
    int v_max, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    /*Skip the magic number*/
    fscanf(in_file, "%s", sbuf);


    //result = malloc(sizeof(PPM_IMG));
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);
    

    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    ibuf         = (char *)malloc(3 * result.w * result.h * sizeof(char));

    
    fread(ibuf,sizeof(unsigned char), 3 * result.w*result.h, in_file);

    for(i = 0; i < result.w*result.h; i ++){
        result.img_r[i] = ibuf[3*i + 0];
        result.img_g[i] = ibuf[3*i + 1];
        result.img_b[i] = ibuf[3*i + 2];
    }
    
    fclose(in_file);
    free(ibuf);
    
    return result;
}

void write_yuv(YUV_IMG img, const char * path){//Output in YUV444 Planar
    FILE * out_file;
    int i;
    

    out_file = fopen(path, "wb");
    fwrite(img.img_y,sizeof(unsigned char), img.w*img.h, out_file);
    fwrite(img.img_u,sizeof(unsigned char), img.w*img.h, out_file);
    fwrite(img.img_v,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}


void write_yuv2(YUV_IMG img, const char * path){ //Output in YUV444 Packed
    FILE * out_file;
    int i;
    
    char * obuf = (char *)malloc(3 * img.w * img.h * sizeof(char));

    for(i = 0; i < img.w*img.h; i ++){
        obuf[3*i + 0] = img.img_y[i];
        obuf[3*i + 1] = img.img_u[i];
        obuf[3*i + 2] = img.img_v[i];
    }

    out_file = fopen(path, "wb");
    fwrite(obuf,sizeof(unsigned char), 3*img.w*img.h, out_file);
    fclose(out_file);
    free(obuf);
}


void write_ppm(PPM_IMG img, const char * path){
    FILE * out_file;
    int i;
    
    char * obuf = (char *)malloc(3 * img.w * img.h * sizeof(char));

    for(i = 0; i < img.w*img.h; i ++){
        obuf[3*i + 0] = img.img_r[i];
        obuf[3*i + 1] = img.img_g[i];
        obuf[3*i + 2] = img.img_b[i];
    }
    out_file = fopen(path, "wb");
    fprintf(out_file, "P6\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(obuf,sizeof(unsigned char), 3*img.w*img.h, out_file);
    fclose(out_file);
    free(obuf);
}

void free_yuv(YUV_IMG img)
{
    free(img.img_y);
    free(img.img_u);
    free(img.img_v);
}

void free_ppm(PPM_IMG img)
{
    free(img.img_r);
    free(img.img_g);
    free(img.img_b);
}


