#ifndef _NCC_H_
#define _NCC_H_

#include "main.h"
#include "ACMH.h"

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "torch/extension.h"
#include "torch/torch.h"


int loadSfm(const std::string &sfmfile, int **coord, float **depth);

struct sfm_data
{
    int* coord;
    int coord_num;
    float* depth;

};


class CudaData{
public:
    CudaData();
    ~CudaData();
    //only for test on dtu
    void writeDepth(int idx);
    void writeCost(int idx);
    //

    int num_images;

    std::vector<float*> images_host_vec;
    std::vector<float*> depths_host_vec;
    std::vector<float4*> normdepths_host_vec;
    std::vector<float*> costs_host_vec;
    std::vector<float4*> planeHypos_host_vec;
    std::vector<Camera> cameras_host_vec;
    cudaTextureObjects texture_objects_host;

    cudaArray *cuArray[MAX_IMAGES];
    cudaArray *cuDepthArray[MAX_IMAGES];
    Camera *cameras_cuda;
    cudaTextureObjects *texture_objects_cuda;
    float *costs_cuda;
    float4 *plane_hypotheses_cuda;

    PatchMatchParams params;
    int gpu_warps = 32;

    bool usage_flag = false;

private:
    int counter = 0;
};

class ResultSaver{
public:
/**
 * @brief Construct a new Result Saver object to save all checkerboard-related intermediate results.
 * 
 */
    ResultSaver();
    ~ResultSaver();

    // void getDepthAndNormal_coord(int *coord, int coord_num, int height,int width);
    void writeCostTest();
    void writeDepthTest();
    void writeFloatChannelTest(int type_n);

    //

    bool data_flag = false;
    float *ncc_cu;
    float4 *hypos_cu;
    int coord_num;
    int count = 0;

    bool backp_flag = false;
    float *backp_geo_cu;
    float *backp_ncc_cu;
};

class nccInstance{
public:
    nccInstance();
    ~nccInstance();

    void HostInitialization(const Problem &problem, float *ncc_grayImgs, std::vector<Camera> &ncc_cameras, float* ncc_costs_host, float4* ncc_planeHypos_host, float4* ncc_normdepths_host,int *coord, int coord_num_input);
    void DeviceInitialization();
    void Run();
    void getDepthAndNormal();

    int num_images;
    int *coordmap_host;
    int coord_num;
    std::vector<float*> images_host_vec;
    std::vector<float*> depths_host_vec;
    std::vector<float4*> normdepths_host_vec;
    std::vector<float*> costs_host_vec;
    std::vector<float4*> planeHypos_host_vec;
    std::vector<Camera> cameras_host_vec;
    cudaTextureObjects texture_objects_host;
    cudaTextureObjects texture_depths_host;
    PatchMatchParams params;
    int gpu_warps = 32;


    int *coordmap_cuda;
    Camera *cameras_cuda;
    cudaArray *cuArray[MAX_IMAGES];
    cudaArray *cuDepthArray[MAX_IMAGES];
    cudaTextureObjects *texture_objects_cuda;
    cudaTextureObjects *texture_depths_cuda;
    float4 *plane_hypotheses_cuda;
    float *costs_cuda;
    curandState *rand_states_cuda;
    unsigned int *selected_views_cuda;
    float *depths_cuda;
    float4 *normdepths_cuda;
};

class cu_nccInstance{
public:
    cu_nccInstance();
    ~cu_nccInstance();

    void Initialization(ResultSaver &saver, CudaData &cuData, const Problem &problem, int *coord,int coord_num_input);
    void cuRun();

    int instance_w;
    int instance_h;
    int *instance_coordmap_host;
    int instance_coord_num;
    Problem instance_nccproblem;
    PatchMatchParams instance_params;


    int *instance_coordmap_cuda;
    Camera *instance_cameras_cuda;
    cudaArray *instance_cuArray[MAX_IMAGES];
    cudaArray *instance_cuDepthArray[MAX_IMAGES];
    cudaTextureObjects *instance_texture_objects_cuda;
    cudaTextureObjects *instance_texture_depths_cuda;
    float4 *instance_plane_hypotheses_cuda;
    float *instance_costs_cuda;
    curandState *instance_rand_states_cuda;
    unsigned int *instance_selected_views_cuda;
    float *instance_depths_cuda;
    float4 *instance_normdepths_cuda;
    float *instance_nccboard_cuda_nofree;
    float4 *instance_hyposboard_cuda_nofree;

    float *instance_backgeo_cuda_nofree;
    float *instance_backncc_cuda_nofree;


    int gpu_warps = 32; //128
};


class cu_sfmInstance{
public:
    cu_sfmInstance();
    ~cu_sfmInstance();

    void Initialization(CudaData &cuData, const Problem &problem, const sfm_data &cu_sfmData);
    // void cuRun();

    int instance_w;
    int instance_h;
    int *instance_coordmap_host;
    int instance_coord_num;
    Problem instance_nccproblem;
    PatchMatchParams instance_params;


    float *instance_coorddepth_cuda;
    int *instance_coordmap_cuda;
    Camera *instance_cameras_cuda;
    cudaArray *instance_cuArray[MAX_IMAGES];
    cudaArray *instance_cuDepthArray[MAX_IMAGES];
    cudaTextureObjects *instance_texture_objects_cuda;
    cudaTextureObjects *instance_texture_depths_cuda;
    float4 *instance_plane_hypotheses_cuda;
    float *instance_costs_cuda;
    curandState *instance_rand_states_cuda;
    unsigned int *instance_selected_views_cuda;
    float *instance_depths_cuda;
    float4 *instance_normdepths_cuda;
    float *instance_nccboard_cuda_nofree;
    float4 *instance_hyposboard_cuda_nofree;

    float *result_cost_cuda;
    float4 *result_hypos_cuda;


    int gpu_warps = 32;
};



class NCCmodule{
public:
    NCCmodule();
    ~NCCmodule();
    int byte_int = int(sizeof(int));
    int byte_float = int(sizeof(float));
    int *fakecoord;
    int fake_coord_num;
    int *ncc_coord;
    int ncc_coord_num;
    bool fakecoord_flag = false;
    bool fakecamnp_flag = false;
    bool fakepythoninput_flag = false;

    void dataToCuda();
    void set_hw(int height,int width);
    void get_hw();
    void load_pair(const std::string &pairtxt);
    void ProcessNcc(int img_id,int *coord, int coord_num);
    void ProcessCuNcc(int img_id,int *coord, int coord_num);
    void fakeProcessNcc(int img_id);
    void fakePythonInput(const std::string &dense_folder);
    void fakeCamNy();
    void fakeallcoord();
    void genCamFromNp();
    void init(float *py_cameras, float *py_grayImgs, float4 *py_normdepths, float *py_costs, float4 *py_planeHypos);
    void set_cameras_np(float *py_cameras);
    void set_ncc_grayImgs_host(float *py_grayImgs);
    void set_ncc_planeHypos_host(float4 *py_planeHypos);
    void set_ncc_normdepths_host(float4 *py_normdepths);
    void set_ncc_costs_host(float *py_costs);
    void set_init_flag(bool flag);
    void set_normdepths_flag(bool flag);
    void set_back_propagation_flag(bool flag);
    void write_cuData_depth(int idx);
    void write_cuData_cost(int idx);
    void ProcessDtuAll(int img_id);
    void __version__();

    //data accessor
    torch::Tensor bind_ncc_cu();
    torch::Tensor bind_hypos_cu();
    torch::Tensor bind_back_geo();
    torch::Tensor bind_back_ncc();
    torch::Tensor bind_checkerboard_ncc_cu();
    torch::Tensor bind_checkerboard_hypos_cu();
    torch::Tensor hypos_w2c(int idx);
    

    void saverncc_test();
    //
    
    int w = 0;
    int h = 0;
    int num_images;

    CudaData cuData;
    ResultSaver saver;
    bool init_flag = false;
    bool normdepths_flag = false;
    bool back_propagation_flag = false;
    float *cameras_np; //n*2*4*4 (K with dmin\interval\layernum\dmax,T)
    std::vector<Problem> problems;
    float4 *ncc_normdepths_host;
    float *ncc_costs_host;
    float4 *ncc_planeHypos_host;
    float *ncc_grayImgs_host;
    std::vector<Camera> ncc_cameras;

    // sfm module
    void ProcessCuSfm(const std::string &sfmtxt,int img_id);

private:
    float4 *normdepths_buffer;
    bool normdepths_buffer_flag = false;
};



#endif