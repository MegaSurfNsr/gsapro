#ifndef _ACMH_H_
#define _ACMH_H_

#include "main.h"

int readDepthDmb(const std::string file_path, cv::Mat_<float> &depth);
int readNormalDmb(const std::string file_path, cv::Mat_<cv::Vec3f> &normal);
int writeDepthDmb(const std::string file_path, const cv::Mat_<float> depth);
int writeNormalDmb(const std::string file_path, const cv::Mat_<cv::Vec3f> normal);

Camera ReadCamera(const std::string &cam_path);
void  RescaleImageAndCamera(cv::Mat_<cv::Vec3b> &src, cv::Mat_<cv::Vec3b> &dst, cv::Mat_<float> &depth, Camera &camera);
float3 Get3DPointonWorld(const int x, const int y, const float depth, const Camera camera);
void ProjectonCamera(const float3 PointX, const Camera camera, float2 &point, float &depth);
float GetAngle(const cv::Vec3f &v1, const cv::Vec3f &v2);
void StoreColorPlyFileBinaryPointCloud (const std::string &plyFilePath, const std::vector<PointList> &pc);

#define CUDA_SAFE_CALL(error) CudaSafeCall(error, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR() CudaCheckError(__FILE__, __LINE__)

void CudaSafeCall(const cudaError_t error, const std::string& file, const int line);
void CudaCheckError(const char* file, const int line);

struct cudaTextureObjects {
    cudaTextureObject_t images[MAX_IMAGES];
};

struct PatchMatchParams {
    int max_iterations = 3;
    int patch_size = 11;
    int num_images = 5;
    int max_image_size=3200;
    int radius_increment = 2;
    float sigma_spatial = 5.0f;
    float sigma_color = 3.0f;
    int top_k = 4;
    float baseline = 0.54f;
    float depth_min = 0.0f;
    float depth_max = 1.0f;
    float disparity_min = 0.0f;
    float disparity_max = 1.0f;
    float prpgtn_ncc_threshold_back = 0.4f;
    int prpgtn_radius_back = 7;
    float prpgtn_geo_threshold_back = 0.01f;
    float prpgtn_ncc_percent_threshold_back = 0.1f;
    int radius_arm = 25;

    bool geom_consistency = false;
    bool random_init = false;
    bool return_normdepth = false;
    bool return_boardrecord = true;
    bool back_propagation = false;
};

class ACMH {
public:
    ACMH();
    ~ACMH();

    void InuputInitialization(const std::string &dense_folder, const Problem &problem);
    void Colmap2MVS(const std::string &dense_folder, std::vector<Problem> &problems);
    void CudaSpaceInitialization(const std::string &dense_folder, const Problem &problem);
    void RunPatchMatch();
    void SetGeomConsistencyParams();
    int GetReferenceImageWidth();
    int GetReferenceImageHeight();
    cv::Mat GetReferenceImage();
    float4 GetPlaneHypothesis(const int index);
    float GetCost(const int index);
private:
    int num_images;
    int *coordmap_cuda;
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> depths;
    std::vector<Camera> cameras;
    cudaTextureObjects texture_objects_host;
    cudaTextureObjects texture_depths_host;
    float4 *plane_hypotheses_host;
    float *costs_host;
    PatchMatchParams params;

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
};

//cuda function
__device__  void sort_small(float *d, const int n);
__device__ void sort_small_weighted(float *d, float *w, int n);
__device__ int FindMinCostIndex(const float *costs, const int n);
__device__  void setBit(unsigned int &input, const unsigned int n);
__device__  int isSet(unsigned int input, const unsigned int n);
__device__ void Mat33DotVec3(const float mat[9], const float4 vec, float4 *result);
__device__ float Vec3DotVec3(const float4 vec1, const float4 vec2);
__device__ void NormalizeVec3 (float4 *vec);
__device__ void TransformPDFToCDF(float* probs, const int num_probs);
__device__ void Get3DPoint(const Camera camera, const int2 p, const float depth, float *X);
__device__ float4 GetViewDirection(const Camera camera, const int2 p, const float depth);
__device__ float GetDistance2Origin(const Camera camera, const int2 p, const float depth, const float4 normal);
__device__ float ComputeDepthfromPlaneHypothesis(const Camera camera, const float4 plane_hypothesis, const int2 p);
__device__ float4 GenerateRandomNormal(const Camera camera, const int2 p, curandState *rand_state, const float depth);
__device__ float4 GeneratePerturbedNormal(const Camera camera, const int2 p, const float4 normal, curandState *rand_state, const float perturbation);
__device__ float4 GenerateRandomPlaneHypothesis(const Camera camera, const int2 p, curandState *rand_state, const float depth_min, const float depth_max);
__device__ float4 GeneratePertubedPlaneHypothesis(const Camera camera, const int2 p, curandState *rand_state, const float perturbation, const float4 plane_hypothesis_now, const float depth_now, const float depth_min, const float depth_max);
__device__ void ComputeHomography(const Camera ref_camera, const Camera src_camera, const float4 plane_hypothesis, float *H);
__device__ float2 ComputeCorrespondingPoint(const float *H, const int2 p);
__device__ float4 TransformNormal(const Camera camera, float4 plane_hypothesis);
__device__ float4 TransformNormal2RefCam(const Camera camera, float4 plane_hypothesis);
__device__ float ComputeBilateralWeight(const float x_dist, const float y_dist, const float pix, const float center_pix, const float sigma_spatial, const float sigma_color);
__device__ float ComputeBilateralNCC(const cudaTextureObject_t ref_image, const Camera ref_camera, const cudaTextureObject_t src_image, const Camera src_camera, const int2 p, const float4 plane_hypothesis, const PatchMatchParams params);
__device__ float ComputeMultiViewInitialCostandSelectedViews(const cudaTextureObject_t *images, const Camera *cameras, const int2 p, const float4 plane_hypothesis, unsigned int *selected_views, const PatchMatchParams params);
__device__ void ComputeMultiViewCostVector(const cudaTextureObject_t *images, const Camera *cameras, const int2 p, const float4 plane_hypothesis, float *cost_vector, const PatchMatchParams params);
__device__ float3 Get3DPointonWorld_cu(const float x, const float y, const float depth, const Camera camera);
__device__ void ProjectonCamera_cu(const float3 PointX, const Camera camera, float2 &point, float &depth);
__device__ float ComputeGeomConsistencyCost(const cudaTextureObject_t depth_image, const Camera ref_camera, const Camera src_camera, const float4 plane_hypothesis, const int2 p);
__global__ void RandomInitialization(cudaTextureObjects *texture_objects, Camera *cameras, float4 *plane_hypotheses, float *costs, curandState *rand_states, unsigned int *selected_views, const PatchMatchParams params);
__device__ void PlaneHypothesisRefinement(const cudaTextureObject_t *images, const cudaTextureObject_t *depth_images, const Camera *cameras, float4 *plane_hypothesis, float *depth, float *cost, curandState *rand_state, const float *view_weights, const float weight_norm, const int2 p, const PatchMatchParams params);
__device__ void CheckerboardPropagation(const cudaTextureObject_t *images, const cudaTextureObject_t *depths, const Camera *cameras, float4 *plane_hypotheses, float *costs, curandState *rand_states, unsigned int *selected_views, const int2 p, const PatchMatchParams params, const int iter);
__global__ void BlackPixelUpdate(cudaTextureObjects *texture_objects, cudaTextureObjects *texture_depths, Camera *cameras, float4 *plane_hypotheses, float *costs, curandState *rand_states, unsigned int *selected_views, const PatchMatchParams params, const int iter);
__global__ void AllPixelUpdate(cudaTextureObjects *texture_objects, cudaTextureObjects *texture_depths, Camera *cameras, float4 *plane_hypotheses, float *costs, curandState *rand_states, unsigned int *selected_views, const PatchMatchParams params, const int iter,const int* coordmap_cuda);
__global__ void SelectedPixelUpdate(cudaTextureObjects *texture_objects, cudaTextureObjects *texture_depths, Camera *cameras, float4 *plane_hypotheses, float *costs, curandState *rand_states, unsigned int *selected_views, const PatchMatchParams params, const int iter,const int* coordmap_cuda,const int coord_num);
__global__ void RedPixelUpdate(cudaTextureObjects *texture_objects, cudaTextureObjects *texture_depths, Camera *cameras, float4 *plane_hypotheses, float *costs, curandState *rand_states, unsigned int *selected_views, const PatchMatchParams params, const int iter);
__global__ void GetDepthandNormal(Camera *cameras, float4 *plane_hypotheses, const PatchMatchParams params);
__device__ void CheckerboardFilter(const Camera *cameras, float4 *plane_hypotheses, float *costs, const int2 p);
__global__ void BlackPixelFilter(const Camera *cameras, float4 *plane_hypotheses, float *costs);
__global__ void RedPixelFilter(const Camera *cameras, float4 *plane_hypotheses, float *costs);
__device__ void cuCheckerboardPropagation(const cudaTextureObject_t *images, const cudaTextureObject_t *depths, const Camera *cameras, float4 *plane_hypotheses, float *costs, curandState *rand_states, unsigned int *selected_views, const int2 p, const PatchMatchParams params, const int iter,float4* rechypos, float* recncc, int coordidx);
__global__ void cuSelectedPixelUpdate(cudaTextureObjects *texture_objects, cudaTextureObjects *texture_depths, Camera *cameras, float4 *plane_hypotheses, float *costs, curandState *rand_states, unsigned int *selected_views, const PatchMatchParams params, const int iter,const int* coordmap_cuda,const int coord_num,float4* rechypos, float* recncc);
__global__ void cuGetDepthandNormal(Camera *cameras_cuda, float4 *plane_hypotheses_cuda, int *coord_cuda, int coord_num, int hyposPerCoord);
__global__ void cuSelectPixelInitialization(cudaTextureObjects *texture_objects, Camera *cameras, float4 *plane_hypotheses, float *costs, curandState *rand_states, unsigned int *selected_views, const PatchMatchParams params, const int* coordmap_cuda,const int coord_num);
// __global__ void AllPixelFilter(const Camera *cameras, float4 *plane_hypotheses, float *costs,const int* coordmap_cuda,const int coord_num);
__global__ void sfm_RandomInit(cudaTextureObjects *texture_objects, Camera *cameras, float4 *result_hypos_cuda, float *result_cost_cuda, curandState *rand_states, unsigned int *selected_views, const PatchMatchParams params,int* instance_coordmap_cuda, float* instance_coorddepth_cuda, int instance_coord_num);
__device__ float4 sfm_GenerateRandomPlaneHypothesis(const Camera camera, const int2 p, curandState *rand_state, const float depth_min, const float depth_max,int2 p_sfm, float depth_sfm);
__global__ void sfm_init(cudaTextureObjects *texture_objects, Camera *cameras, float4 *plane_hypotheses, float *costs, curandState *rand_states, unsigned int *selected_views, const PatchMatchParams params, bool flag_estCost);
__device__ void cuBackCheckerboardPropagation(const cudaTextureObject_t *images, const cudaTextureObject_t *depths, const Camera *cameras, float4 *plane_hypotheses, float *costs, curandState *rand_states, unsigned int *selected_views, const int2 p, const PatchMatchParams params, const int iter,float* recgeo, float* recncc, int coordidx);
__global__ void cuBackSelectedPixelUpdate(cudaTextureObjects *texture_objects, cudaTextureObjects *texture_depths, Camera *cameras, float4 *plane_hypotheses, float *costs, curandState *rand_states, unsigned int *selected_views, const PatchMatchParams params, const int iter,const int* coordmap_cuda,const int coord_num,float* recgeo, float* recncc);
__global__ void accessNeighbors_cu(cudaTextureObjects *texture_objects, cudaTextureObjects *texture_depths, Camera *cameras, float4 *plane_hypotheses, float *costs, curandState *rand_states, unsigned int *selected_views, const PatchMatchParams params, const int iter,const int* coordmap_cuda,const int coord_num,float* recgeo, float* recncc,float* temtest);
__device__ void cuCheckerboardPropagation_v2(const cudaTextureObject_t *images, const cudaTextureObject_t *depths, const Camera *cameras, float4 *plane_hypotheses, float *costs, curandState *rand_states, unsigned int *selected_views, const int2 p, const PatchMatchParams params, const int iter,float4* rechypos, float* recncc, int coordidx);

//

#endif // _ACMH_H_
