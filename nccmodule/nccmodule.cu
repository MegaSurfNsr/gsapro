#include "nccmodule.h"


int loadSfm(const std::string &sfmfile, int **coord, float **depth){
    int num;
    std::ifstream file(sfmfile);
    file >> num;

    *coord = new int[num*2];
    *depth = new float[num];

    for (int i = 0; i < num; ++i) {
        file >> *(*coord + i*2);
        file >> *(*coord + i*2 + 1);
        file >> *(*depth + i);
    }
    // for (int i = 0; i < 53; ++i) {
    //     std::cout<< i << " " << *(coord + i*2) << " " << *(coord + i*2 + 1)<< " " << *(depth + i) <<std::endl;
    // }
    std::cout<<"loading sfm points: "<< num <<std::endl;
    return num;
}

NCCmodule::NCCmodule(){
    std::cout << "--NccModule: NCC Module Management" << std::endl;
    std::cout << "--NccModule: The int format: int"<< sizeof(int)*8<< std::endl;
    std::cout << "--NccModule: The float format: float"<< sizeof(float)*8<< std::endl;
    std::cout << "--NccModule: Please Make Sure the system data format is specified in a valid format!"<< std::endl;
    std::cout << "--NccModule: Please Make Sure the size of images is correct!"<< std::endl;
    std::cout << "--NccModule: Note that insufficient gpu memory may not be warned!"<< std::endl;
    // std::cout << "--NccModule: Please Make Sure the image_id is started from 0!"<< std::endl;
}

NCCmodule::~NCCmodule(){
    if(fakepythoninput_flag){
        delete[] ncc_grayImgs_host;
        delete[] ncc_costs_host;
        delete[] ncc_planeHypos_host;
    }
    if(fakecoord_flag){
        delete[] fakecoord;
    }
    if(fakecamnp_flag){
        delete[] cameras_np;
    }
    if(normdepths_buffer_flag){
        cudaFree(normdepths_buffer);
    }
    std::cout<<"destroy NCCmodule"<<std::endl;
}

void NCCmodule::set_hw(int height,int width){
    w = width;
    h = height;
}

void NCCmodule::get_hw(){
    std::cout<<"height: "<< h <<", width: "<< w <<std::endl;
}

void NCCmodule::load_pair(const std::string &pairtxt){
    problems.clear();
    std::ifstream file(pairtxt);
    file >> num_images;
    for (int i = 0; i < num_images; ++i) {
        Problem problem;
        problem.src_image_ids.clear();
        file >> problem.ref_image_id;
        int num_src_images;
        file >> num_src_images;
        for (int j = 0; j < num_src_images; ++j) {
            int id;
            float score;
            file >> id >> score;
            if (score <= 0.0f) {
                continue;
            }
            problem.src_image_ids.push_back(id);
        }
        problems.push_back(problem);
    }
    std::cout<<"loading pairs: "<< problems.size()<<std::endl;
}

void NCCmodule::fakePythonInput(const std::string &dense_folder){
    fakepythoninput_flag = true;
    ncc_grayImgs_host = new float[num_images * h * w];
    ncc_cameras.clear();

    std::string image_folder = dense_folder + std::string("/images");
    std::string cam_folder = dense_folder + std::string("/cams");


    for(int i=0;i<num_images;i++){
        std::stringstream image_path;
        image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << i << ".jpg";
        cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
        cv::Mat image_float;
        image_uint.convertTo(image_float, CV_32FC1);
        memcpy(ncc_grayImgs_host + i * w * h,image_float.data,sizeof(float)*w*h);

        std::stringstream cam_path;
        cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << i << "_cam.txt";
        Camera camera = ReadCamera(cam_path.str());
        camera.height = image_float.rows;
        camera.width = image_float.cols;
        ncc_cameras.push_back(camera);
    }

    ncc_planeHypos_host = new float4[num_images*h*w];
    ncc_costs_host = new float[num_images*h*w];

}

void NCCmodule::fakeallcoord(){
    fakecoord = new int[h*w*2];
    fakecoord_flag = true;
    int cn = 0;
    for (int i = 0; i < h; i++){
        for (int j = 0; j < w; j++){
            *(fakecoord+2*cn) = i;
            *(fakecoord+2*cn+1) = j;
            cn = cn + 1;
        }
    }
    fake_coord_num = h*w;
    // // debug use only
    // for(int n =0;n<10;n++){
    //     std::cout<<"check the last "<< n <<" coord: "<<*(fakecoord+(h*w-10 + n)*2) <<","<<*(fakecoord+(h*w-10 + n)*2+1)<<std::endl;
    // }

}

void NCCmodule::fakeCamNy(){
    cameras_np = new float[num_images*2*4*4];
    fakecamnp_flag = true;
    Camera temcam;
    for(int i=0;i<num_images;i++){
        temcam = ncc_cameras[i];
        *(cameras_np + i * 32 + 0) = temcam.K[0];
        *(cameras_np + i * 32 + 1) = temcam.K[1];
        *(cameras_np + i * 32 + 2) = temcam.K[2];
        *(cameras_np + i * 32 + 3) = 0;
        *(cameras_np + i * 32 + 4) = temcam.K[3];
        *(cameras_np + i * 32 + 5) = temcam.K[4];
        *(cameras_np + i * 32 + 6) = temcam.K[5];
        *(cameras_np + i * 32 + 7) = 0;
        *(cameras_np + i * 32 + 8) = temcam.K[6];
        *(cameras_np + i * 32 + 9) = temcam.K[7];
        *(cameras_np + i * 32 + 10) = temcam.K[8];
        *(cameras_np + i * 32 + 11) = 0;
        *(cameras_np + i * 32 + 12) = temcam.depth_min;
        *(cameras_np + i * 32 + 13) = 2.5;
        *(cameras_np + i * 32 + 14) = 256;
        *(cameras_np + i * 32 + 15) = temcam.depth_max;

        *(cameras_np + i * 32 + 16) = temcam.R[0];
        *(cameras_np + i * 32 + 17) = temcam.R[1];
        *(cameras_np + i * 32 + 18) = temcam.R[2];
        *(cameras_np + i * 32 + 19) = temcam.t[0];
        *(cameras_np + i * 32 + 20) = temcam.R[3];
        *(cameras_np + i * 32 + 21) = temcam.R[4];
        *(cameras_np + i * 32 + 22) = temcam.R[5];
        *(cameras_np + i * 32 + 23) = temcam.t[1];;
        *(cameras_np + i * 32 + 24) = temcam.R[6];
        *(cameras_np + i * 32 + 25) = temcam.R[7];
        *(cameras_np + i * 32 + 26) = temcam.R[8];
        *(cameras_np + i * 32 + 27) = temcam.t[2];;
        *(cameras_np + i * 32 + 28) = 0;
        *(cameras_np + i * 32 + 29) = 0;
        *(cameras_np + i * 32 + 30) = 0;
        *(cameras_np + i * 32 + 31) = 1;
    }
}

void NCCmodule::genCamFromNp(){
    ncc_cameras.clear();

    for(int i=0;i<num_images;i++){
        Camera temcam;

        temcam.K[0] = *(cameras_np + i * 32 + 0);
        temcam.K[1] = *(cameras_np + i * 32 + 1);
        temcam.K[2] = *(cameras_np + i * 32 + 2);
        temcam.K[3] = *(cameras_np + i * 32 + 4);
        temcam.K[4] = *(cameras_np + i * 32 + 5);
        temcam.K[5] = *(cameras_np + i * 32 + 6);
        temcam.K[6] = *(cameras_np + i * 32 + 8);
        temcam.K[7] = *(cameras_np + i * 32 + 9);
        temcam.K[8] = *(cameras_np + i * 32 + 10);
        temcam.depth_min = *(cameras_np + i * 32 + 12);
        temcam.depth_max = *(cameras_np + i * 32 + 15);
        
        temcam.R[0] = *(cameras_np + i * 32 + 16);
        temcam.R[1] = *(cameras_np + i * 32 + 17);
        temcam.R[2] = *(cameras_np + i * 32 + 18);
        temcam.t[0] = *(cameras_np + i * 32 + 19);
        temcam.R[3] = *(cameras_np + i * 32 + 20);
        temcam.R[4] = *(cameras_np + i * 32 + 21);
        temcam.R[5] = *(cameras_np + i * 32 + 22);
        temcam.t[1] = *(cameras_np + i * 32 + 23);
        temcam.R[6] = *(cameras_np + i * 32 + 24);
        temcam.R[7] = *(cameras_np + i * 32 + 25);
        temcam.R[8] = *(cameras_np + i * 32 + 26);
        temcam.t[2] = *(cameras_np + i * 32 + 27);

        temcam.height = h;
        temcam.width = w;
        ncc_cameras.push_back(temcam);
    }

}

void NCCmodule::init(float *py_cameras, float *py_grayImgs, float4 *py_normdepths, float *py_costs, float4 *py_planeHypos){
    ncc_grayImgs_host = py_grayImgs;
    cameras_np = py_cameras;
    ncc_normdepths_host = py_normdepths;
    ncc_costs_host = py_costs;
    ncc_planeHypos_host = py_planeHypos;
}
void NCCmodule::set_init_flag(bool flag){
    init_flag = flag;
}
void NCCmodule::set_normdepths_flag(bool flag){
    normdepths_flag = flag;
}

void NCCmodule::set_back_propagation_flag(bool flag){
    back_propagation_flag = flag;
}

void NCCmodule::set_cameras_np(float *py_cameras){
    cameras_np = py_cameras;
}
void NCCmodule::set_ncc_grayImgs_host(float *py_grayImgs){
    ncc_grayImgs_host = py_grayImgs;
}
void NCCmodule::set_ncc_planeHypos_host(float4 *py_planeHypos){
    ncc_planeHypos_host = py_planeHypos;
}
void NCCmodule::set_ncc_normdepths_host(float4 *py_normdepths){
    ncc_normdepths_host = py_normdepths;
}
void NCCmodule::set_ncc_costs_host(float *py_costs){
    ncc_costs_host = py_costs;
}

void NCCmodule::write_cuData_depth(int idx){
    cuData.writeDepth(idx);
}
void NCCmodule::write_cuData_cost(int idx){
    cuData.writeCost(idx);
}

void NCCmodule::ProcessDtuAll(int img_id){
    int *coord = new int[1200*1600*2];
    coordDim2(coord,1200,1600);
    coordshuffle(coord,1200*1600);
    ProcessCuNcc(img_id,coord, 1200*1600);
}


//void NCCmodule::ProcessProblem(int img_id, int *coord)
void NCCmodule::ProcessNcc(int img_id, int *coord, int coord_num){
    // cout here debug use only
    // std::cout<<"generate nccInstance"<<std::endl;
    nccInstance ncc;
    ncc.params.random_init = init_flag;
    ncc.params.return_normdepth = normdepths_flag;
    // std::cout<<"HostInitialization"<<std::endl;
    ncc.HostInitialization(problems[img_id],ncc_grayImgs_host,ncc_cameras, ncc_costs_host, ncc_planeHypos_host, ncc_normdepths_host,coord, coord_num);
    // std::cout<<"DeviceInitialization"<<std::endl;
    // sleep(1);
    // ncc.DeviceInitialization();
    // std::cout<<"Run"<<std::endl;
    ncc.Run();
    // std::cout<<"Run end"<<std::endl;
    if(ncc.params.return_normdepth){
        ncc.getDepthAndNormal();
    }
}

void NCCmodule::fakeProcessNcc(int img_id){
    // cout here debug use only
    std::cout<<"generate nccInstance"<<std::endl;
    nccInstance ncc;
    ncc.params.random_init = init_flag;
    ncc.params.return_normdepth = normdepths_flag;
    std::cout<<"HostInitialization"<<std::endl;
    ncc.HostInitialization(problems[img_id],ncc_grayImgs_host,ncc_cameras, ncc_costs_host, ncc_planeHypos_host, ncc_normdepths_host,fakecoord, fake_coord_num);//fake_coord_num
    // std::cout<<"DeviceInitialization"<<std::endl;
    ncc.DeviceInitialization();
    std::cout<<"Run"<<std::endl;
    ncc.Run();
    if(ncc.params.return_normdepth){
        ncc.getDepthAndNormal();
    }

}

void NCCmodule::ProcessCuNcc(int img_id, int *coord, int coord_num){
    // cout here debug use only
    clock_t start,end,mid;
    start = clock();
    // std::cout<<"generate nccInstance"<<std::endl;
    cu_nccInstance cu_ncc;
    cu_ncc.instance_params.random_init = init_flag;
    cu_ncc.instance_params.return_normdepth = normdepths_flag;
    cu_ncc.instance_params.back_propagation = back_propagation_flag;
    // std::cout<<"HostInitialization"<<std::endl;
    cu_ncc.Initialization(saver,cuData,problems[img_id],coord, coord_num);
    // write_cuData_depth(img_id);
    // write_cuData_cost(img_id);
    // std::cout<<"depth min "<<    cu_ncc.instance_params.depth_min <<std::endl;
    // std::cout<<"depth max "<<    cu_ncc.instance_params.depth_max <<std::endl;
    // cuData.writeDepth(img_id);
    // cuData.writeCost(img_id);
    // std::cout<<" saver ncc at: "<<saver.ncc_cu<<std::endl;
    // std::cout<<"DeviceInitialization"<<std::endl;
    // sleep(1);
    // ncc.DeviceInitialization();
    // std::cout<<"Run"<<std::endl;
    mid = clock();
    // cuData.writeCost(0);
    // cuData.writeDepth(0);
    cu_ncc.cuRun();
    // cuData.writeCost(0);
    // cuData.writeDepth(0);
    end = clock();
    // write_cuData_depth(img_id);
    // write_cuData_cost(img_id);

    // // std::cout<<"Run end"<<std::endl;
    // if(ncc.params.return_normdepth){
    //     ncc.getDepthAndNormal();
    // }

    std::cout<<"init time: "<<double(mid-start)/CLOCKS_PER_SEC<<"s" <<std::endl;
    std::cout<<"running time: "<<double(end-mid)/CLOCKS_PER_SEC<<"s" <<std::endl;
}

void NCCmodule::dataToCuda(){
    cuData.usage_flag = true;
    cuData.num_images = num_images;
    cuData.params.random_init = init_flag;
    cuData.params.return_normdepth = normdepths_flag;
    int height = ncc_cameras[problems[0].ref_image_id].height;
    int width = ncc_cameras[problems[0].ref_image_id].width;

    for (int i = 0; i < num_images; ++i) {
        cuData.images_host_vec.push_back(ncc_grayImgs_host + height*width*i);
        cuData.cameras_host_vec.push_back(ncc_cameras[i]);
        cuData.costs_host_vec.push_back(ncc_costs_host + height*width*i);
        cuData.planeHypos_host_vec.push_back(ncc_planeHypos_host + height*width*i);
        cuData.normdepths_host_vec.push_back(ncc_normdepths_host + height*width*i);
    }


    for (int i = 0; i < num_images; ++i) {
        int rows = height;
        int cols = width;

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat); // number of bits of each component,here is a float32 channel
        cudaMallocArray(&cuData.cuArray[i], &channelDesc, cols, rows);
        cudaMemcpy2DToArray (cuData.cuArray[i], 0, 0, cuData.images_host_vec[i], cols*sizeof(float), cols*sizeof(float), rows, cudaMemcpyHostToDevice);
        // for 2D: step[0] bytes of one row (1600*4), step[1] bytes of one element (4). I think this step[0] always equals to cols*sizeof float.
        // std::cout<< "step:"<<images[i].step[0]<< "  cols size:"<< cols*sizeof(float) <<std::endl;

        struct cudaResourceDesc resDesc; // remove struct here in c++ is ok
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuData.cuArray[i];

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap; //只在标准化坐标系下有效，当越界时向下取整，if >2?。
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode  = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        cudaCreateTextureObject(&(cuData.texture_objects_host.images[i]), &resDesc, &texDesc, NULL);
    }

    cudaMalloc((void**)&cuData.texture_objects_cuda, sizeof(cudaTextureObjects)); //和类型在一起是引用，和变量在一起是取址
    cudaMemcpy(cuData.texture_objects_cuda, &cuData.texture_objects_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);
    // std::cout<<&cuData.texture_objects_host<<std::endl;
    // std::cout<<&cuData.texture_objects_host.images[0]<<std::endl;

    cudaMalloc((void**)&cuData.cameras_cuda, sizeof(Camera) * (num_images));
    cudaMemcpy(cuData.cameras_cuda, &cuData.cameras_host_vec[0], sizeof(Camera) * (num_images), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&cuData.plane_hypotheses_cuda, sizeof(float4) * (height * width)*num_images);
    cudaMalloc((void**)&cuData.costs_cuda, sizeof(float) * (height * width)*num_images);
    cudaMemset((void*)cuData.plane_hypotheses_cuda, 0, sizeof(float4) * (height * width)*num_images);
    cudaMemset((void*)cuData.costs_cuda, 0, sizeof(float) * (height * width)*num_images);
    // cudaMalloc((void**)&rand_states_cuda, sizeof(curandState) * (height * width));
    // cudaMalloc((void**)&selected_views_cuda, sizeof(unsigned int) * (height * width));
    // cudaMalloc((void**)&depths_cuda, sizeof(float) * (height * width)*num_images);

    if(!init_flag){
        int shift = 0;
        for(int i=0;i<=num_images;i++){
            shift = sizeof(float4) * height * width *i;
            cudaMemcpy(cuData.plane_hypotheses_cuda+shift, cuData.planeHypos_host_vec[0],  sizeof(float4) * height * width, cudaMemcpyHostToDevice);
            cudaMemcpy(cuData.costs_cuda+shift, cuData.costs_host_vec[0],  sizeof(float4) * height * width, cudaMemcpyHostToDevice);
        }
        std::cout<<"transfer hypos"<<std::endl;
    }
    std::cout<<"data has transfered to the device"<<std::endl;
}

void NCCmodule::ProcessCuSfm(const std::string &sfmtxt,int img_id){
    std::cout<<sfmtxt<<std::endl;
    std::ifstream file(sfmtxt);
    sfm_data sfmData;
    file >> sfmData.coord_num;
    sfmData.coord = new int[sfmData.coord_num*2];
    sfmData.depth  = new float[sfmData.coord_num];
    for (int i = 0; i < sfmData.coord_num; ++i) {
        file >> *(sfmData.coord + i*2);
        file >> *(sfmData.coord + i*2 + 1);
        file >> *(sfmData.depth + i);
    }

    // for (int i = 0; i < 100; ++i) {
    //     std::cout<< i << " " << *(sfmData.coord + i*2) << " " << *(sfmData.coord + i*2 + 1)<< " " << *(sfmData.depth + i) <<std::endl;
    // }

    std::cout<<"num sfm points:"<< sfmData.coord_num <<std::endl;

    sfm_data cu_sfmData;
    cu_sfmData.coord_num = sfmData.coord_num;
    cudaMalloc((void**)&cu_sfmData.coord, sizeof(int) * (sfmData.coord_num*2));
    cudaMemcpy(cu_sfmData.coord, sfmData.coord, sizeof(int) * (sfmData.coord_num*2), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&cu_sfmData.depth, sizeof(int) * sfmData.coord_num);
    cudaMemcpy(cu_sfmData.depth, sfmData.depth, sizeof(int) * sfmData.coord_num, cudaMemcpyHostToDevice);

    // process region
    clock_t start,end;
    start = clock();
    cu_sfmInstance sfmInstance;
    sfmInstance.Initialization(cuData,problems[img_id],cu_sfmData);
    end = clock();
    std::cout<<"sfm time: "<<double(end-start)/CLOCKS_PER_SEC<<"s"<<std::endl;

    //

    cudaFree(cu_sfmData.coord);
    cudaFree(cu_sfmData.depth);

    std::cout<<"sfm process done!"<<std::endl;

}


CudaData::CudaData(){}
CudaData::~CudaData(){
    if(usage_flag){
        for (int i = 0; i < num_images; ++i) {
        cudaDestroyTextureObject(texture_objects_host.images[i]);
        cudaFreeArray(cuArray[i]);
        }
        cudaFree(texture_objects_cuda);
        cudaFree(cameras_cuda);
        cudaFree(plane_hypotheses_cuda);
        cudaFree(costs_cuda);
        std::cout<<"destroy cudaData"<<std::endl;
    }

}

nccInstance::nccInstance(){}
nccInstance::~nccInstance(){
    // delete[] plane_hypotheses_host;
    // delete[] costs_host;

    for (int i = 0; i < num_images; ++i) {
        cudaDestroyTextureObject(texture_objects_host.images[i]);
        cudaFreeArray(cuArray[i]);
    }
    cudaFree(texture_objects_cuda);
    cudaFree(cameras_cuda);
    cudaFree(plane_hypotheses_cuda);
    cudaFree(costs_cuda);
    cudaFree(rand_states_cuda);
    cudaFree(selected_views_cuda);
    cudaFree(depths_cuda);
    cudaFree(coordmap_cuda);
    // std::cout<<"destroy nccInstance"<<std::endl;
}

void nccInstance::HostInitialization(const Problem &problem, float *ncc_grayImgs_host, std::vector<Camera> &ncc_cameras,float* ncc_costs_host, float4* ncc_planeHypos_host,float4* ncc_normdepths_host,int *coord,int coord_num_input){
    coordmap_host = coord;
    coord_num = coord_num_input;

    int height = ncc_cameras[problem.ref_image_id].height;
    int width = ncc_cameras[problem.ref_image_id].width;
    images_host_vec.push_back(ncc_grayImgs_host + height*width*problem.ref_image_id);
    cameras_host_vec.push_back(ncc_cameras[problem.ref_image_id]);
    planeHypos_host_vec.push_back(ncc_planeHypos_host + height*width*problem.ref_image_id);
    costs_host_vec.push_back(ncc_costs_host + height*width*problem.ref_image_id);
    normdepths_host_vec.push_back(ncc_normdepths_host);

    size_t num_src_images = problem.src_image_ids.size();
    for (size_t i = 0; i < num_src_images; ++i) {
        images_host_vec.push_back(ncc_grayImgs_host + height*width*problem.src_image_ids[i]);
        cameras_host_vec.push_back(ncc_cameras[problem.src_image_ids[i]]);
        costs_host_vec.push_back(ncc_costs_host + height*width*problem.src_image_ids[i]);
        planeHypos_host_vec.push_back(ncc_planeHypos_host + height*width*problem.src_image_ids[i]);
        normdepths_host_vec.push_back(ncc_normdepths_host + height*width*problem.src_image_ids[i]);
    }

    params.depth_min = cameras_host_vec[0].depth_min * 0.6f;
    params.depth_max = cameras_host_vec[0].depth_max * 1.2f;
    params.num_images = (int)images_host_vec.size();
    // assume fx=fy
    params.disparity_min = cameras_host_vec[0].K[0] * params.baseline / params.depth_max;
    params.disparity_max = cameras_host_vec[0].K[0] * params.baseline / params.depth_min;


    // debug use only
    // std::cout<<"K matrix:"<<std::endl;
    // std::cout.precision(6);
    // std::cout<<cameras_host_vec[0].K[0] << " " <<cameras_host_vec[0].K[1] << " " <<cameras_host_vec[0].K[2]<<std::endl;

    // std::cout<<cameras_host_vec[0].K[3] << " " <<cameras_host_vec[0].K[4] << " "<<cameras_host_vec[0].K[5]<<std::endl;

    // std::cout<<cameras_host_vec[0].K[6] << " " <<cameras_host_vec[0].K[7] << " " <<cameras_host_vec[0].K[8]<<std::endl;

    // std::cout<<"T matrix:"<<std::endl;

    // std::cout<<cameras_host_vec[0].R[0] << " " <<cameras_host_vec[0].R[1] << " " <<cameras_host_vec[0].R[2] << " "<<cameras_host_vec[0].t[0] <<std::endl;

    // std::cout<<cameras_host_vec[0].R[3] << " " <<cameras_host_vec[0].R[4] << " " <<cameras_host_vec[0].R[5] << " "<<cameras_host_vec[0].t[1] <<std::endl;

    // std::cout<<cameras_host_vec[0].R[6] << " " <<cameras_host_vec[0].R[7] << " " <<cameras_host_vec[0].R[8] << " "<<cameras_host_vec[0].t[2] <<std::endl;


    // std::cout<<"depth_min: "<< params.depth_min <<std::endl;
    // std::cout<<"depth_max: "<< params.depth_max <<std::endl;
    // std::cout<<"num_images: "<< params.num_images <<std::endl;

    // std::stringstream refimg;
    // refimg << "/home/yswang/Downloads/test/refimg.dmb" ;
    // std::string result_folder = refimg.str();  
    // cv::Mat_<float> img = cv::Mat::zeros(height, width, CV_32FC1);
    // for (int col = 0; col < width; ++col) {
    //     for (int row = 0; row < height; ++row) {
    //         int center = row * width + col;
    //         img(row, col) = images[0][center];
    //     }
    // }
    // writeDepthDmb(result_folder, img);
    // std::cout<<"testhere"<<std::endl;
    nccInstance::DeviceInitialization();

}

void nccInstance::DeviceInitialization(){
    num_images = (int)images_host_vec.size();

    for (int i = 0; i < num_images; ++i) {
        int rows = cameras_host_vec[i].height;
        int cols = cameras_host_vec[i].width;

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat); // number of bits of each component,here is a float32 channel
        cudaMallocArray(&cuArray[i], &channelDesc, cols, rows);
        cudaMemcpy2DToArray (cuArray[i], 0, 0, images_host_vec[i], cols*sizeof(float), cols*sizeof(float), rows, cudaMemcpyHostToDevice);
        // for 2D: step[0] bytes of one row (1600*4), step[1] bytes of one element (4). I think this step[0] always equals to cols*sizeof float.
        // std::cout<< "step:"<<images[i].step[0]<< "  cols size:"<< cols*sizeof(float) <<std::endl;


        struct cudaResourceDesc resDesc; // remove struct here in c++ is ok
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[i];

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap; //只在标准化坐标系下有效，当越界时向下取整，if >2?。
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode  = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        cudaCreateTextureObject(&(texture_objects_host.images[i]), &resDesc, &texDesc, NULL);
    }
    cudaMalloc((void**)&texture_objects_cuda, sizeof(cudaTextureObjects)); //和类型在一起是引用，和变量在一起是取址
    cudaMemcpy(texture_objects_cuda, &texture_objects_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&cameras_cuda, sizeof(Camera) * (num_images));
    cudaMemcpy(cameras_cuda, &cameras_host_vec[0], sizeof(Camera) * (num_images), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&plane_hypotheses_cuda, sizeof(float4) * (cameras_host_vec[0].height * cameras_host_vec[0].width));

    cudaMalloc((void**)&costs_cuda, sizeof(float) * (cameras_host_vec[0].height * cameras_host_vec[0].width));
    cudaMalloc((void**)&rand_states_cuda, sizeof(curandState) * (cameras_host_vec[0].height * cameras_host_vec[0].width));
    cudaMalloc((void**)&selected_views_cuda, sizeof(unsigned int) * (cameras_host_vec[0].height * cameras_host_vec[0].width));
    cudaMalloc((void**)&depths_cuda, sizeof(float) * (cameras_host_vec[0].height * cameras_host_vec[0].width));

    cudaMalloc((void**)&coordmap_cuda, sizeof(int) * (coord_num *2));
    cudaMemcpy(coordmap_cuda, coordmap_host,  sizeof(int) * (coord_num*2), cudaMemcpyHostToDevice);

    if(params.random_init){
        dim3 grid_size_randinit;
        grid_size_randinit.x = (cameras_host_vec[0].width + 16 - 1) / 16; // 100 75 1
        grid_size_randinit.y=(cameras_host_vec[0].height + 16 - 1) / 16;
        grid_size_randinit.z = 1;
        dim3 block_size_randinit; // 16 16 1
        block_size_randinit.x = 16;
        block_size_randinit.y = 16;
        block_size_randinit.z = 1;
        // std::cout<<"random_init"<<std::endl;
        RandomInitialization<<<grid_size_randinit, block_size_randinit>>>(texture_objects_cuda, cameras_cuda, plane_hypotheses_cuda, costs_cuda, rand_states_cuda, selected_views_cuda, params);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        // std::cout<<"end random_init"<<std::endl;
    }
    else{
        // std::cout<<"transfer plane & cost to cuda"<<std::endl;
        cudaMemcpy(plane_hypotheses_cuda, planeHypos_host_vec[0],  sizeof(float4) * cameras_host_vec[0].height * cameras_host_vec[0].width, cudaMemcpyHostToDevice);
        cudaMemcpy(costs_cuda, costs_host_vec[0],  sizeof(float) * cameras_host_vec[0].height * cameras_host_vec[0].width, cudaMemcpyHostToDevice);
    }
}


void nccInstance::getDepthAndNormal(){
    const int width = cameras_host_vec[0].width;
    const int height = cameras_host_vec[0].height;
    // idx.x:height, idx.y:width
    // coord_num = 2048;
    int BLOCK_W = 32;
    int BLOCK_H = (BLOCK_W / 2);

    dim3 block_size_bunck;
    block_size_bunck.x = gpu_warps;
    block_size_bunck.y = 1;
    block_size_bunck.z = 1;

    dim3 grid_size_bunck;
    grid_size_bunck.x = (coord_num + gpu_warps -1) / gpu_warps;
    grid_size_bunck.y = 1;
    grid_size_bunck.z = 1;

    dim3 grid_size_randinit;
    grid_size_randinit.x = (width + 16 - 1) / 16; // 100 75 1
    grid_size_randinit.y=(height + 16 - 1) / 16;
    grid_size_randinit.z = 1;
    dim3 block_size_randinit; // 16 16 1
    block_size_randinit.x = 16;
    block_size_randinit.y = 16;
    block_size_randinit.z = 1;

    dim3 grid_size_checkerboard; // 50 38 1
    grid_size_checkerboard.x = (width + BLOCK_W - 1) / BLOCK_W;
    grid_size_checkerboard.y= ( (height / 2) + BLOCK_H - 1) / BLOCK_H;
    grid_size_checkerboard.z = 1;
    dim3 block_size_checkerboard; // 32 16 1
    block_size_checkerboard.x = BLOCK_W;
    block_size_checkerboard.y = BLOCK_H;
    block_size_checkerboard.z = 1;

    // std::cout<<"depth normal test:"<<std::endl;
    GetDepthandNormal<<<grid_size_randinit, block_size_randinit>>>(cameras_cuda, plane_hypotheses_cuda, params);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    // std::cout<<"calculation done"<<std::endl;
    cudaMemcpy(normdepths_host_vec[0], plane_hypotheses_cuda, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
    // std::cout<<"transfer done"<<std::endl;

}

void nccInstance::Run(){
    const int width = cameras_host_vec[0].width;
    const int height = cameras_host_vec[0].height;
    // idx.x:height, idx.y:width
    // coord_num = 2048;
    int BLOCK_W = 32;
    int BLOCK_H = (BLOCK_W / 2);

    dim3 block_size_bunck;
    block_size_bunck.x = gpu_warps;
    block_size_bunck.y = 1;
    block_size_bunck.z = 1;

    dim3 grid_size_bunck;
    grid_size_bunck.x = (coord_num + gpu_warps -1) / gpu_warps;
    grid_size_bunck.y = 1;
    grid_size_bunck.z = 1;

    dim3 grid_size_randinit;
    grid_size_randinit.x = (width + 16 - 1) / 16; // 100 75 1
    grid_size_randinit.y=(height + 16 - 1) / 16;
    grid_size_randinit.z = 1;
    dim3 block_size_randinit; // 16 16 1
    block_size_randinit.x = 16;
    block_size_randinit.y = 16;
    block_size_randinit.z = 1;

    dim3 grid_size_checkerboard; // 50 38 1
    grid_size_checkerboard.x = (width + BLOCK_W - 1) / BLOCK_W;
    grid_size_checkerboard.y= ( (height / 2) + BLOCK_H - 1) / BLOCK_H;
    grid_size_checkerboard.z = 1;
    dim3 block_size_checkerboard; // 32 16 1
    block_size_checkerboard.x = BLOCK_W;
    block_size_checkerboard.y = BLOCK_H;
    block_size_checkerboard.z = 1;

    int max_iterations = params.max_iterations;

    // clock_t start,end,mid;
    // start = clock();

    // // debug use only
    // for (int i = 0; i < max_iterations; ++i) {
    //     // only for test
    //     std::stringstream costpath;
    //     costpath << "/home/yswang/Downloads/test/cost_"<<i<<".dmb" ;
    //     std::string result_folder = costpath.str();  
    //     cv::Mat_<float> costs = cv::Mat::zeros(height, width, CV_32FC1);
    //     cudaMemcpy(costs_host_vec[0], costs_cuda, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
    //     CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //     for (int col = 0; col < width; ++col) {
    //         for (int row = 0; row < height; ++row) {
    //             int center = row * width + col;
    //             costs(row, col) = costs_host_vec[0][center];
    //         }
    //     }
    //     writeDepthDmb(result_folder, costs);
    //     printf("iteration begin: %d\n", i);
    //     SelectedPixelUpdate<<<grid_size_bunck, block_size_bunck>>>(texture_objects_cuda, texture_depths_cuda, cameras_cuda, plane_hypotheses_cuda, costs_cuda, rand_states_cuda, selected_views_cuda, params, i,coordmap_cuda,coord_num);
    //     CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //     printf("iteration done: %d\n", i);
    // }
    // //

    // debug use only
    // std::stringstream costpath;
    // costpath << "/home/yswang/Downloads/test/cost_before.dmb" ;
    // std::string result_folder = costpath.str();  
    // cv::Mat_<float> costs = cv::Mat::zeros(height, width, CV_32FC1);
    // cudaMemcpy(costs_host_vec[0], costs_cuda, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

    // for (int col = 0; col < width; ++col) {
    //     for (int row = 0; row < height; ++row) {
    //         int center = row * width + col;
    //         costs(row, col) = costs_host_vec[0][center];
    //     }
    // }
    // writeDepthDmb(result_folder, costs);
    //

    // the influence of iter here need to be investigate. warning: iter here controls the cost threshold
    SelectedPixelUpdate<<<grid_size_bunck, block_size_bunck>>>(texture_objects_cuda, texture_depths_cuda, cameras_cuda, plane_hypotheses_cuda, costs_cuda, rand_states_cuda, selected_views_cuda, params, 0,coordmap_cuda,coord_num);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // debug use only
    // std::stringstream costpath2;
    // costpath2 << "/home/yswang/Downloads/test/cost_after.dmb" ;
    // std::string result_folder2 = costpath2.str();  
    // cv::Mat_<float> costs2 = cv::Mat::zeros(height, width, CV_32FC1);
    // cudaMemcpy(costs_host_vec[0], costs_cuda, sizeof(float) * width * height, cudaMemcpyDeviceToHost);

    // for (int col = 0; col < width; ++col) {
    //     for (int row = 0; row < height; ++row) {
    //         int center = row * width + col;
    //         costs2(row, col) = costs_host_vec[0][center];
    //     }
    // }
    // writeDepthDmb(result_folder2, costs2);
    //


    // mid = clock();

    // // transfer the data per pixel
    // int location_shift = 0;

    // for(int i=0;i<coord_num;i++){
    //     location_shift = width * (*(coordmap_host + i *2)) + (*(coordmap_host + i *2 +1));
    //     cudaMemcpy(planeHypos_host+location_shift, plane_hypotheses_cuda+location_shift, sizeof(float4), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(costs_host+location_shift, costs_cuda+location_shift, sizeof(float), cudaMemcpyDeviceToHost);
    // }

    // cudaMemcpy(planeHypos_host, plane_hypotheses_cuda, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
    cudaMemcpy(costs_host_vec[0], costs_cuda, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
    cudaMemcpy(planeHypos_host_vec[0], plane_hypotheses_cuda, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);


    // end = clock();
    // std::cout<<"time1 = "<<double(end-mid)/CLOCKS_PER_SEC<<"s , time2= " << double(mid-start)/CLOCKS_PER_SEC <<std::endl;
}


void cu_nccInstance::Initialization(ResultSaver &saver, CudaData &cuData, const Problem &problem, int *coord,int coord_num_input){
    saver.coord_num = coord_num_input;
    instance_coordmap_host = coord;
    instance_coord_num = coord_num_input;
    instance_nccproblem = problem;
    int height = cuData.cameras_host_vec[problem.ref_image_id].height;
    int width = cuData.cameras_host_vec[problem.ref_image_id].width;

    instance_h = height;
    instance_w = width;

    // size_t num_src_images = problem.src_image_ids.size();
    // for (size_t i = 0; i < num_src_images; ++i) {
    //     images_host_vec.push_back(ncc_grayImgs_host + height*width*problem.src_image_ids[i]);
    //     cameras_host_vec.push_back(ncc_cameras[problem.src_image_ids[i]]);
    //     costs_host_vec.push_back(ncc_costs_host + height*width*problem.src_image_ids[i]);
    //     planeHypos_host_vec.push_back(ncc_planeHypos_host + height*width*problem.src_image_ids[i]);
    //     normdepths_host_vec.push_back(ncc_normdepths_host + height*width*problem.src_image_ids[i]);
    // }

    instance_params.depth_min = cuData.cameras_host_vec[problem.ref_image_id].depth_min * 0.95f;
    instance_params.depth_max = cuData.cameras_host_vec[problem.ref_image_id].depth_max * 1.05f;

    instance_params.num_images = (int)problem.src_image_ids.size() +1;
    // assume fx=fy
    instance_params.disparity_min = cuData.cameras_host_vec[problem.ref_image_id].K[0] * instance_params.baseline / instance_params.depth_max;
    instance_params.disparity_max = cuData.cameras_host_vec[problem.ref_image_id].K[0] * instance_params.baseline / instance_params.depth_min;


    //  // debug use only
    // std::cout<<"K matrix:"<<std::endl;
    // std::cout.precision(6);
    // std::cout<<cuData.cameras_host_vec[problem.ref_image_id].K[0] << " " <<cuData.cameras_host_vec[problem.ref_image_id].K[1] << " " <<cuData.cameras_host_vec[problem.ref_image_id].K[2]<<std::endl;
    // std::cout<<cuData.cameras_host_vec[problem.ref_image_id].K[3] << " " <<cuData.cameras_host_vec[problem.ref_image_id].K[4] << " "<<cuData.cameras_host_vec[problem.ref_image_id].K[5]<<std::endl;
    // std::cout<<cuData.cameras_host_vec[problem.ref_image_id].K[6] << " " <<cuData.cameras_host_vec[problem.ref_image_id].K[7] << " " <<cuData.cameras_host_vec[problem.ref_image_id].K[8]<<std::endl;
    // std::cout<<"T matrix:"<<std::endl;
    // std::cout<<cuData.cameras_host_vec[problem.ref_image_id].R[0] << " " <<cuData.cameras_host_vec[problem.ref_image_id].R[1] << " " <<cuData.cameras_host_vec[problem.ref_image_id].R[2] << " "<<cuData.cameras_host_vec[problem.ref_image_id].t[0] <<std::endl;
    // std::cout<<cuData.cameras_host_vec[problem.ref_image_id].R[3] << " " <<cuData.cameras_host_vec[problem.ref_image_id].R[4] << " " <<cuData.cameras_host_vec[problem.ref_image_id].R[5] << " "<<cuData.cameras_host_vec[problem.ref_image_id].t[1] <<std::endl;
    // std::cout<<cuData.cameras_host_vec[problem.ref_image_id].R[6] << " " <<cuData.cameras_host_vec[problem.ref_image_id].R[7] << " " <<cuData.cameras_host_vec[problem.ref_image_id].R[8] << " "<<cuData.cameras_host_vec[problem.ref_image_id].t[2] <<std::endl;
    // std::cout<<"depth_min: "<< instance_params.depth_min <<std::endl;
    // std::cout<<"depth_max: "<< instance_params.depth_max <<std::endl;
    // std::cout<<"num_images: "<< instance_params.num_images <<std::endl;
    // std::stringstream refimg;
    // refimg << "/home/yswang/Downloads/test/refimg.dmb" ;
    // std::string result_folder = refimg.str();  
    // cv::Mat_<float> img = cv::Mat::zeros(height, width, CV_32FC1);
    // for (int col = 0; col < width; ++col) {
    //     for (int row = 0; row < height; ++row) {
    //         int center = row * width + col;
    //         img(row, col) = images[0][center];
    //     }
    // }
    // writeDepthDmb(result_folder, img);
    // std::cout<<"testhere"<<std::endl;
    //  // debug use only


    // // Device initialization:

    // clock_t t1,t2,t3,t4,t5,t6,t7,t8;
    int num_images = instance_params.num_images;
    
    if(saver.data_flag){
        cudaFree(saver.hypos_cu);
        cudaFree(saver.ncc_cu);
    }
    if(saver.backp_flag){
        cudaFree(saver.backp_geo_cu);
        cudaFree(saver.backp_ncc_cu);
    }

    // t1 = clock();
    cudaMalloc((void**)&instance_coordmap_cuda, sizeof(int) * (instance_coord_num *2));
    cudaMemcpy(instance_coordmap_cuda, instance_coordmap_host,  sizeof(int) * (instance_coord_num*2), cudaMemcpyHostToDevice);
    // t2 = clock();
    cudaMalloc((void**)&instance_texture_objects_cuda, sizeof(cudaTextureObjects));
    cudaMemcpy((void**)&instance_texture_objects_cuda->images[0],&cuData.texture_objects_cuda->images[problem.ref_image_id],sizeof(cuData.texture_objects_cuda->images[problem.ref_image_id]),cudaMemcpyDeviceToDevice);
    // t3 = clock();
    cudaMalloc((void**)&instance_cameras_cuda, sizeof(Camera) * (num_images));
    cudaMemcpy((void**)&(instance_cameras_cuda[0]),cuData.cameras_cuda + problem.ref_image_id,sizeof(Camera),cudaMemcpyDeviceToDevice);
    // t4 = clock();
    instance_plane_hypotheses_cuda = cuData.plane_hypotheses_cuda + (problem.ref_image_id * height * width);
    instance_costs_cuda = cuData.costs_cuda + (problem.ref_image_id * height * width);

    size_t num_src_images = problem.src_image_ids.size();
    for (size_t i = 0; i < num_src_images; ++i) {
        cudaMemcpy((void**)&instance_texture_objects_cuda->images[i+1],&cuData.texture_objects_cuda->images[problem.src_image_ids[i]],sizeof(cuData.texture_objects_cuda->images[problem.ref_image_id]),cudaMemcpyDeviceToDevice);
        cudaMemcpy((void**)&instance_cameras_cuda[i+1],cuData.cameras_cuda + problem.src_image_ids[i],sizeof(Camera),cudaMemcpyDeviceToDevice);
    }
    // t5 = clock();
    cudaMalloc((void**)&instance_rand_states_cuda, sizeof(curandState) * (height * width));
    cudaMalloc((void**)&instance_selected_views_cuda, sizeof(unsigned int) * (height * width));
    cudaMalloc((void**)&instance_depths_cuda, sizeof(float) * (height * width));

    cudaMalloc((void**)&saver.ncc_cu,sizeof(float)*9*instance_coord_num);
    cudaMalloc((void**)&saver.hypos_cu,sizeof(float4)*9*instance_coord_num);

    // saver.ncc_cu;
    // saver.hypos_cu;
    instance_nccboard_cuda_nofree = saver.ncc_cu;
    instance_hyposboard_cuda_nofree = saver.hypos_cu;
    saver.data_flag = true;
    saver.count += 1;
    // std::cout<<"saver count: "<<saver.count <<std::endl;

    // t6 = clock();
    if(instance_params.random_init){
        dim3 grid_size_randinit;
        grid_size_randinit.x = (width + 16 - 1) / 16; // 100 75 1
        grid_size_randinit.y=(height + 16 - 1) / 16;
        grid_size_randinit.z = 1;
        dim3 block_size_randinit; // 16 16 1
        block_size_randinit.x = 16;
        block_size_randinit.y = 16;
        block_size_randinit.z = 1;
        std::cout<<"random_init"<<std::endl;
        RandomInitialization<<<grid_size_randinit, block_size_randinit>>>(instance_texture_objects_cuda, instance_cameras_cuda, instance_plane_hypotheses_cuda, instance_costs_cuda, instance_rand_states_cuda, instance_selected_views_cuda, instance_params);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());

        // std::cout<<"end random_init"<<std::endl;
    }
    else{
        dim3 grid_size_bunck;
        grid_size_bunck.x = (instance_coord_num + gpu_warps -1) / gpu_warps;
        grid_size_bunck.y = 1;
        grid_size_bunck.z = 1;
        dim3 block_size_bunck;
        block_size_bunck.x = gpu_warps;
        block_size_bunck.y = 1;
        block_size_bunck.z = 1;
        cuSelectPixelInitialization<<<grid_size_bunck, block_size_bunck>>>(instance_texture_objects_cuda, instance_cameras_cuda, instance_plane_hypotheses_cuda, instance_costs_cuda, instance_rand_states_cuda, instance_selected_views_cuda, instance_params, instance_coordmap_cuda,instance_coord_num);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    if(instance_params.back_propagation){
        std::cout<< "back prop on!" <<std::endl;
        saver.backp_flag = true;
        cudaMalloc((void**)&saver.backp_geo_cu,sizeof(float)*instance_coord_num);
        cudaMalloc((void**)&saver.backp_ncc_cu,sizeof(float)*instance_coord_num);
        instance_backgeo_cuda_nofree = saver.backp_geo_cu;
        instance_backncc_cuda_nofree = saver.backp_ncc_cu;
    }
    // t7 = clock();
    // std::cout<<"init time consumption: "<< std::endl 
    // << double(t2-t1)/CLOCKS_PER_SEC<< std::endl
    // << double(t3-t2)/CLOCKS_PER_SEC<< std::endl
    // << double(t4-t3)/CLOCKS_PER_SEC<< std::endl
    // << double(t5-t4)/CLOCKS_PER_SEC<< std::endl
    // << double(t6-t5)/CLOCKS_PER_SEC<< std::endl
    // << double(t7-t6)/CLOCKS_PER_SEC<< std::endl;
}

void cu_nccInstance::cuRun(){
    // idx.x:height, idx.y:width
    // coord_num = 2048;
    int BLOCK_W = 32;
    int BLOCK_H = (BLOCK_W / 2);

    dim3 block_size_bunck;
    block_size_bunck.x = gpu_warps;
    block_size_bunck.y = 1;
    block_size_bunck.z = 1;

    dim3 grid_size_bunck;
    grid_size_bunck.x = (instance_coord_num + gpu_warps -1) / gpu_warps;
    grid_size_bunck.y = 1;
    grid_size_bunck.z = 1;

    dim3 grid_size_randinit;
    grid_size_randinit.x = (instance_w + 16 - 1) / 16; // 100 75 1
    grid_size_randinit.y=(instance_h + 16 - 1) / 16;
    grid_size_randinit.z = 1;
    dim3 block_size_randinit; // 16 16 1
    block_size_randinit.x = 16;
    block_size_randinit.y = 16;
    block_size_randinit.z = 1;

    dim3 grid_size_checkerboard; // 50 38 1
    grid_size_checkerboard.x = (instance_w + BLOCK_W - 1) / BLOCK_W;
    grid_size_checkerboard.y= ((instance_h / 2) + BLOCK_H - 1) / BLOCK_H;
    grid_size_checkerboard.z = 1;
    dim3 block_size_checkerboard; // 32 16 1
    block_size_checkerboard.x = BLOCK_W;
    block_size_checkerboard.y = BLOCK_H;
    block_size_checkerboard.z = 1;

    int max_iterations = instance_params.max_iterations;

    clock_t start,end,mid;
    start = clock();

    // // debug use only
    // for (int i = 0; i < max_iterations; ++i) {
    //     // only for test
    //     std::stringstream costpath;
    //     costpath << "/home/yswang/Downloads/test/cost_"<<i<<".dmb" ;
    //     std::string result_folder = costpath.str();  
    //     cv::Mat_<float> costs = cv::Mat::zeros(height, width, CV_32FC1);
    //     cudaMemcpy(costs_host_vec[0], costs_cuda, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
    //     CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //     for (int col = 0; col < width; ++col) {
    //         for (int row = 0; row < height; ++row) {
    //             int center = row * width + col;
    //             costs(row, col) = costs_host_vec[0][center];
    //         }
    //     }
    //     writeDepthDmb(result_folder, costs);
    //     printf("iteration begin: %d\n", i);
    //     SelectedPixelUpdate<<<grid_size_bunck, block_size_bunck>>>(texture_objects_cuda, texture_depths_cuda, cameras_cuda, plane_hypotheses_cuda, costs_cuda, rand_states_cuda, selected_views_cuda, params, i,coordmap_cuda,coord_num);
    //     CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //     printf("iteration done: %d\n", i);
    // }
    // //

    // debug use only
    // std::stringstream costpath;
    // costpath << "/home/yswang/Downloads/test/cost_before.dmb" ;
    // std::string result_folder = costpath.str();  
    // cv::Mat_<float> costs = cv::Mat::zeros(instance_h, instance_w, CV_32FC1);
    // float *cost1 = new float[instance_w * instance_h];
    // cudaMemcpy(cost1, instance_costs_cuda, sizeof(float) * instance_w * instance_h, cudaMemcpyDeviceToHost);

    // for (int col = 0; col < instance_w; ++col) {
    //     for (int row = 0; row < instance_h; ++row) {
    //         int center = row * instance_w + col;
    //         costs(row, col) = cost1[center];
    //     }
    // }
    // writeDepthDmb(result_folder, costs);
    //

    // the influence of iter here need to be investigate. warning: iter here controls the cost threshold
    // std::cout<<"end instance_coord_num:"<< instance_coord_num <<std::endl;

    // clock_t t1,t2;
    // t1 = clock();
    // cuSelectedPixelUpdate<<<1, 10>>>(instance_texture_objects_cuda, instance_texture_depths_cuda, instance_cameras_cuda, instance_plane_hypotheses_cuda, instance_costs_cuda, instance_rand_states_cuda, instance_selected_views_cuda, instance_params, 0,instance_coordmap_cuda,100,instance_hyposboard_cuda_nofree, instance_nccboard_cuda_nofree);
    // CUDA_SAFE_CALL(cudaDeviceSynchronize());
    std::cout<<"ok0 "<< std::endl;

    cuSelectedPixelUpdate<<<grid_size_bunck, block_size_bunck>>>(instance_texture_objects_cuda, instance_texture_depths_cuda, instance_cameras_cuda, instance_plane_hypotheses_cuda, instance_costs_cuda, instance_rand_states_cuda, instance_selected_views_cuda, instance_params, 0,instance_coordmap_cuda,instance_coord_num,instance_hyposboard_cuda_nofree, instance_nccboard_cuda_nofree);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // cuSelectedPixelUpdate<<<grid_size_bunck, block_size_bunck>>>(instance_texture_objects_cuda, instance_texture_depths_cuda, instance_cameras_cuda, instance_plane_hypotheses_cuda, instance_costs_cuda, instance_rand_states_cuda, instance_selected_views_cuda, instance_params, 0,instance_coordmap_cuda,instance_coord_num,instance_hyposboard_cuda_nofree, instance_nccboard_cuda_nofree);
    // CUDA_SAFE_CALL(cudaDeviceSynchronize());
    // t2 = clock();
    // std::cout<<"run time consumption: "<< std::endl 
    // << double(t2-t1)/CLOCKS_PER_SEC<< std::endl;

    // AllPixelFilter<<<grid_size_bunck, block_size_bunck>>>(instance_cameras_cuda, instance_plane_hypotheses_cuda, instance_costs_cuda,instance_coordmap_cuda,instance_coord_num);
    std::cout<<"ok1 "<< std::endl;

    if(instance_params.return_normdepth){
        cuGetDepthandNormal<<<grid_size_bunck, block_size_bunck>>>(instance_cameras_cuda, instance_hyposboard_cuda_nofree, instance_coordmap_cuda, instance_coord_num, 9); //hyposPerCoord
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
    std::cout<<"ok2 "<< std::endl;

    mid = clock();



    
    if (instance_params.back_propagation)
    {
        std::cout<<"back propagation!" <<std::endl;
        cuBackSelectedPixelUpdate<<<grid_size_bunck, block_size_bunck>>>(instance_texture_objects_cuda, instance_texture_depths_cuda, instance_cameras_cuda, instance_plane_hypotheses_cuda, instance_costs_cuda, instance_rand_states_cuda, instance_selected_views_cuda, instance_params, 0,instance_coordmap_cuda,instance_coord_num,instance_backgeo_cuda_nofree, instance_backncc_cuda_nofree);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
    
    // if(false){
    //     std::cout<<"accessNeighbors_cu"<<std::endl;
    //     float* temtest;
    //     cudaMalloc((void**)&temtest, sizeof(float) * (1200*1600));
    //     accessNeighbors_cu<<<grid_size_bunck, block_size_bunck>>>(instance_texture_objects_cuda, instance_texture_depths_cuda, instance_cameras_cuda, instance_plane_hypotheses_cuda, instance_costs_cuda, instance_rand_states_cuda, instance_selected_views_cuda, instance_params, 0,instance_coordmap_cuda,instance_coord_num,instance_backgeo_cuda_nofree, instance_backncc_cuda_nofree,temtest);
    //     CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //         // test only
    //     std::cout<<"only for test use, input must be whole images,1200*1600"<<std::endl;
    //     std::stringstream costpath;
    //     costpath << "./temtestcost.dmb" ;
    //     std::string result_folder = costpath.str();  
    //     int height = 1200;
    //     int width = 1600;
    //     cv::Mat_<float> costs = cv::Mat::zeros(height, width, CV_32FC1);

    //     float *tem = new float[height*width * 9];
    //     cudaMemcpy(tem, temtest, sizeof(float) * 1200*1600, cudaMemcpyDeviceToHost);
    //     CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //     for (int col = 0; col < width; ++col) {
    //         for (int row = 0; row < height; ++row) {
    //             int center = row * width + col;
    //             costs(row, col) = tem[center];
    //         }
    //     }
    //     writeDepthDmb(result_folder, costs);
    //     delete[] tem;
    // }


    end = clock();




    std::cout<<"run forward: "<< double(mid-start)/CLOCKS_PER_SEC<< std::endl << "run backward: "<< double(end-mid)/CLOCKS_PER_SEC<< std::endl;
    // mid = clock();

    // // transfer the data per pixel
    // int location_shift = 0;

    // for(int i=0;i<coord_num;i++){
    //     location_shift = width * (*(coordmap_host + i *2)) + (*(coordmap_host + i *2 +1));
    //     cudaMemcpy(planeHypos_host+location_shift, plane_hypotheses_cuda+location_shift, sizeof(float4), cudaMemcpyDeviceToHost);
    //     cudaMemcpy(costs_host+location_shift, costs_cuda+location_shift, sizeof(float), cudaMemcpyDeviceToHost);
    // }

    // cudaMemcpy(planeHypos_host, plane_hypotheses_cuda, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
}

cu_nccInstance::cu_nccInstance(){}
cu_nccInstance::~cu_nccInstance(){
    // std::cout<<"destroy cu_nccinstance"<<std::endl;
    cudaFree(instance_texture_objects_cuda);
    cudaFree(instance_cameras_cuda);
    cudaFree(instance_rand_states_cuda);
    cudaFree(instance_selected_views_cuda);
    cudaFree(instance_depths_cuda);
    cudaFree(instance_coordmap_cuda);
    // cudaFree(instance_nccboard_cuda);
    // cudaFree(instance_hyposboard_cuda);
}

void CudaData::writeCost(int idx){
    std::cout<<"only for test use, input must be whole images,1200*1600"<<std::endl;
    std::stringstream costpath;
    counter = counter + 1;
    costpath << "./temcost_"<<counter<<".dmb" ;
    std::string result_folder = costpath.str();  
    int height = 1200;
    int width = 1600;
    cv::Mat_<float> costs = cv::Mat::zeros(height, width, CV_32FC1);

    float *tem = new float[height*width];
    cudaMemcpy(tem, costs_cuda + idx * (width * height), sizeof(float) * width * height, cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    for (int col = 0; col < width; ++col) {
        for (int row = 0; row < height; ++row) {
            int center = row * width + col;
            costs(row, col) = tem[center];
        }
    }
    writeDepthDmb(result_folder, costs);
    delete[] tem;
}

void CudaData::writeDepth(int idx){
    std::cout<<"only for test use, input must be whole images,1200*1600"<<std::endl;
    std::stringstream costpath;
    counter = counter + 1;
    costpath << "./temdepth_"<<counter<<".dmb" ;
    std::string result_folder = costpath.str();  
    int height = 1200;
    int width = 1600;
    cv::Mat_<float> costs = cv::Mat::zeros(height, width, CV_32FC1);
    float4 *tem = new float4[height*width];

    float4 *tem_cuda;
    cudaMalloc((void**)&tem_cuda, sizeof(float4) * (height*width));
    cudaMemcpy(tem_cuda, plane_hypotheses_cuda + idx* width * height, sizeof(float4) * width * height, cudaMemcpyDeviceToDevice);

    dim3 grid_size_randinit;
    grid_size_randinit.x = (width + 16 - 1) / 16; // 100 75 1
    grid_size_randinit.y=(height + 16 - 1) / 16;
    grid_size_randinit.z = 1;
    dim3 block_size_randinit; // 16 16 1
    block_size_randinit.x = 16;
    block_size_randinit.y = 16;
    block_size_randinit.z = 1;

    GetDepthandNormal<<<grid_size_randinit, block_size_randinit>>>(cameras_cuda+idx, tem_cuda, params);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    cudaMemcpy(tem, tem_cuda, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    for (int col = 0; col < width; ++col) {
        // std::cout<<tem[(col)*9+6].w<<std::endl;
        for (int row = 0; row < height; ++row) {
            int center = row * width + col;
            costs(row, col) = tem[center].w;
            
        }
    }
    writeDepthDmb(result_folder, costs);
    delete[] tem;
    cudaFree(tem_cuda);
}

void ResultSaver::writeCostTest(){
    std::cout<<"only for test use, input must be whole images,1200*1600"<<std::endl;
    std::stringstream costpath;
    costpath << "/home/yswang/Downloads/test/temcost_"<<count<<"_saver.dmb" ;
    std::string result_folder = costpath.str();  
    int height = 1200;
    int width = 1600;
    cv::Mat_<float> costs = cv::Mat::zeros(height, width, CV_32FC1);

    float *tem = new float[height*width * 9];
    cudaMemcpy(tem, ncc_cu, sizeof(float) * width * height * 9, cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    for (int col = 0; col < width; ++col) {
        for (int row = 0; row < height; ++row) {
            int center = row * width + col;
            costs(row, col) = tem[center*9+8];
        }
    }
    writeDepthDmb(result_folder, costs);
    delete[] tem;
}

void ResultSaver::writeDepthTest(){
    std::cout<<"only for test use, input must be whole images,1200*1600"<<std::endl;
    std::stringstream costpath;
    costpath << "/home/yswang/Downloads/test/temdepth_"<<count<<"_saver.dmb" ;
    std::string result_folder = costpath.str();  
    int height = 1200;
    int width = 1600;
    cv::Mat_<float> costs = cv::Mat::zeros(height, width, CV_32FC1);

    float4 *tem = new float4[height*width * 9];

    

    cudaMemcpy(tem, hypos_cu, sizeof(float4) * width * height * 9, cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    for (int col = 0; col < width; ++col) {
        // std::cout<<tem[(col)*9+6].w<<std::endl;
        for (int row = 0; row < height; ++row) {
            int center = row * width + col;
            costs(row, col) = tem[center*9+8].w;
            
        }
    }
    writeDepthDmb(result_folder, costs);
    delete[] tem;
}


void ResultSaver::writeFloatChannelTest(int type_n){
    
    std::cout<<"only for test use, input must be whole images,1200*1600"<<std::endl;
    std::stringstream costpath;
    int height = 1200;
    int width = 1600;
    cv::Mat_<float> costs = cv::Mat::zeros(height, width, CV_32FC1);
    float *tem = new float[height*width];

    if (type_n==1)
    {
        costpath << "/home/yswang/Downloads/test/backgeo_"<<count<<"_saver.dmb" ;
        cudaMemcpy(tem, backp_geo_cu, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
    }
    if (type_n==2)
    {
        costpath << "/home/yswang/Downloads/test/backncc_"<<count<<"_saver.dmb" ;
        cudaMemcpy(tem, backp_ncc_cu, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    std::string result_folder = costpath.str();  

    for (int col = 0; col < width; ++col) {
        for (int row = 0; row < height; ++row) {
            int center = row * width + col;
            costs(row, col) = tem[center];
        }
    }
    writeDepthDmb(result_folder, costs);
    delete[] tem;
}


ResultSaver::ResultSaver(){}

ResultSaver::~ResultSaver(){
    std::cout<<"saver data flag: "<<data_flag<<std::endl;
    if(data_flag){
        cudaFree(ncc_cu);
        cudaFree(hypos_cu);
    }
    if(backp_flag){
        cudaFree(backp_geo_cu);
        cudaFree(backp_ncc_cu);
    }
    std::cout<<"destroy saver"<<std::endl;
}


void cu_sfmInstance::Initialization(CudaData &cuData, const Problem &problem, const sfm_data &cu_sfmData){

    instance_nccproblem = problem;
    int height = cuData.cameras_host_vec[problem.ref_image_id].height;
    int width = cuData.cameras_host_vec[problem.ref_image_id].width;

    instance_h = height;
    instance_w = width;



    instance_params.depth_min = cuData.cameras_host_vec[problem.ref_image_id].depth_min;
    instance_params.depth_max = cuData.cameras_host_vec[problem.ref_image_id].depth_max;
    instance_params.num_images = (int)problem.src_image_ids.size() +1;

    instance_params.disparity_min = cuData.cameras_host_vec[problem.ref_image_id].K[0] * instance_params.baseline / instance_params.depth_max;
    instance_params.disparity_max = cuData.cameras_host_vec[problem.ref_image_id].K[0] * instance_params.baseline / instance_params.depth_min;


    int num_images = instance_params.num_images;
    
    instance_coordmap_cuda = cu_sfmData.coord;
    instance_coorddepth_cuda = cu_sfmData.depth;
    instance_coord_num = cu_sfmData.coord_num;

    cudaMalloc((void**)&instance_texture_objects_cuda, sizeof(cudaTextureObjects));
    cudaMemcpy((void**)&instance_texture_objects_cuda->images[0],&cuData.texture_objects_cuda->images[problem.ref_image_id],sizeof(cuData.texture_objects_cuda->images[problem.ref_image_id]),cudaMemcpyDeviceToDevice);

    cudaMalloc((void**)&instance_cameras_cuda, sizeof(Camera) * (num_images));
    cudaMemcpy((void**)&(instance_cameras_cuda[0]),cuData.cameras_cuda + problem.ref_image_id,sizeof(Camera),cudaMemcpyDeviceToDevice);

    instance_plane_hypotheses_cuda = cuData.plane_hypotheses_cuda + (problem.ref_image_id * height * width);
    instance_costs_cuda = cuData.costs_cuda + (problem.ref_image_id * height * width);

    size_t num_src_images = problem.src_image_ids.size();
    for (size_t i = 0; i < num_src_images; ++i) {
        cudaMemcpy((void**)&instance_texture_objects_cuda->images[i+1],&cuData.texture_objects_cuda->images[problem.src_image_ids[i]],sizeof(cuData.texture_objects_cuda->images[problem.ref_image_id]),cudaMemcpyDeviceToDevice);
        cudaMemcpy((void**)&instance_cameras_cuda[i+1],cuData.cameras_cuda + problem.src_image_ids[i],sizeof(Camera),cudaMemcpyDeviceToDevice);
    }

    cudaMalloc((void**)&instance_rand_states_cuda, sizeof(curandState) * (height * width));
    cudaMalloc((void**)&instance_selected_views_cuda, sizeof(unsigned int) * (height * width));
    cudaMalloc((void**)&instance_depths_cuda, sizeof(float) * (height * width));

    // cudaMalloc((void**)&result_cost_cuda, sizeof(float) * 8 * 16 * instance_coord_num);
    // cudaMalloc((void**)&result_hypos_cuda, sizeof(float4) * 8 * 16 * instance_coord_num);

    dim3 grid_size_randinit;
    grid_size_randinit.x = (width + 16 - 1) / 16; // 100 75 1
    grid_size_randinit.y=(height + 16 - 1) / 16;
    grid_size_randinit.z = 1;
    dim3 block_size_randinit; // 16 16 1
    block_size_randinit.x = 16;
    block_size_randinit.y = 16;
    block_size_randinit.z = 1;
    std::cout<<"random_init"<<std::endl;
    sfm_init<<<grid_size_randinit, block_size_randinit>>>(instance_texture_objects_cuda, instance_cameras_cuda, instance_plane_hypotheses_cuda, instance_costs_cuda, instance_rand_states_cuda, instance_selected_views_cuda, instance_params,false);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    dim3 grid_size_sfmrandinit;
    grid_size_sfmrandinit.x = cu_sfmData.coord_num;
    grid_size_sfmrandinit.y = 1;
    grid_size_sfmrandinit.z = 1;
    dim3 block_size_sfmrandinit;
    block_size_sfmrandinit.x = 16; 
    block_size_sfmrandinit.y = 16; 
    block_size_sfmrandinit.z = 1;
    sfm_RandomInit<<<grid_size_sfmrandinit,block_size_sfmrandinit>>>(instance_texture_objects_cuda, instance_cameras_cuda, instance_plane_hypotheses_cuda, instance_costs_cuda, instance_rand_states_cuda, instance_selected_views_cuda, instance_params, instance_coordmap_cuda, instance_coorddepth_cuda, instance_coord_num);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    sfm_init<<<grid_size_randinit, block_size_randinit>>>(instance_texture_objects_cuda, instance_cameras_cuda, instance_plane_hypotheses_cuda, instance_costs_cuda, instance_rand_states_cuda, instance_selected_views_cuda, instance_params,true);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    std::cout<<"ver.1"<<std::endl;
    // RandomInitialization<<<grid_size_randinit, block_size_randinit>>>(instance_texture_objects_cuda, instance_cameras_cuda, instance_plane_hypotheses_cuda, instance_costs_cuda, instance_rand_states_cuda, instance_selected_views_cuda, instance_params);

    // if(instance_params.random_init){
    //     dim3 grid_size_randinit;
    //     grid_size_randinit.x = (width + 16 - 1) / 16; // 100 75 1
    //     grid_size_randinit.y=(height + 16 - 1) / 16;
    //     grid_size_randinit.z = 1;
    //     dim3 block_size_randinit; // 16 16 1
    //     block_size_randinit.x = 16;
    //     block_size_randinit.y = 16;
    //     block_size_randinit.z = 1;
    //     std::cout<<"random_init"<<std::endl;
    //     RandomInitialization<<<grid_size_randinit, block_size_randinit>>>(instance_texture_objects_cuda, instance_cameras_cuda, instance_plane_hypotheses_cuda, instance_costs_cuda, instance_rand_states_cuda, instance_selected_views_cuda, instance_params);
    //     CUDA_SAFE_CALL(cudaDeviceSynchronize());
    //     // std::cout<<"end random_init"<<std::endl;
    // }
    // else{
    //     dim3 grid_size_bunck;
    //     grid_size_bunck.x = (instance_coord_num + gpu_warps -1) / gpu_warps;
    //     grid_size_bunck.y = 1;
    //     grid_size_bunck.z = 1;
    //     dim3 block_size_bunck;
    //     block_size_bunck.x = gpu_warps;
    //     block_size_bunck.y = 1;
    //     block_size_bunck.z = 1;
    //     cuSelectPixelInitialization<<<grid_size_bunck, block_size_bunck>>>(instance_texture_objects_cuda, instance_cameras_cuda, instance_plane_hypotheses_cuda, instance_costs_cuda, instance_rand_states_cuda, instance_selected_views_cuda, instance_params, instance_coordmap_cuda,instance_coord_num);
    //     CUDA_SAFE_CALL(cudaDeviceSynchronize());
    // }

}

cu_sfmInstance::cu_sfmInstance(){}
cu_sfmInstance::~cu_sfmInstance(){
    std::cout<<"destroy cu_sfminstance"<<std::endl;
    cudaFree(instance_texture_objects_cuda);
    cudaFree(instance_cameras_cuda);
    cudaFree(instance_rand_states_cuda);
    cudaFree(instance_selected_views_cuda);
    cudaFree(instance_depths_cuda);
    // cudaFree(result_cost_cuda);
    // cudaFree(result_hypos_cuda);
    // cudaFree(instance_coordmap_cuda);
    // cudaFree(instance_nccboard_cuda);
    // cudaFree(instance_hyposboard_cuda);
}













// // // // // // // // // // // // // // // // // // //
// // // // // // // torch test  // // // // // // // //
void NCCmodule::__version__(){
    std::cout<<"NCCmodule ver 2023.7.13"<<std::endl;
}


torch::Tensor NCCmodule::bind_ncc_cu(){
    auto options =
    torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
    return torch::from_blob(cuData.costs_cuda, {num_images,h,w},options);
}
torch::Tensor NCCmodule::bind_hypos_cu(){
    auto options =
    torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
    return torch::from_blob(cuData.plane_hypotheses_cuda, {num_images,h,w,4},options);
}
torch::Tensor NCCmodule::bind_checkerboard_ncc_cu(){
    auto options =
    torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);

    return torch::from_blob(saver.ncc_cu, {saver.coord_num,9},options); //{h,w,9,4}
}
torch::Tensor NCCmodule::bind_checkerboard_hypos_cu(){
    auto options =
    torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
    return torch::from_blob(saver.hypos_cu, {saver.coord_num,9,4},options); //{h,w,9,4}
}

torch::Tensor NCCmodule::bind_back_geo(){
    auto options =
    torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
    return torch::from_blob(saver.backp_geo_cu, {saver.coord_num,1},options);
}

torch::Tensor NCCmodule::bind_back_ncc(){
    auto options =
    torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
    return torch::from_blob(saver.backp_ncc_cu, {saver.coord_num,1},options);
}

// test region
void NCCmodule::saverncc_test(){
    saver.writeCostTest();
}

torch::Tensor NCCmodule::hypos_w2c(int idx){


    int height = h;
    int width = w;
    cv::Mat_<float> costs = cv::Mat::zeros(height, width, CV_32FC1);

    if(!normdepths_buffer_flag){
        cudaMalloc((void**)&normdepths_buffer, sizeof(float4) * (height*width));
        normdepths_buffer_flag = true;
    }

    cudaMemcpy(normdepths_buffer, cuData.plane_hypotheses_cuda + idx* width * height, sizeof(float4) * width * height, cudaMemcpyDeviceToDevice);

    dim3 grid_size_randinit;
    grid_size_randinit.x = (width + 16 - 1) / 16; // 100 75 1
    grid_size_randinit.y=(height + 16 - 1) / 16;
    grid_size_randinit.z = 1;
    dim3 block_size_randinit; // 16 16 1
    block_size_randinit.x = 16;
    block_size_randinit.y = 16;
    block_size_randinit.z = 1;

    GetDepthandNormal<<<grid_size_randinit, block_size_randinit>>>(cuData.cameras_cuda+idx, normdepths_buffer, cuData.params);

    CUDA_SAFE_CALL(cudaDeviceSynchronize());
        auto options =
    torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
    return torch::from_blob(normdepths_buffer, {h,w,4},options); //{h,w,4}
}