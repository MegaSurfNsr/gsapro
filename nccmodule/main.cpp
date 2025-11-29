#include "main.h"
#include "ACMH.h"
#include "nccmodule.h"




void coordDim2(int *arr, int bound_1, int bound_2)
{
    int cn = 0;
    for (int i = 0; i < bound_1; i++){
        for (int j = 0; j < bound_2; j++){
            *(arr+2*cn) = i;
            *(arr+2*cn+1) = j;
            cn = cn + 1;
        }
    }
}

void coordshuffle(int *arr, int coord_num){
    int randxNum1;
    int randxNum2;
    int tem1;
    int tem2;
    for (int i =0; i<coord_num;i++){
        randxNum1 = rand() % coord_num;
        randxNum2 = rand() % coord_num;
        tem1 = *(arr+2*randxNum1);
        tem2 = *(arr+2*randxNum1 +1);
        *(arr+2*randxNum1) = *(arr+2*randxNum2);
        *(arr+2*randxNum1+1) = *(arr+2*randxNum2+1);
        *(arr+2*randxNum2) = tem1;
        *(arr+2*randxNum2+1) = tem2;
    }
}


int main(int argc, char** argv)
{
    std::cout << "Hello YS!" << std::endl;
    if (argc < 2) {
        std::cout << "USAGE: ACMH dense_folder" << std::endl;
        return -1;
    }
    struct cudaDeviceProp device_Property;
    int device_Count;
    cudaGetDeviceCount(&device_Count);
    std::cout << "CUDA device count!" << device_Count << std::endl;

    for (int i = 0; i < device_Count; i++)
    {
        cudaGetDeviceProperties(&device_Property,i);
        std::cout << "device info: " << i << std::endl;
        std::cout << "Maximum number of threads per multiprocessor:" << device_Property.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Maximum number of threads per block:" << device_Property.maxThreadsPerBlock << std::endl;
        std::cout << "Maximum number of block per multiprocessor:" << device_Property.maxBlocksPerMultiProcessor << std::endl;
    }
    
    // prepare the data here
    std::string dense_folder = argv[1];//"/home/yswang/Downloads/test/yswang_dtu24";//argv[1]; //"/home/yswang/Downloads/test/dtu24/acmm_data"; //
    // std::vector<Problem> problems;
    // GenerateSampleList(dense_folder, problems);
    // std::string output_folder = dense_folder + std::string("/ACMH");
    // mkdir(output_folder.c_str(), 0777);

    // size_t num_images = problems.size();
    // std::cout << "dense_folder: " << dense_folder << "....." << std::endl;
    // std::cout << "There are " << num_images << " problems needed to be processed!" << std::endl;
    std::string pairtxt = dense_folder + std::string("/pair.txt");

  


    if(true){

    std::cout << "nccmodule test:" << std::endl;
    NCCmodule nccmodule;
    nccmodule.set_hw(1200,1600);
    nccmodule.load_pair(pairtxt);
    float *py_ims = new float[nccmodule.h * nccmodule.w * nccmodule.num_images];
    std::cout<<"py_ims size:"<< nccmodule.h * nccmodule.w * nccmodule.num_images <<std::endl;

    float4 *py_normdepths = new float4[nccmodule.h * nccmodule.w * nccmodule.num_images];
    float *py_costs = new float[nccmodule.h * nccmodule.w * nccmodule.num_images];
    float *py_cams = new float[4*4*2 * nccmodule.num_images];
    float4 *py_planeHypos = new float4[nccmodule.h * nccmodule.w * nccmodule.num_images];

    nccmodule.fakePythonInput(dense_folder);
    nccmodule.fakeCamNy();
    // nccmodule.genCamFromNp();
    // nccmodule.fakeallcoord();
    // nccmodule.fakeProcessNcc(0);
    memcpy(py_cams, nccmodule.cameras_np,sizeof(float)*4*4*2 * nccmodule.num_images);
    memcpy(py_ims, nccmodule.ncc_grayImgs_host,sizeof(float)*nccmodule.h * nccmodule.w * nccmodule.num_images);
    std::cout<<"transfer size:"<< nccmodule.h * nccmodule.w * nccmodule.num_images <<std::endl;



    int *coord = new int[1200*1600*2];
    coordDim2(coord,1200,1600);    
    coordshuffle(coord,1200*1600);



    NCCmodule nccmodule2;
    nccmodule2.set_hw(1200,1600);
    nccmodule2.load_pair(pairtxt);
    nccmodule2.init(py_cams, py_ims, py_normdepths, py_costs, py_planeHypos);
    nccmodule2.genCamFromNp();
    nccmodule2.set_init_flag(true);
    nccmodule2.set_normdepths_flag(true);
    nccmodule2.dataToCuda();

    // sfm test

    // nccmodule2.cuData.writeDepth(15);

    // std::stringstream sfmfile;
    // sfmfile << dense_folder << "/sfm/" << std::setw(8) << std::setfill('0') << 15 << ".txt";
    // // std::cout<<sfmfile.str()<<std::endl;
    // nccmodule2.ProcessCuSfm(sfmfile.str(),15);
    // sfmfile.str(std::string());
    // sfmfile.clear();
    // sfmfile << dense_folder << "/sfm/" << std::setw(8) << std::setfill('0') << 0 << ".txt";
    // std::cout<<sfmfile.str()<<std::endl;
    // nccmodule2.ProcessCuSfm(sfmfile.str(),0);

    int idx = 15;
    nccmodule2.set_back_propagation_flag(false);
    
    nccmodule2.ProcessCuNcc(idx,coord, 1200*1600);
    nccmodule2.write_cuData_depth(idx);
    nccmodule2.write_cuData_cost(idx);

    nccmodule2.set_init_flag(false);

    nccmodule2.ProcessCuNcc(idx,coord, 1200*1600);
    nccmodule2.write_cuData_depth(idx);
    nccmodule2.write_cuData_cost(idx);
    // nccmodule2.ProcessCuNcc(0,coord, 1200*1600);
    // nccmodule2.write_cuData_depth(0);
    // nccmodule2.write_cuData_cost(0);
    // nccmodule2.ProcessCuNcc(15,coord, 1200*1600);

    // nccmodule2.ProcessCuNcc(15,coord, 1200*1600);

    // for (size_t i = 0; i < 300; i++) // 1200*1600/300 = 6400
    // {
    //    nccmodule2.ProcessCuNcc(15,coord + 2*i*6400,6400);
    // }
    // nccmodule2.ProcessCuNcc(15,coord, 1200*1600);
    // nccmodule2.ProcessCuNcc(0,coord, 1200*1600);
    // nccmodule2.set_init_flag(false);


    // nccmodule2.ProcessCuNcc(15,coord, 1200*1600);
    // nccmodule2.ProcessCuNcc(15,coord, 1200*1600);
    // nccmodule2.ProcessCuNcc(15,coord, 1200*1600);

    // for (size_t i = 0; i < 300; i++) // 1200*1600/300 = 6400
    // {
    //    nccmodule2.ProcessCuNcc(15,coord + 2*i*6400,6400);
    // }
    // coordshuffle(coord,1200*1600);
    // for (size_t i = 0; i < 300; i++) // 1200*1600/300 = 6400
    // {
    //    nccmodule2.ProcessCuNcc(15,coord + 2*i*6400,6400);
    // }
    // coordshuffle(coord,1200*1600);
    // for (size_t i = 0; i < 300; i++) // 1200*1600/300 = 6400
    // {
    //    nccmodule2.ProcessCuNcc(15,coord + 2*i*6400,6400);
    // }

    // nccmodule2.set_init_flag(false);
    // nccmodule2.ProcessCuNcc(15,coord,1200*1600);
    // nccmodule2.ProcessCuNcc(15,coord,1200*1600);

    // //ncc test
    // nccmodule2.ProcessCuNcc(0,coord,1200*1600);
    // nccmodule2.set_init_flag(false);
    // nccmodule2.ProcessCuNcc(0,coord,1200*1600);
    // nccmodule2.cuData.writeCost(15);
    // nccmodule2.cuData.writeDepth(15);

    // nccmodule2.saver.writeCostTest();
    // nccmodule2.saver.writeDepthTest();
    // nccmodule2.saver.writeFloatChannelTest(1);
    // nccmodule2.saver.writeFloatChannelTest(2);

    delete[] py_ims;
    delete[] py_normdepths;
    delete[] py_costs;
    delete[] py_planeHypos;
    delete[] coord;
    }




    std::cout<<"ver 04" <<std::endl;
    std::cout<<"test ok"<<std::endl;
    return 0;
}

