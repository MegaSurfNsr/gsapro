#include "nccmodule.h"

namespace py = pybind11;


void printTest(std::string input){
    std::cout << input << std::endl;
}

PYBIND11_MODULE(ncc, m) {
    m.doc() = R"pbdoc(
        Pybind11 ncc plugin
        -----------------------
        usage:NCCmodule()
    )pbdoc";


    m.def("printTest", &printTest, R"pbdoc(
    print input
    Some other explanation about the function.
    )pbdoc");
    
    py::class_<NCCmodule>(m, "NCCmodule")
    .def(py::init<>())
    .def("set_hw",&NCCmodule::set_hw, R"pbdoc(set images height and width.)pbdoc")
    .def("get_hw",&NCCmodule::get_hw, R"pbdoc(get images height and width.)pbdoc")
    .def("load_pair",&NCCmodule::load_pair, R"pbdoc(the path to the pair.txt file, init the problems.)pbdoc")
    .def("fakePythonInput",&NCCmodule::fakePythonInput, R"pbdoc(give the dense folder and fake a input for test.)pbdoc")
    .def("fakeCamNy",&NCCmodule::fakeCamNy)
    .def("fakeallcoord",&NCCmodule::fakeallcoord)
    .def("genCamFromNp",&NCCmodule::genCamFromNp)
    .def("fakeProcessNcc",&NCCmodule::fakeProcessNcc)
    .def("set_init_flag",&NCCmodule::set_init_flag)
    .def("set_normdepths_flag",&NCCmodule::set_normdepths_flag)
    .def("__version__",&NCCmodule::__version__)
    .def("bind_ncc_cu",&NCCmodule::bind_ncc_cu)
    .def("bind_hypos_cu",&NCCmodule::bind_hypos_cu)
    .def("bind_back_geo",&NCCmodule::bind_back_geo)
    .def("bind_back_ncc",&NCCmodule::bind_back_ncc)
    .def("bind_checkerboard_ncc_cu",&NCCmodule::bind_checkerboard_ncc_cu)
    .def("bind_checkerboard_hypos_cu",&NCCmodule::bind_checkerboard_hypos_cu)
    .def("hypos_w2c",&NCCmodule::hypos_w2c)
    .def("dataToCuda",&NCCmodule::dataToCuda)
    .def("saverncc_test",&NCCmodule::saverncc_test)
    .def("set_back_propagation_flag",&NCCmodule::set_back_propagation_flag)
    .def("ProcessCuSfm",&NCCmodule::ProcessCuSfm)
    .def("ProcessDtuAll",&NCCmodule::ProcessDtuAll)
    .def("set_cameras_np",[](NCCmodule &selfclass,py::buffer arr){
        py::buffer_info cam_arr_info=arr.request();
        selfclass.set_cameras_np(static_cast<float*>(cam_arr_info.ptr));
    })
    .def("set_ncc_grayImgs_host",[](NCCmodule &selfclass,py::buffer arr){
        py::buffer_info gray_arr_info=arr.request();
        selfclass.set_ncc_grayImgs_host(static_cast<float*>(gray_arr_info.ptr));
    })
    .def("set_ncc_planeHypos_host",[](NCCmodule &selfclass,py::buffer arr){
        py::buffer_info hypo_arr_info=arr.request();
        selfclass.set_ncc_planeHypos_host(static_cast<float4*>(hypo_arr_info.ptr));
    })
    .def("set_ncc_normdepths_host",[](NCCmodule &selfclass,py::buffer arr){
        py::buffer_info d_arr_info=arr.request();
        selfclass.set_ncc_normdepths_host(static_cast<float4*>(d_arr_info.ptr));
    })
    .def("set_ncc_costs_host",[](NCCmodule &selfclass,py::buffer arr){
        py::buffer_info c_arr_info=arr.request();
        selfclass.set_ncc_costs_host(static_cast<float*>(c_arr_info.ptr));
    })
    .def("ProcessNcc",[](NCCmodule &selfclass,int img_idx, py::buffer coord, int n_coord){
        py::buffer_info arr_info=coord.request();
        selfclass.ProcessNcc(img_idx,static_cast<int*>(arr_info.ptr),n_coord);
    }, R"pbdoc(solve the problem of image idx with the given coord.)pbdoc")
    .def("ProcessCuNcc",[](NCCmodule &selfclass,int img_idx, py::buffer coord, int n_coord){
        py::buffer_info arr_info=coord.request();
        selfclass.ProcessCuNcc(img_idx,static_cast<int*>(arr_info.ptr),n_coord);
    }, R"pbdoc(solve the problem of image idx with the given coord.)pbdoc")
    .def("init",[](NCCmodule &selfclass,py::buffer py_cameras, py::buffer py_grayImgs, py::buffer py_normdepths, py::buffer py_costs, py::buffer py_planeHypos){
        py::buffer_info py_cameras_info=py_cameras.request();
        py::buffer_info py_grayImgs_info=py_grayImgs.request();
        py::buffer_info py_normdepths_info=py_normdepths.request();
        py::buffer_info py_costs_info=py_costs.request();
        py::buffer_info py_planeHypos_info=py_planeHypos.request();
        selfclass.init(static_cast<float*>(py_cameras_info.ptr),static_cast<float*>(py_grayImgs_info.ptr),static_cast<float4*>(py_normdepths_info.ptr),
        static_cast<float*>(py_costs_info.ptr),static_cast<float4*>(py_planeHypos_info.ptr));
    }, R"pbdoc(binding the python data to the plugin.)pbdoc");
        



    m.attr("__version__") = "2023.7.13";
}
