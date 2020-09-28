#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"

#include "NvInfer.h"
#include <typeinfo>

#include "ZyTrt.h"
#include "spdlog/spdlog.h"

namespace py = pybind11;

PYBIND11_MODULE(pytrt, m) {
    m.doc() = "python interface of trt-tensorrt";
    py::class_<ZyTrt>(m, "ZyTrt")
    .def(py::init([]() {
        return std::unique_ptr<ZyTrt>(new ZyTrt());
    }))
    .def(py::init([](const std::string &engineFile) {
        return std::unique_ptr<ZyTrt>(new ZyTrt(engineFile));
    }))

    .def(py::init([](const std::string &engineFile, int profileIndex, std::vector<int> inputDims) {
        nvinfer1::Dims dim{inputDims.size()};
        for (int i = 0; i < inputDims.size(); ++i) {
            dim.d[i] = inputDims[i];
        }
        return std::unique_ptr<ZyTrt>(new ZyTrt(engineFile, profileIndex, dim));
    }))
    .def("DoDynamicInferenceAsync", [](ZyTrt& self, py::array_t<float, py::array::c_style | py::array::forcecast> array, std::vector<int> inputDims) {
        nvinfer1::Dims dim{inputDims.size()};
        for (int i = 0; i < inputDims.size(); ++i) {
            dim.d[i] = inputDims[i];
        }
        auto batch_size = array.shape(0);
        std::vector<float *> outputList;
        std::vector<nvinfer1::Dims> outputDimList;

        self.DoDynamicInferenceAsync(array.data(), dim, outputList, outputDimList);

        std::vector<py::array> fusionOutput;
        for (size_t i = 0; i < outputList.size(); ++i) {
            auto dim = outputDimList[i];
            auto output = outputList[i];
            ssize_t nbDims= dim.nbDims;

            std::vector<ssize_t> shape;
            for(int i=0;i<nbDims;i++){
                if ( i == 0  && batch_size != dim.d[i]) {
                    shape.push_back(batch_size);
                }else{
                    shape.push_back(dim.d[i]);
                };
            }
            std::vector<ssize_t> strides;
            for(int i=0;i<nbDims;i++){
                ssize_t stride = sizeof(float);
                for(int j=i+1;j<nbDims;j++) {
                    stride = stride * shape[j];
                }
                strides.push_back(stride);
            }
            fusionOutput.push_back(py::array(py::buffer_info(
                output,
                sizeof(float),
                py::format_descriptor<float>::format(),
                nbDims,
                shape,
                strides
            )));
        }
        return fusionOutput;
    })
    .def("DoInference", [](ZyTrt& self, py::array_t<float, py::array::c_style | py::array::forcecast> array) {
        std::vector<float *> outputList;
        std::vector<nvinfer1::Dims> outputDimList;

        auto batch_size = array.shape(0);

        // 推理
        self.DoInferenceAsync(array.data(), batch_size, outputList, outputDimList);

        std::vector<py::array> fusionOutput;
        for (size_t i = 0; i < outputList.size(); ++i) {
            auto dim = outputDimList[i];
            auto output = outputList[i];
            ssize_t nbDims= dim.nbDims;

            std::vector<ssize_t> shape;
            for (int i = 0 ; i < nbDims; i++) {
                if ( i == 0  && batch_size != dim.d[i]) {
                    shape.push_back(batch_size);
                }else{
                    shape.push_back(dim.d[i]);
                };
            }
            std::vector<ssize_t> strides;
            for(int i=0;i<nbDims;i++){
                ssize_t stride = sizeof(float);
                for(int j=i+1;j<nbDims;j++) {
                    stride = stride * shape[j];
                }
                strides.push_back(stride);
            }
            fusionOutput.push_back(py::array(py::buffer_info(
                    output,
                    sizeof(float),
                    py::format_descriptor<float>::format(),
                    nbDims,
                    shape,
                    strides
            )));
        }
        return fusionOutput;
    })
    ;
}