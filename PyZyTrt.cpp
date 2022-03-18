#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/stl.h"
#include "ZyTrt.h"
#include "spdlog/spdlog.h"
#include "NvInfer.h"

#include <typeinfo>
#include <cstring>



namespace py = pybind11;

PYBIND11_MODULE(pytrt, m) {
    m.doc() = "python interface of trt-tensorrt";
    py::class_<ZyTrt>(m, "ZyTrt")
    .def(py::init([]() {
        int gpuId = 0; 
        return std::unique_ptr<ZyTrt>(new ZyTrt(gpuId));
    }))
    .def(py::init([](int num_worker, const std::string &engineFile) {
        return std::unique_ptr<ZyTrt>(new ZyTrt(num_worker, engineFile));
    }))
    .def(py::init([](int num_worker, const std::string &engineFile, int profileIndex, std::vector<int> inputDims) {
        nvinfer1::Dims dim{inputDims.size()};
        for (unsigned int i = 0; i < inputDims.size(); ++i) {
            dim.d[i] = inputDims[i];
        }
        return std::unique_ptr<ZyTrt>(new ZyTrt(num_worker, engineFile, profileIndex, dim));
    }))
    .def(py::init([](int num_worker, const std::string &engineFile, int profileIndex, std::vector<std::vector<int>> inputDimList) {
        std::vector<nvinfer1::Dims> dimList;
        for (unsigned int i = 0; i < inputDimList.size(); ++i){
            nvinfer1::Dims dim{inputDimList[i].size()};
            for (unsigned int j = 0; j < inputDimList[i].size(); ++j)
            {
                dim.d[j] = inputDimList[i][j];
            }
            dimList.push_back(dim);
        }
        return std::unique_ptr<ZyTrt>(new ZyTrt(num_worker, engineFile, profileIndex, dimList));
    }))
    .def("DoDynamicInferenceAsync", [](ZyTrt& self, py::array_t<float, py::array::c_style | py::array::forcecast> array, std::vector<int> inputDims) {
        // 释放GIL
        py::gil_scoped_release release;

        nvinfer1::Dims dim{inputDims.size()};
        for (unsigned int i = 0; i < inputDims.size(); ++i) {
            dim.d[i] = inputDims[i];
        }
        auto batch_size = array.shape(0);
        std::vector<float *> outputList;
        std::vector<nvinfer1::Dims> outputDimList;
        // std::cout << "array.data: " << array.data() << "id trt:"  << &self << std::endl;

        std::function<void()> releaseFunc = self.DoDynamicInferenceAsync(array.data(), dim, outputList, outputDimList);

        std::vector<py::array> fusionOutput;
        std::vector<std::vector<ssize_t> > shapeList;
        std::vector<std::vector<ssize_t> > stridesList;
        for (size_t i = 0; i < outputList.size(); ++i) {
            auto dim = outputDimList[i];
            ssize_t nbDims= dim.nbDims;

            std::vector<ssize_t> shape;
            for(int i=0;i<nbDims;i++){
                if ( i == 0  && batch_size != dim.d[i]) {
                    shape.push_back(batch_size);
                }else{
                    shape.push_back(dim.d[i]);
                };
            }
            shapeList.push_back(shape);

            std::vector<ssize_t> strides;
            for(int i=0;i<nbDims;i++){
                ssize_t stride = sizeof(float);
                for(int j=i+1;j<nbDims;j++) {
                    stride = stride * shape[j];
                }
                strides.push_back(stride);
            }

            stridesList.push_back(strides);
        }

        // 获取GIL
        py::gil_scoped_acquire acquire;
        for (size_t i = 0; i < outputList.size(); ++i) {
            auto output = outputList[i];
            auto dim = outputDimList[i];
            ssize_t nbDims= dim.nbDims;
            auto shape = shapeList[i];
            auto strides = stridesList[i];

            fusionOutput.push_back(py::array(py::buffer_info(
                output,
                sizeof(float),
                py::format_descriptor<float>::format(),
                nbDims,
                shape,
                strides
            )));
        }

        // 释放buffer
        releaseFunc();
        return fusionOutput;
    // }, py::call_guard<py::gil_scoped_release>())
    })
    .def("DoInference", [](ZyTrt& self, py::array_t<float, py::array::c_style | py::array::forcecast> array) {
        py::gil_scoped_release release;

        std::vector<float *> outputList;
        std::vector<nvinfer1::Dims> outputDimList;

        auto batch_size = array.shape(0);

        // 推理
        std::function<void()> releaseFunc = self.DoInferenceAsync(array.data(), batch_size, outputList, outputDimList);

        std::vector<py::array> fusionOutput;
        std::vector<std::vector<ssize_t> > shapeList;
        std::vector<std::vector<ssize_t> > stridesList;
        for (size_t i = 0; i < outputList.size(); ++i) {
            auto dim = outputDimList[i];
            ssize_t nbDims= dim.nbDims;

            std::vector<ssize_t> shape;
            for(int i=0; i<nbDims; i++){
                if ( i == 0  && batch_size != dim.d[i]) {
                    shape.push_back(batch_size);
                }else{
                    shape.push_back(dim.d[i]);
                };
            }
            shapeList.push_back(shape);

            std::vector<ssize_t> strides;
            for(int i=0; i<nbDims; i++){
                ssize_t stride = sizeof(float);
                for(int j=i+1;j<nbDims;j++) {
                    stride = stride * shape[j];
                }
                strides.push_back(stride);
            }

            stridesList.push_back(strides);
        }

        // 获取GIL
        py::gil_scoped_acquire acquire;
        for (size_t i = 0; i < outputList.size(); ++i) {
            auto output = outputList[i];
            auto dim = outputDimList[i];
            ssize_t nbDims= dim.nbDims;
            auto shape = shapeList[i];
            auto strides = stridesList[i];

            fusionOutput.push_back(py::array(py::buffer_info(
                output,
                sizeof(float),
                py::format_descriptor<float>::format(),
                nbDims,
                shape,
                strides)));
            }

            releaseFunc();
            return fusionOutput;
        })
        ;
}