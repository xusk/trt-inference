#ifndef __TRT_BUFFER_H__
#define __TRT_BUFFER_H__

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

#include <cuda_runtime.h>
#include "NvInfer.h"
#include "concurrentqueue/blockingconcurrentqueue.h"

#include "utils.h"


class TrtBuffer {
public:
    TrtBuffer(){};

    ~TrtBuffer();

    TrtBuffer(const std::string &engineFile);

    TrtBuffer(const std::string &engineFile, int profileIndex, nvinfer1::Dims maxDims);

    bool DeserializeEngine(const std::string &engineFile);

    // 模型初始化
    void InitEngine();

    // 模型初始化 动态
    void InitEngine(int profileIndex, nvinfer1::Dims maxDims);

    void DataTransferAsync(int size, int bindIndex, bool isHostToDevice);

    void ForwardAsync();  // 固定维度

    void ForwardAsync(nvinfer1::Dims &dim);

    void GetOutput();


    size_t GetRuntimeBindingSize(int bindIndex) const {
        return volume(GetRuntimeBindingDims(bindIndex));
    }
    nvinfer1::Dims GetRuntimeBindingDims(int bindIndex) const {
        return context->getBindingDimensions(bindIndex);
    }
    nvinfer1::DataType GetBindingDataType(int bindIndex) const {
        return bindingDataType[bindIndex];
    }
    void StreamSynchronize() const {
        cudaStreamSynchronize(stream);
    }

    // var
    cudaStream_t stream;
    nvinfer1::IExecutionContext *context = nullptr;

    std::vector<void *> bindingDevice;   // 设备内存
    std::vector<float *> bindingHost;    // 锁页内存

    std::vector<int> inputBindIndex;     // 输入index
    std::vector<int> outputBindIndex;    // 输出index
    std::vector<int> bindingSize;        // 输入输入最大大小

    std::vector<nvinfer1::Dims> bindingDims;
    std::vector<nvinfer1::DataType> bindingDataType;

private:
    TrtLogger mLogger;
    nvinfer1::ICudaEngine  *_engine = nullptr;

};



#endif  // !__TRT_BUFFER_H__