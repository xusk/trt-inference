#include <iostream>
#include <cstring>
#include "ZyTrt.h"


ZyTrt::ZyTrt(int num_worker, const std::string &engineFile) {
    isWorker = true;
    numWorker = num_worker;
    for (int i = 0; i < num_worker; ++i) {
        std::cout << i << " start" << std::endl;
        _bufferPool.enqueue(new TrtBuffer(engineFile));
        std::cout << i << " finish" << std::endl;
    }
}


ZyTrt::ZyTrt(int num_worker, const std::string &engineFile, int profileIndex, nvinfer1::Dims maxDims) {
    isWorker = true;
    numWorker = num_worker;
    for (int i = 0; i < num_worker; ++i) {
        std::cout << i << " start" << std::endl;
        _bufferPool.enqueue(new TrtBuffer(engineFile, profileIndex, maxDims));
        std::cout << i << " finish" << std::endl;
    }
}


ZyTrt::~ZyTrt(){
    for (int i = 0;i < numWorker;i++){
        TrtBuffer *buffer = nullptr;
        _bufferPool.wait_dequeue(buffer);
        delete(buffer);
    }
}

std::function<void()> ZyTrt::DoDynamicInferenceAsync(
        const void *input, nvinfer1::Dims &inputDim,
        std::vector<float *> &outputList, std::vector <nvinfer1::Dims> &outputDimList
) {
    size_t inputIndex = 0;
    size_t size = volume(inputDim);

    // 获取buffer
    TrtBuffer *buffer = getTrtBuffer();

    // 数据复制到锁页内存
    std::memcpy(buffer->bindingHost[inputIndex], input, size * sizeof(float));

    // 数据 host -> gpu
    buffer->DataTransferAsync(size, inputIndex, true);

    // 推理
    buffer->ForwardAsync(inputDim);

    // 数据 gpu -> host
    buffer->GetOutput();

    // 等待数据执行完成
    buffer->StreamSynchronize();

    for (auto bindIndex : buffer->outputBindIndex) {
        // 复制指针
        outputList.push_back(buffer->bindingHost[bindIndex]);
        outputDimList.push_back(buffer->GetRuntimeBindingDims(bindIndex));
    }

    return std::bind(&ZyTrt::releaseTrtBuffer, this, buffer);
}

std::function<void()> ZyTrt::DoInferenceAsync(
        const void *input, int batch_size,
        std::vector<float *> &outputList, std::vector <nvinfer1::Dims> &outputDimList
) {
    size_t inputIndex = 0;

    // 获取buffer
    TrtBuffer *buffer = getTrtBuffer();

    auto max_batch_size = buffer->bindingDims[inputIndex].d[0];

    // 当前输入的大小
    size_t size = buffer->bindingSize[inputIndex] / sizeof(float) / max_batch_size * batch_size;

    // 数据复制到锁页内存
    std::memcpy(buffer->bindingHost[inputIndex], input, size * sizeof(float));

    // 数据 host -> gpu
    buffer->DataTransferAsync(size, inputIndex, true);

    // 推理
    buffer->ForwardAsync();

    // 数据 gpu -> host
    buffer->GetOutput();

    // 等待数据执行完成
    buffer->StreamSynchronize();

    for (auto bindIndex : buffer->outputBindIndex) {
        // 复制指针
        outputList.push_back(buffer->bindingHost[bindIndex]);
        outputDimList.push_back(buffer->GetRuntimeBindingDims(bindIndex));
    }

    return std::bind(&ZyTrt::releaseTrtBuffer, this, buffer);
}