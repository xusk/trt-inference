#ifndef __TRT_H__
#define __TRT_H__

#include <iostream>
#include <string>
#include <vector>
#include <functional>

#include "utils.h"
#include "TrtBuffer.h"
#include "spdlog/spdlog.h"


class ZyTrt {
public:
    ZyTrt() {};

    ~ZyTrt();

    ZyTrt(int num_worker, const std::string &engineFile);

    ZyTrt(int num_worker, const std::string &engineFile, int profileIndex, nvinfer1::Dims maxDims);

    ZyTrt(int num_worker, const std::string &engineFile, int profileIndex, std::vector<nvinfer1::Dims> maxDimList);

    std::function<void()> DoDynamicInferenceAsync(
        const void *input, nvinfer1::Dims &inputDim,
        std::vector<float *> &outputList, std::vector<nvinfer1::Dims> &outputDimList);

    std::function<void()> DoInferenceAsync(
        const void *input, int batch_size,
        std::vector<float *> &outputList, std::vector<nvinfer1::Dims> &outputDimList);

protected:
    // 构建队列
    TrtBuffer *getTrtBuffer() {
        TrtBuffer *buffer = nullptr;
        if (isWorker) {
            _bufferPool.wait_dequeue(buffer);
        }
        else {
            buffer = trtBuffer;
        }
        return buffer;
    }

    void releaseTrtBuffer(TrtBuffer *buffer) {
        if (isWorker) {
            _bufferPool.enqueue(buffer);
        }
    }

private:
    moodycamel::BlockingConcurrentQueue<TrtBuffer *> _bufferPool;
    bool isWorker = false;
    int numWorker = 0;

    // 构建推理buffer
    TrtBuffer *trtBuffer;
};


#endif // !__TRT_H__