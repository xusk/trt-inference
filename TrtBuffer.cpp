#include <iostream>
#include <fstream>
#include <string>

#include "TrtBuffer.h"
#include "spdlog/spdlog.h"

bool TrtBuffer::DeserializeEngine(const std::string &engineFile) {
    std::ifstream in(engineFile.c_str(), std::ifstream::binary);
    if (in.is_open()) {
        spdlog::info("deserialize engine from {}", engineFile);
        auto const start_pos = in.tellg();
        in.ignore(std::numeric_limits<std::streamsize>::max());
        size_t bufCount = in.gcount();
        in.seekg(start_pos);
        std::unique_ptr<char[]> engineBuf(new char[bufCount]);
        in.read(engineBuf.get(), bufCount);

        nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(mLogger);
        _engine = runtime->deserializeCudaEngine((void *) engineBuf.get(), bufCount, nullptr);
        assert(_engine != nullptr);
        runtime->destroy();
        return true;
    }
    return false;
}

void TrtBuffer::InitEngine(int profileIndex, nvinfer1::Dims maxDims) {
    spdlog::info("init engine...");

    context = _engine->createExecutionContext();
    context->setOptimizationProfile(0);
    assert(context != nullptr);

    // 设置最大bindshape，已获取最大输出大小
    context->setBindingDimensions(0, maxDims);

    spdlog::info("create cuda stream");
    CUDA_CHECK(cudaStreamCreate(&stream));

    spdlog::info("malloc device memory");
    int nbBindings = _engine->getNbBindings();
    std::cout << "nbBingdings: " << nbBindings << std::endl;
    bindingDevice.resize(nbBindings);
    bindingHost.resize(nbBindings);
    bindingSize.resize(nbBindings);
    bindingDims.resize(nbBindings);
    bindingDataType.resize(nbBindings);

    for (int i = 0; i < nbBindings; i++) {
        nvinfer1::Dims dims = context->getBindingDimensions(i);
        nvinfer1::DataType dtype = _engine->getBindingDataType(i);
        const char *name = _engine->getBindingName(i);
        int64_t totalSize = volume(dims) * getElementSize(dtype);

        bindingDims[i] = dims;
        bindingDataType[i] = dtype;
        bindingSize[i] = totalSize;

        if (_engine->bindingIsInput(i)) {
            spdlog::info("input: ");
            inputBindIndex.push_back(i);
        } else {
            spdlog::info("output: ");
            outputBindIndex.push_back(i);
        }
        spdlog::info("bindingDevice bindIndex: {}, name: {}, size in byte: {}", i, name, totalSize);
        spdlog::info("bindingDevice dims with {} dimemsion", dims.nbDims);

        bindingDevice[i] = safeCudaMalloc(totalSize);
        bindingHost[i] = safeCudaHostAlloc(totalSize);
    }
    spdlog::info("engine init finish");
}

void TrtBuffer::InitEngine() {
    spdlog::info("init engine...");

    context = _engine->createExecutionContext();
    assert(context != nullptr);

    spdlog::info("create cuda stream");
    CUDA_CHECK(cudaStreamCreate(&stream));

    spdlog::info("malloc device memory");
    int nbBindings = _engine->getNbBindings();
    std::cout << "nbBingdings: " << nbBindings << std::endl;
    bindingDevice.resize(nbBindings);
    bindingHost.resize(nbBindings);
    bindingSize.resize(nbBindings);
    bindingDims.resize(nbBindings);
    bindingDataType.resize(nbBindings);

    for (int i = 0; i < nbBindings; i++) {
        nvinfer1::Dims dims = _engine->getBindingDimensions(i);
        nvinfer1::DataType dtype = _engine->getBindingDataType(i);
        const char *name = _engine->getBindingName(i);
        int64_t totalSize = volume(dims) * getElementSize(dtype);

        bindingDims[i] = dims;
        bindingDataType[i] = dtype;
        bindingSize[i] = totalSize;

        if (_engine->bindingIsInput(i)) {
            spdlog::info("input: ");
            inputBindIndex.push_back(i);
        } else {
            spdlog::info("output: ");
            outputBindIndex.push_back(i);
        }
        spdlog::info("bindingDevice bindIndex: {}, name: {}, size in byte: {}", i, name, totalSize);
        spdlog::info("bindingDevice dims with {} dimemsion", dims.nbDims);

        bindingDevice[i] = safeCudaMalloc(totalSize);
        bindingHost[i] = safeCudaHostAlloc(totalSize);
    }
    spdlog::info("engine init finish");
}

TrtBuffer::TrtBuffer(const std::string &engineFile) {
    DeserializeEngine(engineFile);
    InitEngine();
}

TrtBuffer::TrtBuffer(const std::string &engineFile, int profileIndex, nvinfer1::Dims maxDims) {
    DeserializeEngine(engineFile);
    InitEngine(profileIndex, maxDims);
}

TrtBuffer::~TrtBuffer() {
    if (context != nullptr) {
        context->destroy();
        context = nullptr;
    }
    if (_engine != nullptr) {
        _engine->destroy();
        _engine = nullptr;
    }

    for (auto bind: bindingDevice) {
        safeCudaFree(&bind);
    }
}


void TrtBuffer::DataTransferAsync(int size, int bindIndex, bool isHostToDevice) {
    auto host = bindingHost[bindIndex];
    if (isHostToDevice) {
        assert(size * sizeof(float) <= bindingSize[bindIndex]);
        CUDA_CHECK(cudaMemcpyAsync(
                bindingDevice[bindIndex], host, size * sizeof(float), cudaMemcpyHostToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpyAsync(
                host, bindingDevice[bindIndex], size * sizeof(float), cudaMemcpyDeviceToHost, stream));
    }
}


void TrtBuffer::ForwardAsync(nvinfer1::Dims &dim) {
    context->setBindingDimensions(0, dim);
    context->enqueueV2(bindingDevice.data(), stream, nullptr);
}

void TrtBuffer::ForwardAsync() {
    context->enqueueV2(bindingDevice.data(), stream, nullptr);
}


void TrtBuffer::GetOutput() {
    for (int idx = 0; idx < outputBindIndex.size(); ++idx) {
        int bindIndex = outputBindIndex[idx];
        auto hostOutput = bindingHost[bindIndex];

        // 数据 GPU -> host
        DataTransferAsync(GetRuntimeBindingSize(bindIndex), bindIndex, false);
    }
}