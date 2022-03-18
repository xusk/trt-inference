#ifndef _ENTROY_CALIBRATOR_H
#define _ENTROY_CALIBRATOR_H

#include <cudnn.h>
#include <string>
#include <vector>
#include "NvInfer.h"
#include "utils.h"


class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator(int BatchSize, const std::vector <std::vector<float>> &data,
                          const std::string &CalibDataName = "", bool readCache = true);

    virtual ~Int8EntropyCalibrator();

    int getBatchSize() const TRT_NOEXCEPT override {
        std::cout << "getbatchSize: " << mBatchSize << std::endl;
        return mBatchSize;
    }

    bool getBatch(void *bindings[], const char *names[], int nbBindings) TRT_NOEXCEPT override;

    const void *readCalibrationCache(size_t &length) TRT_NOEXCEPT override;

    void writeCalibrationCache(const void *cache, size_t length) TRT_NOEXCEPT override;

private:
    std::string mCalibDataName;
    std::vector <std::vector<float>> mDatas;
    int mBatchSize;

    int mCurBatchIdx;
    float *mCurBatchData{nullptr};

    size_t mInputCount;
    bool mReadCache;
    void *mDeviceInput{nullptr};

    std::vector<char> mCalibrationCache;
};


#endif //_ENTROY_CALIBRATOR_H