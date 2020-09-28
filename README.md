# trt-inference



## 说明

支持动态输入，静态输入，支持多线程调用
- 封装 c++ api，方便 python调用 
- 优化数据传输，数据复制到锁页内存

base on tinytrt


## 环境
- cuda 10.2 cudnn 8.0.3.33 tensorrt 7.1.3
- tensorrt 7.0.11 有显存泄露的风险


## 编译
``` shell
mkdir build
cmake .. 
make
```



## 使用
```python
import pytrt

# create trt engine
trt = pytrt.ZyTrt(num_worker, engineFile);

# create dynamic trt engine 
trt = pytrt.ZyTrt(num_worker, engineFile, profileIndex, inputDims);

# DoInference
trt.DoInference(input) 

# DynamicInference
trt.DoDynamicInferenceAsync(input)
```
