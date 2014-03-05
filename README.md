OCLAlgo framework
=================

## Brief Description
OpenCL framework, which is based on C++ OpenCL API Wrapper and C++11. It provides simple access to
OpenCL platforms and devices for sync/async calculations.

## Details
**OCLAlgo provides simple OpenCL platform and device initialization by:**
* part platform and device names (isn't case sensetive)
* number of platform and device in your system

```cpp
// 1st case
oclalgo::Queue queue("NVIDIA", "GeForce");
// 2nd case
oclalgo::Queue queue(0, 0);
```

**To enqueue OpenCL kernel you should make four simple steps:**

1.  Create KernelArg objects for OpenCL memory buffers with corresponding markers (IN, OUT, IN_OUT).
1.  Create Task object using OpenCL program name (*.cl file), kernel name, compilation options
and arguments in the same order as in OpenCL kernel (if you need to pass primitive types
(such as int, float, double, char) to kernel, you simply pass them to Task constructor).
1.  Create Grid object to define dimensions of OpenCL task.
1.  Enqueue created task with corresponding grid.
```cpp
/* 1 */
oclalgo::KernelArg arg(host_array, oclalgo::ArgType::IN_OUT);
int add_value = 101;
/* 2 */
oclalgo::Task task("vector.cl", "modify", "", arg, add_value);
/* 3 */
oclalgo::Grid grid(cl::NDRange(size));
/* 4 */
auto ocl_res = queue.EnqueueTask(task, grid);
```

<b>To get the output results you should call *\<future_object\>.get()*</b>. In this case you wait
while OpenCL finishes task, then *get()* method returns *std::vector* with output OpenCL buffers
(this buffers was marked as ArgType::OUT or ArgType::IN_OUT when was passed to KernelArg object).
You also can call *\<future_object\>.wait()* to wait while OpenCL finishes task.
```cpp
std::vector<cl::Buffer> v_res = ocl_res.get();
```

**If you want to copy OpenCL buffer to host array or vise versa, you should call Queue::memcpy**
(it's available to use sync or async approach to copy memory objects between Host and OpneCL devices).
In async case oclalgo::future object is returned).
```cpp
queue.memcpy(host_array, v_res[0]);
```

## License
The source for OCLAlgo is licensed under the BSD licence
Copyright (c) 2014, Samsung Electronics Co.,Ltd.

## Example
As an example of framework using oclalgo::Matrix and oclalgo::DMatrix calsses was added.
This classes implement basic matrix operations ( + / - / * ) using host (oclalgo::Matrix) and
device (oclalgo::DMatrix) resources.

You can find function test below, which checks correct work of
oclalgo::Queue class in case of vectors addition.

```cpp
#include <algorithm>
#include <iostream>
#include <string>

#include <gtest/gtest.h>
#include <src/gtest_main.cc>
#include <oclalgo/queue.h>

TEST(Queue, VectorAdd) {
  try {
    // create OpenCL queue for sync/async task running using
    // part platform and device names
    oclalgo::Queue queue("NVIDIA", "GeForce");

    // create and initialize input shared arrays
    int size = 1024;
    oclalgo::shared_array<int> a(size), b(size);
    for (int i = 0; i < size; ++i) {
      a[i] = i;
      b[i] = size - i;
    }

    // initialize OpenCl kernel arguments
    using oclalgo::ArgType;
    using oclalgo::BufferArg;
    BufferArg a_arg = queue.CreateKernelArg(a, ArgType::IN);
    BufferArg b_arg = queue.CreateKernelArg(b, ArgType::IN);
    BufferArg c_arg = queue.CreateKernelArg<int>(size, ArgType::OUT);

    // create task using OpenCL program and kernel names, compilation options
    // and arguments in the same order as in OpenCL kernel
    oclalgo::Task task = queue.CreateTask("vector.cl", "vector_add", "",
                                          a_arg, b_arg, c_arg);

    // create grid to define dimensions of OpenCL task
    // in global and local (group size) space
    oclalgo::Grid grid = oclalgo::Grid(cl::NDRange(size));

    // enqueue OpenCL task (EnqueueTask() returns oclalgo::future object
    // for async task running)
    auto ocl_res = queue.EnqueueTask(task, grid);

    // copy device memory with result to host
    // (ocl_res.get() waits while OpenCL finished task
    // and returns std::vector with output OpenCL buffers,
    // which was marked as ArgType::OUT or ArgType::IN_OUT when was created)
    queue.memcpy(a, ocl_res.get()[0]);

    // check result
    auto it = std::find_if(a.get_raw(), a.get_raw() + a.size(),
                           [size](int x) { return x != size; });
    ASSERT_EQ(a.get_raw() + a.size(), it);
  } catch (const cl::Error& e) {
    std::cerr << e.what() << " (err_code = "
              << oclalgo::Queue::StatusStr(e.err()) << ")" << std::endl;
    throw e;
  }
}
```
