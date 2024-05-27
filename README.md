# Nick's CUDA Ray Tracer

![Final Render](https://www.nsdigirolamo.com/posts/cuda-ray-tracing/images/book1_final.png)

This is a ray tracer written in C++ accelerated with the CUDA parallel computing
platform. When I started working on this project, I was following the guide of
the [Ray Tracing in One Weekend](https://raytracing.github.io/) textbook by
Peter Shirley, Trever D. Black, and Steve Hollasch. You can find the older serial 
version of this codebase at [this repository.](https://github.com/nsdigirolamo/ray-tracing-playground)

## Overview

To build this project, your system must have the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
installed. To run this project, your system must have a CUDA capable NVIDIA device.

This project's soure code includes the [doctest](https://github.com/doctest/doctest/tree/master)
C++ unit testing framework, which is released under [the MIT License.](https://github.com/doctest/doctest/blob/master/LICENSE.txt)
