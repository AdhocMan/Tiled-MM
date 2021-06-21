/*
 * Copyright (c) 2019 ETH Zurich, Simon Frasch
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
#pragma once

#include <utility>
#include <stdexcept>

#if defined(TILED_MM_CUDA)
#include <cublas_v2.h>

#elif defined(TILED_MM_ROCM)
#include <hipblas.h>

#else
#error Either TILED_MM_CUDA or TILED_MM_ROCM must be defined!
#endif

namespace gpu {
namespace blas_api {

#if defined(TILED_MM_CUDA)
using HandleType = cublasHandle_t;
using StatusType = cublasStatus_t;
using OperationType = cublasOperation_t;
using ComplexFloatType = cuComplex;
using ComplexDoubleType = cuDoubleComplex;
#endif

#if defined(TILED_MM_ROCM)
using HandleType = hipblasHandle_t;
using StatusType = hipblasStatus_t;
using OperationType = hipblasOperation_t;
using ComplexFloatType = hipblasComplex;
using ComplexDoubleType = hipblasDoubleComplex;
#endif

namespace operation {
#if defined(TILED_MM_CUDA)
constexpr auto None = CUBLAS_OP_N;
constexpr auto Transpose = CUBLAS_OP_T;
constexpr auto ConjugateTranspose = CUBLAS_OP_C;
#endif

#if defined(TILED_MM_ROCM)
constexpr auto None = HIPBLAS_OP_N;
constexpr auto Transpose = HIPBLAS_OP_T;
constexpr auto ConjugateTranspose = HIPBLAS_OP_C;
#endif
}  // namespace operation

namespace status {
#if defined(TILED_MM_CUDA)
constexpr auto Success = CUBLAS_STATUS_SUCCESS;
#endif

#if defined(TILED_MM_ROCM)
constexpr auto Success = HIPBLAS_STATUS_SUCCESS;
#endif

static const char* get_string(StatusType error)
{
#if defined(TILED_MM_CUDA)
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }
#endif

#if defined(TILED_MM_ROCM)
    switch (error)
    {
        case HIPBLAS_STATUS_SUCCESS:
            return "HIPBLAS_STATUS_SUCCESS";

        case HIPBLAS_STATUS_NOT_INITIALIZED:
            return "HIPBLAS_STATUS_NOT_INITIALIZED";

        case HIPBLAS_STATUS_ALLOC_FAILED:
            return "HIPBLAS_STATUS_ALLOC_FAILED";

        case HIPBLAS_STATUS_MAPPING_ERROR:
            return "HIPBLAS_STATUS_MAPPING_ERROR";

        case HIPBLAS_STATUS_EXECUTION_FAILED:
            return "HIPBLAS_STATUS_EXECUTION_FAILED";

        case HIPBLAS_STATUS_INTERNAL_ERROR:
            return "HIPBLAS_STATUS_INTERNAL_ERROR";

        case HIPBLAS_STATUS_NOT_SUPPORTED:
            return "HIPBLAS_STATUS_NOT_SUPPORTED";

        case HIPBLAS_STATUS_ARCH_MISMATCH:
            return "HIPBLAS_STATUS_ARCH_MISMATCH";

        case HIPBLAS_STATUS_HANDLE_IS_NULLPTR:
            return "HIPBLAS_STATUS_HANDLE_IS_NULLPTR";
    }
#endif

    return "<unknown>";
}
}  // namespace operation

// =======================================
// Forwarding functions of to GPU BLAS API
// =======================================
template <typename... ARGS>
inline auto create(ARGS&&... args) -> StatusType {
#if defined(TILED_MM_CUDA)
  return cublasCreate(std::forward<ARGS>(args)...);
#else
  return hipblasCreate(std::forward<ARGS>(args)...);
#endif
}

template <typename... ARGS>
inline auto destroy(ARGS&&... args) -> StatusType {
#if defined(TILED_MM_CUDA)
  return cublasDestroy(std::forward<ARGS>(args)...);
#else
  return hipblasDestroy(std::forward<ARGS>(args)...);
#endif
}

template <typename... ARGS>
inline auto set_stream(ARGS&&... args) -> StatusType {
#if defined(TILED_MM_CUDA)
  return cublasSetStream(std::forward<ARGS>(args)...);
#else
  return hipblasSetStream(std::forward<ARGS>(args)...);
#endif
}

template <typename... ARGS>
inline auto sgemm(ARGS&&... args) -> StatusType {
#if defined(TILED_MM_CUDA)
  return cublasSgemm(std::forward<ARGS>(args)...);
#else
  return hipblasSgemm(std::forward<ARGS>(args)...);
#endif // TILED_MM_ROCBLAS_HAS_SGEMM
}

template <typename... ARGS>
inline auto dgemm(ARGS&&... args) -> StatusType {
#if defined(TILED_MM_CUDA)
  return cublasDgemm(std::forward<ARGS>(args)...);
#else
  return hipblasDgemm(std::forward<ARGS>(args)...);
#endif // TILED_MM_CUDA
}

template <typename... ARGS>
inline auto cgemm(ARGS&&... args) -> StatusType {
#if defined(TILED_MM_CUDA)
  return cublasCgemm(std::forward<ARGS>(args)...);
#else
  return hipblasCgemm(std::forward<ARGS>(args)...);
#endif // TILED_MM_CUDA
}

template <typename... ARGS>
inline auto zgemm(ARGS&&... args) -> StatusType {
#if defined(TILED_MM_CUDA)
  return cublasZgemm(std::forward<ARGS>(args)...);
#else
  return hipblasZgemm(std::forward<ARGS>(args)...);
#endif // TILED_MM_CUDA
}

}  // namespace blas_api
}  // namespace gpu

