#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "util.hpp"

namespace gpu {

// wrapper around cuda events
class cuda_event {
  public:

    cuda_event() {
        auto status = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
        cuda_check_status(status);
        valid_ = true;
    }

    ~cuda_event() {
        // note that cudaEventDestroy can be called on an event before is has
        // been reached in a stream, and the CUDA runtime will defer clean up
        // of the event until it has been completed.
        if (valid_) {
            auto status = cudaEventDestroy(event_);
            cuda_check_status(status);
        }
    }

    // move constructor
    cuda_event(cuda_event&& other) {
        event_ = other.event_;
        valid_ = other.valid_;
        other.valid_ = false;
    }

    // move-assignment operator
    cuda_event& operator=(cuda_event&& other) {
        if (this != &other) {
            if (valid_) {
                auto status = cudaEventDestroy(event_);
                cuda_check_status(status);
            }
            event_ = other.event_;
            valid_ = other.valid_;
            other.valid_ = false;
        }
        return *this;
    }

    // copy-constructor disabled
    cuda_event(cuda_event&) = delete;
    // copy-operator disabled
    cuda_event& operator=(cuda_event&) = delete;

    // return the underlying event handle
    cudaEvent_t& event() {
        return event_;
    }

    // force host execution to wait for event completion
    void wait() {
        auto status = cudaEventSynchronize(event_);
        cuda_check_status(status);
    }

    // returns time in seconds taken between this cuda event and another cuda event
    double time_since(cuda_event& other) {
        float time_taken = 0.0f;

        auto status = cudaEventElapsedTime(&time_taken, other.event(), event_);
        cuda_check_status(status);
        return double(time_taken/1.e3);
    }

  private:
    bool valid_ = false;
    cudaEvent_t event_;
};
}

