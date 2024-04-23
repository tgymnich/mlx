// Copyright © 2023-2024 Apple Inc.
#include <cstdlib>
#include <memory>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::metal {

bool is_available() {
  return true;
}

int max_ops_per_buffer() {
  auto get_val = []() {
    if (const char* buff_str = std::getenv("MLX_MAX_OPS_PER_BUFFER")) {
      return atoi(buff_str);
    } else {
      return 10;
    }
  };
  static int max_ops_per_buffer_ = get_val();
  return max_ops_per_buffer_;
}

#define MAX_OPS_PER_BUFFER max_ops_per_buffer()

inline void check_error(MTL::CommandBuffer* cbuf) {
  if (cbuf->status() == MTL::CommandBufferStatusError) {
    std::ostringstream msg;
    msg << "[METAL] Command buffer execution failed: "
        << cbuf->error()->localizedDescription()->utf8String();
    throw std::runtime_error(msg.str());
  }
}

std::function<void()> make_task(array arr, bool signal) {
  auto task = [arr = std::move(arr), signal]() mutable {
    auto pool = new_scoped_memory_pool();
    auto s = arr.primitive().stream();

    auto& d = metal::device(s.device);

    // Fetch an active command buffer
    auto& command_buffer = d.get_command_buffer(s.index);
    command_buffer.ops++;

    for (auto& input : arr.inputs()) {
      if (input.event().valid() &&
          input.event().stream() != arr.primitive().stream()) {
        // TODO, consider committing the buffer and encoding a wait in the new
        // buffer rather than on the task thread
        input.event().wait();
      }
    }

    // Remove existing input arrays from the command buffer
    // so they can be donated
    command_buffer.remove_arrays(arr.inputs());

    auto outputs = arr.outputs();
    {
      // If the array is a tracer hold a reference
      // to its inputs so they don't get donated
      std::vector<array> inputs;
      if (arr.is_tracer()) {
        inputs = arr.inputs();
      }

      debug_set_primitive_buffer_label(command_buffer, arr.primitive());
      arr.primitive().eval_gpu(arr.inputs(), outputs);
    }
    command_buffer.add_input_arrays(arr.inputs());
    command_buffer.add_output_arrays(arr.siblings());
    if (!arr.is_tracer()) {
      arr.detach();
    }

    if (signal || command_buffer.ops >= MAX_OPS_PER_BUFFER) {
      d.end_encoding(s.index);
      auto donated_buffers = std::move(command_buffer.out_donated_buffers);
      donated_buffers.insert(
          std::make_move_iterator(command_buffer.in_donated_buffers.begin()),
          std::make_move_iterator(command_buffer.in_donated_buffers.end()));

      decltype(donated_buffers) capture_buffers;
      if (signal) {
        capture_buffers = std::move(donated_buffers);
      }

      command_buffer->encodeSignalEvent(
          static_cast<MTL::Event*>(arr.event().raw_event().get()),
          arr.event().value());
      scheduler::notify_new_task(s);
      command_buffer->addCompletedHandler(
          [s,
           event = arr.event(),
           buffers = std::move(command_buffer.buffers),
           capture_buffers =
               std::move(capture_buffers)](MTL::CommandBuffer* cbuf) {
            scheduler::notify_task_completion(s);
            check_error(cbuf);
          });
      d.commit_command_buffer(s.index);

      if (!signal) {
        auto& new_cb = d.get_command_buffer(s.index);
        new_cb.in_donated_buffers = std::move(donated_buffers);
      }
    }
  };
  return task;
}

std::function<void()> make_synchronize_task(
    Stream s,
    std::shared_ptr<std::promise<void>> p) {
  return [s, p = std::move(p)]() {
    auto& d = metal::device(s.device);
    auto& cb = d.get_command_buffer(s.index);
    d.end_encoding(s.index);
    cb->addCompletedHandler([p = std::move(p)](MTL::CommandBuffer* cbuf) {
      check_error(cbuf);
      p->set_value();
    });
    d.commit_command_buffer(s.index);
  };
}

void start_capture(std::string path, id object) {
  auto pool = new_scoped_memory_pool();

  auto descriptor = MTL::CaptureDescriptor::alloc()->init();
  descriptor->setCaptureObject(object);

  if (!path.empty()) {
    auto string = NS::String::string(path.c_str(), NS::UTF8StringEncoding);
    auto url = NS::URL::fileURLWithPath(string);
    descriptor->setDestination(MTL::CaptureDestinationGPUTraceDocument);
    descriptor->setOutputURL(url);
  }

  auto manager = MTL::CaptureManager::sharedCaptureManager();
  NS::Error* error;
  bool started = manager->startCapture(descriptor, &error);
  descriptor->release();
  if (!started) {
    std::ostringstream msg;
    msg << "[metal::start_capture] Failed to start: "
        << error->localizedDescription()->utf8String();
    throw std::runtime_error(msg.str());
  }
}

void start_capture(std::string path) {
  auto& device = metal::device(mlx::core::Device::gpu);
  return start_capture(path, device.mtl_device());
}

void stop_capture() {
  auto pool = new_scoped_memory_pool();
  auto manager = MTL::CaptureManager::sharedCaptureManager();
  manager->stopCapture();
}

} // namespace mlx::core::metal
