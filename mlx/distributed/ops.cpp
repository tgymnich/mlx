// Copyright © 2024 Apple Inc.

#include "mlx/distributed/ops.h"
#include "mlx/distributed/primitives.h"

namespace mlx::core::distributed {

namespace {

Group to_group(std::optional<Group> group) {
  if (group.has_value()) {
    return group.value();
  } else {
    return distributed::init();
  }
}

} // namespace

array all_reduce_sum(const array& x, std::optional<Group> group_) {
  return all_reduce_sum(std::vector<array>{x}, std::move(group_))[0];
}

std::vector<array> all_reduce_sum(
    const std::vector<array>& xs,
    std::optional<Group> group_) {
  auto group = to_group(group_);

  if (group.size() == 1) {
    return xs;
  }

  std::vector<std::vector<int>> shapes;
  std::vector<Dtype> dtypes;
  shapes.reserve(xs.size());
  dtypes.reserve(xs.size());

  for (const auto& x : xs) {
    shapes.push_back(x.shape());
    dtypes.push_back(x.dtype());
  }

  return array::make_arrays(
      std::move(shapes),
      std::move(dtypes),
      std::make_shared<AllReduce>(group, AllReduce::Sum),
      xs);
}

array all_gather(const array& x, std::optional<Group> group_) {
  auto group = to_group(group_);

  if (group.size() == 1) {
    return x;
  }

  auto result_shape = x.shape();
  if (result_shape.size() == 0) {
    result_shape.push_back(group.size());
  } else {
    result_shape[0] *= group.size();
  }
  return array(
      std::move(result_shape),
      x.dtype(),
      std::make_shared<AllGather>(group),
      {x});
}

} // namespace mlx::core::distributed
