#pragma once

#include "kvmem/resource_allocation.hpp"

namespace kvmem {
struct NvmeAllocationTag;
using NvmeAllocationHandle = ResourceAllocationHandle<NvmeAllocationTag>;
using NvmeAllocationRegistry = ResourceAllocationRegistry<NvmeAllocationTag>;
using NvmeReleaseResult = ResourceReleaseResult;
using NvmeIoLease = NvmeAllocationRegistry::IoLease;
} // namespace kvmem
