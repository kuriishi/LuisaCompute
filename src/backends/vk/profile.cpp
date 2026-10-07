#include "profile.h"
#include "device.h"
#include "stream.h"
#include "log.h"
#include <algorithm>
#include <sstream>
#ifdef _WIN32
#include <Windows.h>
#endif
namespace lc::vk {
namespace p = luisa::compute::profile;
void profile_barriers(VkCommandBuffer buffer, VkDependencyInfo const &info) {
    auto s = p::sink();
    if (!s || !s->enabled()) { return; }
    auto emit = [&](auto const &barrier, uint64_t resource, uint64_t offset, uint64_t size) {
        auto detail = std::ostringstream{};
        detail << "{\"src_stage\":" << barrier.srcStageMask << ",\"dst_stage\":" << barrier.dstStageMask
               << ",\"src_access\":" << barrier.srcAccessMask << ",\"dst_access\":" << barrier.dstAccessMask
               << ",\"offset\":" << offset << ",\"size\":" << size << "}";
        p::emit({"barrier", "resource_dependency", s->next_id(), s->parent(), 0,
                 resource, reinterpret_cast<uint64_t>(buffer), p::clock_ns(), 0, 0, detail.str()});
    };
    for (uint32_t i = 0; i < info.bufferMemoryBarrierCount; ++i) {
        auto const &b = info.pBufferMemoryBarriers[i];
        emit(b, reinterpret_cast<uint64_t>(b.buffer), b.offset, b.size);
    }
    for (uint32_t i = 0; i < info.imageMemoryBarrierCount; ++i) {
        auto const &b = info.pImageMemoryBarriers[i];
        emit(b, reinterpret_cast<uint64_t>(b.image), b.subresourceRange.baseMipLevel, b.subresourceRange.levelCount);
    }
    for (uint32_t i = 0; i < info.memoryBarrierCount; ++i) { emit(info.pMemoryBarriers[i], 0, 0, 0); }
}
ProfileQueries::ProfileQueries(Device *device, Stream &stream, VkCommandBuffer buffer, uint32_t capacity)
    : _device{device}, _buffer{buffer}, _queue{reinterpret_cast<uint64_t>(stream.queue())} {
    p::Scope overhead{"profiling.query_prepare", 0, _queue};
    auto properties = VkPhysicalDeviceProperties{};
    vkGetPhysicalDeviceProperties(device->physical_device(), &properties);
    _period = properties.limits.timestampPeriod;
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device->physical_device(), &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device->physical_device(), &count, families.data());
    auto family = stream.stream_tag() == StreamTag::COMPUTE ? device->compute_queue_index() :
                  stream.stream_tag() == StreamTag::GRAPHICS ? device->graphics_queue_index() : device->copy_queue_index();
    _valid_bits = families[family].timestampValidBits;
    p::emit({"capability", "gpu.timestamp", 0, 0, _queue, family, _valid_bits, 0, 0, _period, properties.deviceName});
    if (!_valid_bits) { return; }
    VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = capacity * 2;
    VK_CHECK_RESULT(vkCreateQueryPool(device->logic_device(), &info, Device::alloc_callbacks(), &_pool));
    vkCmdResetQueryPool(buffer, _pool, 0, info.queryCount);
#ifdef _WIN32
    if (vkGetCalibratedTimestampsEXT && vkGetPhysicalDeviceCalibrateableTimeDomainsEXT) {
        uint32_t n = 0;
        vkGetPhysicalDeviceCalibrateableTimeDomainsEXT(device->physical_device(), &n, nullptr);
        std::vector<VkTimeDomainEXT> domains(n);
        vkGetPhysicalDeviceCalibrateableTimeDomainsEXT(device->physical_device(), &n, domains.data());
        if (std::find(domains.begin(), domains.end(), VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT) != domains.end()) {
            VkCalibratedTimestampInfoEXT infos[2] = {
                {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT, nullptr, VK_TIME_DOMAIN_DEVICE_EXT},
                {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT, nullptr, VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_EXT}};
            uint64_t times[2]{};
            if (vkGetCalibratedTimestampsEXT(device->logic_device(), 2, infos, times, &_deviation) == VK_SUCCESS) {
                LARGE_INTEGER frequency;
                QueryPerformanceFrequency(&frequency);
                _gpu_anchor = times[0];
                _cpu_anchor = static_cast<uint64_t>(static_cast<long double>(times[1]) * 1.0e9L / frequency.QuadPart);
                _calibrated = true;
                p::emit({"calibration", "gpu.qpc", 0, 0, _queue, _gpu_anchor, _deviation,
                         _cpu_anchor, 0, _period});
            }
        }
    }
#endif
}
ProfileQueries::~ProfileQueries() {
    if (_pool) { vkDestroyQueryPool(_device->logic_device(), _pool, Device::alloc_callbacks()); }
}
void ProfileQueries::begin(std::string_view name, uint64_t parent) {
    if (!_pool) { return; }
    _entries.push_back({p::sink()->next_id(), parent, std::string{name}});
    vkCmdWriteTimestamp2(_buffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, _pool, static_cast<uint32_t>(_entries.size() - 1) * 2);
}
void ProfileQueries::end() {
    if (_pool) {
        vkCmdWriteTimestamp2(_buffer, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, _pool, static_cast<uint32_t>(_entries.size() - 1) * 2 + 1);
    }
}
void ProfileQueries::collect() {
    if (!_pool || _entries.empty()) { return; }
    p::Scope overhead{"profiling.query_collect", 0, _queue};
    struct Result { uint64_t ticks, available; };
    std::vector<Result> results(_entries.size() * 2);
    auto status = vkGetQueryPoolResults(_device->logic_device(), _pool, 0, static_cast<uint32_t>(results.size()),
        results.size() * sizeof(Result), results.data(), sizeof(Result), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    auto mask = _valid_bits == 64 ? ~uint64_t{0} : (uint64_t{1} << _valid_bits) - 1;
    for (size_t i = 0; i < _entries.size(); ++i) {
        auto &entry = _entries[i];
        auto start = results[i * 2];
        auto stop = results[i * 2 + 1];
        if (status != VK_SUCCESS || !start.available || !stop.available) {
            p::emit({"gap", "gpu.query_unavailable", entry.id, entry.parent, _queue});
            continue;
        }
        auto duration = static_cast<uint64_t>(((stop.ticks - start.ticks) & mask) * _period);
        // Signed modular displacement requires a capture shorter than half a wrap.
        auto delta = (start.ticks - _gpu_anchor) & mask;
        auto signed_delta = _valid_bits == 64 ? static_cast<int64_t>(delta) :
            (delta & (uint64_t{1} << (_valid_bits - 1))) ? static_cast<int64_t>(delta | ~mask) : static_cast<int64_t>(delta);
        auto begin = _calibrated ? static_cast<uint64_t>(static_cast<long double>(_cpu_anchor) + signed_delta * static_cast<long double>(_period)) :
                                  static_cast<uint64_t>((start.ticks & mask) * _period);
        p::emit({_calibrated ? "gpu" : "gpu_uncalibrated", entry.name, entry.id, entry.parent,
                 _queue, start.ticks, stop.ticks, begin, begin + duration, static_cast<double>(_deviation),
                 "measured:top-bottom envelope; not exclusive service time"});
    }
}
}
