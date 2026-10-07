#pragma once
#include <luisa/runtime/profile.h>
#include <volk.h>
#include <memory>
#include <vector>
#include <string>
namespace lc::vk {
class Device;
class Stream;
void profile_barriers(VkCommandBuffer buffer, VkDependencyInfo const &info);
class ProfileQueries {
    struct Entry { uint64_t id, parent; std::string name; };
    Device *_device;
    VkQueryPool _pool{};
    VkCommandBuffer _buffer;
    uint64_t _queue;
    double _period{};
    uint32_t _valid_bits{};
    uint64_t _gpu_anchor{}, _cpu_anchor{}, _deviation{};
    bool _calibrated{};
    std::vector<Entry> _entries;
public:
    ProfileQueries(Device *device, Stream &stream, VkCommandBuffer buffer, uint32_t capacity);
    ~ProfileQueries();
    void begin(std::string_view name, uint64_t parent);
    void end();
    void collect();
};
}
