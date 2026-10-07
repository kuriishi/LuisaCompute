#pragma once
#include <luisa/core/dll_export.h>
#include <cstdint>
#include <string_view>

namespace luisa::compute::profile {
// Optional process-wide observer. The owner must drain streams before removing it.
struct Event {
    std::string_view kind, name;
    uint64_t id{}, parent{}, queue{}, object{}, value{};
    uint64_t begin{}, end{};
    double amount{};
    std::string_view detail;
};
class Sink {
public:
    virtual ~Sink() = default;
    virtual bool enabled() const noexcept = 0;
    virtual bool wants_metric(std::string_view) const noexcept { return false; }
    virtual uint64_t command(uint32_t tag) noexcept = 0;
    virtual uint64_t next_id() noexcept = 0;
    virtual uint64_t parent() const noexcept = 0;
    virtual void record(Event const &) noexcept = 0;
};
LUISA_RUNTIME_API void set_sink(Sink *) noexcept;
LUISA_RUNTIME_API Sink *sink() noexcept;
LUISA_RUNTIME_API uint64_t clock_ns() noexcept;
LUISA_RUNTIME_API uint64_t command_created(uint32_t tag) noexcept;
inline void emit(Event const &event) noexcept {
    if (auto s = sink()) { s->record(event); }
}
class Scope {
    Sink *_sink{};
    Event _event;
public:
    Scope(std::string_view name, uint64_t parent = 0, uint64_t queue = 0,
          uint64_t object = 0, uint64_t value = 0) noexcept
        : _event{"cpu", name, 0, parent, queue, object, value} {
        if (auto s = sink(); s && s->enabled()) {
            _sink = s;
            _event.id = s->next_id();
            if (!_event.parent) { _event.parent = s->parent(); }
            _event.begin = clock_ns();
        }
    }
    ~Scope() noexcept { finish(); }
    void finish() noexcept {
        if (_sink) {
            _event.end = clock_ns();
            _sink->record(_event);
            _sink = nullptr;
        }
    }
    Scope(Scope const &) = delete;
    Scope &operator=(Scope const &) = delete;
};
}
