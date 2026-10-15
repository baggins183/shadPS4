#include <cstddef>
#include "common/types.h"

namespace Shader::SharedMemory {

enum class Section : u32 {
    Guest = 0, // intentionally default
    Scratch = 1,
    NumSections
};

static constexpr auto NumSections = static_cast<size_t>(Section::NumSections);

struct Flags {
    Flags(Section _section, bool _is_gds) : section(_section), is_gds(_is_gds) {}
    Flags() : Flags(Section::Guest, false) {}

    bool is_gds;
    Section section;
};

} // namespace Shader::SharedMemory
