
#include "version.hpp"
#include <format>

const std::string g_version = std::format(
    "Enyo {} v.{}{} built {} by Petter Wahlman",
        BUILD_TYPE,
        BUILD_HASH,
        (BUILD_DIRTY ? " (dirty)" : ""),
        BUILD_DATE);
