#pragma once
#include <string>
#include <filesystem>

extern std::string get_exe_path();
std::string get_resolved_exe_name();
std::filesystem::path recursively_find_file(
    const std::string & filename,
    std::filesystem::path basedir = "");
