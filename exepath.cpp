#include <string>
#include <filesystem>

#if defined(_WIN32)
    #include <windows.h>
#elif defined(__linux__)
    #include <unistd.h>
    #include <limits.h>
#elif defined(__APPLE__)
    #include <mach-o/dyld.h>
#endif


std::string get_exe_path() {
    char path[4096];

#if defined(_WIN32)
    GetModuleFileNameA(nullptr, path, MAX_PATH);
#elif defined(__linux__)
    ssize_t len = readlink("/proc/self/exe", path, sizeof(path) - 1);
    if (len != -1) path[len] = '\0';
#elif defined(__APPLE__)
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) != 0) path[0] = '\0';
#endif

    return std::string(path);
}

std::filesystem::path recursively_find_file(
    const std::string& filename,
    std::filesystem::path basedir)
{
    if (basedir.empty())
        basedir = get_exe_path();
    auto const exe_dir = basedir.parent_path();
    std::filesystem::path currentDir = basedir;
    while (!currentDir.empty()) {
        if (std::filesystem::exists(currentDir) && std::filesystem::is_directory(currentDir)) {
            for (const auto& entry : std::filesystem::recursive_directory_iterator(currentDir)) {
                if (entry.path().filename() == filename) {
                    return entry.path();
                }
            }
        }
        currentDir = currentDir.parent_path();
    }
    return {};
}

