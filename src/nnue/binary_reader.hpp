#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <type_traits>

namespace NNUE {

class BinaryReader {
public:
    explicit BinaryReader(const std::filesystem::path & path)
        : stream_(path, std::ios::binary)
    {
        if (!stream_) {
            error_ = "cannot open file";
            return;
        }

        stream_.seekg(0, std::ios::end);
        const auto end = stream_.tellg();
        if (end < 0) {
            error_ = "cannot determine file size";
            return;
        }

        size_ = static_cast<size_t>(end);
        stream_.seekg(0, std::ios::beg);
    }

    explicit operator bool() const { return stream_.is_open() && error_.empty(); }
    size_t size() const { return size_; }
    size_t position() { return static_cast<size_t>(stream_.tellg()); }
    const std::string & error() const { return error_; }

    bool seek(size_t offset) {
        if (offset > size_)
            return fail("seek beyond end of file");

        stream_.clear();
        stream_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
        return stream_ ? true : fail("seek failed");
    }

    bool read(void * destination, size_t bytes, const char * label) {
        stream_.read(static_cast<char *>(destination), static_cast<std::streamsize>(bytes));
        if (stream_.gcount() != static_cast<std::streamsize>(bytes))
            return fail(std::string("short read at ") + label);
        return true;
    }

    template<typename T>
    bool read_little_endian(T & value, const char * label) {
        static_assert(std::is_integral_v<T>);
        std::uint8_t bytes[sizeof(T)];
        if (!read(bytes, sizeof(bytes), label))
            return false;

        using Unsigned = std::make_unsigned_t<T>;
        Unsigned decoded = 0;
        for (size_t i = 0; i < sizeof(T); ++i)
            decoded |= static_cast<Unsigned>(bytes[i]) << (i * 8);
        std::memcpy(&value, &decoded, sizeof(value));
        return true;
    }

    bool at_end() {
        return position() == size_;
    }

private:
    bool fail(std::string message) {
        error_ = std::move(message);
        return false;
    }

    std::ifstream stream_;
    size_t size_ = 0;
    std::string error_;
};

} // namespace NNUE
