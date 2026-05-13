#pragma once

#include <cassert>
#include <optional>
#include <print>
#include <ranges>
#include <string>
#include <unordered_map>
#include <vector>

#include "unique_any.hpp"

// Save old LOG_ID (which is unlikely)
#ifdef LOG_ID
#define ANYBAG_OLD_LOG_ID LOG_ID
#endif

#define LOG_ID "[AnyBag]"

// Type-erased storage for both move-only and copyable types, with copy getter
// only available for copyable types
class AnyBag {
   public:
    // Set value
    void set(const std::string& key, auto&& value) {
#ifndef NDEBUG
        auto msg
            = std::format("Adding value to AnyBag. key:'{}', value type:{}",
                          key, typeid(value).name());
        std::println(stderr, "{}", msg);
        logs_.emplace_back(std::move(msg));
#endif

        data_.insert_or_assign(key, std::forward<decltype(value)>(value));
    }

    // Copy. Only usable when T is copy-constructible.
    // Use take(key) for moving values out of the storage
    template <typename T>
        requires std::is_copy_constructible_v<T>
    std::optional<T> get(const std::string& key) const {
        const T* p = getRawPtr<T>(key);
        if (!p) {
            return std::nullopt;
        }
        return *p;
    }

    template <typename T>
    std::optional<const T*> getPtr(const std::string& key) const {
        const T* p = getRawPtr<T>(key);
        if (!p) {
            return std::nullopt;
        }
        return p;
    }

    template <typename T>
    std::optional<T*> getPtr(const std::string& key) {
        T* p = getRawPtr<T>(key);
        if (!p) {
            return std::nullopt;
        }
        return p;
    }

    // Move the stored value out. Leaves the stored object in its
    // moved-from state
    //
    // The entry still exists; erase() afterwards if
    // you want it gone
    template <typename T>
    std::optional<T> take(const std::string& key,
                          bool erase_after_move = true) {
        T* p = getRawPtr<T>(key);
        if (!p) {
            return std::nullopt;
        }

        auto result = std::move(*p);

        if (erase_after_move) {
            assert(erase(key) && "Delete moved-out value");
        }

        return std::move(result);
    }

    bool has(const std::string& key) const { return data_.contains(key); }
    bool erase(const std::string& key) { return data_.erase(key) > 0; }

#ifndef NDEBUG
    [[nodiscard]] std::vector<std::string> getLogs() const { return logs_; }

    void printLogs() const {
        const auto lines = std::ranges::owning_view{
            logs_ | std::views::join_with(std::string_view("\n\t"))};

        std::println(stderr, "Logs: \n{}", lines);
    }
#endif

   private:
    std::unordered_map<std::string, Any> data_;

#ifndef NDEBUG
    mutable std::vector<std::string> logs_;
#endif

    // Shared lookup logic. Returns nullptr on missing key or type mismatch.
    template <typename T>
    const T* getRawPtr(const std::string& key) const {
        auto it = data_.find(key);

#ifndef NDEBUG
        const auto val_info
            = it == data_.end()
                  ? std::format("value:null, requested type:{}",
                                typeid(T).name())
                  : std::format("type:{}", it->second.getType().name());

        auto msg = std::format("Getting value from anybag. Key: {}, {}", key,
                               val_info);
        std::println(stderr, "{}", msg);
        logs_.emplace_back(std::move(msg));
#endif

        if (it == data_.end()) {
            return nullptr;
        }

#ifndef NDEBUG
        if (it->second.getType() != typeid(T)) {
            std::println(
                stderr, "Type mismatch for key \"{}\": stored {}, requested {}",
                key, it->second.getType().name(), typeid(T).name());
        }
#endif
        return AnyCast<T>(&it->second);
    }

    template <typename T>
    T* getRawPtr(const std::string& key) {
        auto it = data_.find(key);

#ifndef NDEBUG
        const auto val_info
            = it == data_.end()
                  ? std::format("value:null, requested type:{}",
                                typeid(T).name())
                  : std::format("type:{}", it->second.getType().name());

        auto msg = std::format("Getting value from anybag. Key: {}, {}", key,
                               val_info);
        std::println(stderr, "{}", msg);
        logs_.emplace_back(std::move(msg));
#endif

        if (it == data_.end()) {
            return nullptr;
        }

#ifndef NDEBUG
        if (it->second.getType() != typeid(T)) {
            std::println(
                stderr, "Type mismatch for key \"{}\": stored {}, requested {}",
                key, it->second.getType().name(), typeid(T).name());
        }
#endif
        return AnyCast<T>(&it->second);
    }
};

#undef LOG_ID

// Restore old LOG_ID if it was saved
#ifdef ANYBAG_OLD_LOG_ID
#define LOG_ID ANYBAG_OLD_LOG_ID
#undef ANYBAG_OLD_LOG_ID
#endif
