#pragma once

#include <any>
#include <optional>
#include <print>
#include <string>
#include <unordered_map>

class AnyBag {
   public:
    template <typename T>
    void set(const std::string& key, T&& value) {
        data_[key] = std::forward<T>(value);
    }

    template <typename T>
    std::optional<T> get(const std::string& key) const {
        const std::optional<const T*> ptr = getPtr<T>(key);

        if (!ptr) {
            return std::nullopt;
        }
        return *ptr.value();
    }

    template <typename T>
    std::optional<T> get(const std::string& key) {
        const std::optional<T*> ptr = getPtr<T>(key);

        if (!ptr) {
            return std::nullopt;
        }

        return *ptr.value();
    }

    template <typename T>
    std::optional<const T*> getPtr(const std::string& key) const {
        auto item = data_.find(key);
        if (item == data_.end()) {
            return std::nullopt;
        }
#ifndef NDEBUG
        if (item->second.type() != typeid(T)) {
            std::println(
                stderr,
                "Type mismatch for key \"{}\": stored {}, requested {}\n", key,
                item->second.type().name(), typeid(T).name());
        }
#endif
        const T* ptr = std::any_cast<T>(&item->second);
        if (ptr == nullptr) {
            return std::nullopt;
        }

        return {ptr};
    }

    template <typename T>
    std::optional<T*> getPtr(const std::string& key) {
        auto item = data_.find(key);
        if (item == data_.end()) {
            return std::nullopt;
        }
#ifndef NDEBUG
        if (item->second.type() != typeid(T)) {
            std::println(
                stderr,
                "Type mismatch for key \"{}\": stored {}, requested {}\n", key,
                item->second.type().name(), typeid(T).name());
        }
#endif
        T* ptr = std::any_cast<T>(&data_[key]);
        if (ptr == nullptr) {
            return std::nullopt;
        }

        return {ptr};
    }

    bool has(const std::string& key) const { return data_.contains(key); }

    bool erase(const std::string& key) { return data_.erase(key) > 0; }

   private:
    std::unordered_map<std::string, std::any> data_;
};
