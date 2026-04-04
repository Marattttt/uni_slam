#pragma once
#include <any>
#include <optional>
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
        auto ittem = data_.find(key);
        if (ittem == data_.end()) {
            return std::nullopt;
        }
        const T* ptr = std::any_cast<T>(&ittem->second);
        return ptr ? std::optional<T>(*ptr) : std::nullopt;
    }

    bool has(const std::string& key) const { return data_.contains(key); }

    bool erase(const std::string& key) { return data_.erase(key) > 0; }

   private:
    std::unordered_map<std::string, std::any> data_;
};
