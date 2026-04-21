#pragma once

#include <memory>
#include <type_traits>
#include <typeinfo>
#include <utility>

class Any {
    struct base {
        virtual ~base() = default;
        [[nodiscard]] virtual const std::type_info& type() const noexcept = 0;
    };

    template <class T>
    struct model final : base {
        T value;

        template <class... Args>
        explicit model(std::in_place_t _, Args&&... args)
            : value(std::forward<Args>(args)...) {}

        [[nodiscard]] const std::type_info& type() const noexcept override {
            return typeid(T);
        }
    };

    std::unique_ptr<base> ptr_;

   public:
    // Construction
    constexpr Any() noexcept = default;

    Any(const Any&) = delete;
    Any& operator=(const Any&) = delete;

    Any(Any&&) noexcept = default;
    Any& operator=(Any&&) noexcept = default;

    // Construct from a value. Excluded from overload resolution when the
    // argument is itself a Any (so the move ctor wins) or an
    // in_place_type_t (handled by the other constructor).
    template <class T, class D = std::decay_t<T>>
        requires(!std::is_same_v<D, Any>
                 && !std::is_same_v<D, std::in_place_type_t<D>>
                 && std::is_constructible_v<D, T>)
    Any(T&& value)
        : ptr_(std::make_unique<model<D>>(std::in_place,
                                          std::forward<T>(value))) {}

    // In-place construction: Any a(std::in_place_type<T>, args...);
    template <class T, class... Args>
    explicit Any(std::in_place_type_t<T> _, Args&&... args)
        requires(std::is_constructible_v<std::decay_t<T>, Args...>)
        : ptr_(std::make_unique<model<std::decay_t<T>>>(
              std::in_place, std::forward<Args>(args)...)) {}

    // Assignment from a value
    template <class T, typename D = std::decay_t<T>>
        requires(!std::is_same_v<D, Any> && std::is_constructible_v<D, T>)
    Any& operator=(T&& value) {
        ptr_
            = std::make_unique<model<D>>(std::in_place, std::forward<T>(value));
        return *this;
    }

    // Emplace a new value, destroying any existing one. Returns a reference
    // to the newly constructed object.
    template <class T, class... Args>
    std::decay_t<T>& emplace(Args&&... args) {
        using D = std::decay_t<T>;
        auto owned = std::make_unique<model<D>>(std::in_place,
                                                std::forward<Args>(args)...);
        D& ref = owned->value;
        ptr_ = std::move(owned);
        return ref;
    }

    // Observers
    void reset() noexcept { ptr_.reset(); }
    void swap(Any& other) noexcept { ptr_.swap(other.ptr_); }
    [[nodiscard]] bool hasValue() const noexcept {
        return static_cast<bool>(ptr_);
    }
    [[nodiscard]] const std::type_info& getType() const noexcept {
        return ptr_ ? ptr_->type() : typeid(void);
    }

    // Casting
    template <class T>
    friend const T* AnyCast(const Any* any) noexcept;
    template <class T>
    friend T* AnyCast(Any* any) noexcept;
};

// Pointer-form any_cast: returns nullptr on type mismatch or empty.
template <class T>
T* AnyCast(Any* any) noexcept {
    if (!any || !any->ptr_ || any->ptr_->type() != typeid(T)) {
        return nullptr;
    }
    return &static_cast<Any::model<T>*>(any->ptr_.get())->value;
}

template <class T>
const T* AnyCast(const Any* any) noexcept {
    if (!any || !any->ptr_ || any->ptr_->type() != typeid(T)) {
        return nullptr;
    }
    return &static_cast<const Any::model<T>*>(any->ptr_.get())->value;
}

// Reference-form any_cast: throws on mismatch, like std::any_cast.
template <class T>
T AnyCast(Any& a) {
    using U = std::remove_cv_t<std::remove_reference_t<T>>;
    auto* p = AnyCast<U>(&a);
    if (!p) {
        throw std::bad_cast{};
    }
    return static_cast<T>(*p);
}

template <class T>
T AnyCast(const Any& a) {
    using U = std::remove_cv_t<std::remove_reference_t<T>>;
    auto* p = AnyCast<U>(&a);
    if (!p) {
        throw std::bad_cast{};
    }
    return static_cast<T>(*p);
}

template <class T>
T AnyCast(Any&& a) {
    using U = std::remove_cv_t<std::remove_reference_t<T>>;
    auto* p = AnyCast<U>(&a);
    if (!p) {
        throw std::bad_cast{};
    }
    return static_cast<T>(std::move(*p));
}

// make_Any, mirroring std::make_any
template <class T, class... Args>
Any MakeAny(Args&&... args) {
    return Any(std::in_place_type<T>, std::forward<Args>(args)...);
}
