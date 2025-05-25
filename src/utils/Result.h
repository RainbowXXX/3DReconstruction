//
// Created by rainbowx on 25-5-9.
//

#ifndef RESULT_H
#define RESULT_H

#include <string>
#include <optional>
#include <stdexcept>

template<typename Ty>
class Result {
    std::string reason_;
    std::optional<Ty> value_;

protected:
    explicit Result(const Ty &val) : value_(val) { }
    explicit Result(Ty &&val) : value_(std::move(val)) { }

    Result(std::string reason, std::optional<Ty> val): reason_{std::move(reason)}, value_{val} {}

public:
    static Result Ok(const Ty &val) { return Result(val); }
    static Result Ok(Ty &&val) { return Result(std::move(val)); }
    static Result Err(const std::string &err) { return Result(err, std::nullopt); }
    static Result Err(std::string &&err) { return Result(std::move(err), std::nullopt); }

    // Checkers
    [[nodiscard]] bool is_ok() const { return value_.has_value(); }
    [[nodiscard]] bool is_err() const { return !value_.has_value(); }

    // Accessors
    const Ty &unwrap() const {
        if (!value_) {
            throw std::runtime_error("Called unwrap on Err: " + reason_);
        }
        return *value_;
    }

    Ty &unwrap() {
        if (!value_) {
            throw std::runtime_error("Called unwrap on Err: " + reason_);
        }
        return *value_;
    }

    [[nodiscard]] const std::string &unwrap_err() const {
        if (value_) {
            throw std::runtime_error("Called unwrap_err on Ok");
        }
        return reason_;
    }

    // Optional access
    const std::optional<Ty> &ok() const { return value_; }
    [[nodiscard]] const std::string &err() const { return reason_; }
};

#endif //RESULT_H
