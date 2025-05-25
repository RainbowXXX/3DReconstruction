//
// Created by rainbowx on 25-5-11.
//

#ifndef EVENT_H
#define EVENT_H

#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class Event {
protected:
    std::string type_;
    double progress_;

    // 私有构造函数，禁止直接构造
    Event() : progress_(0) {}

public:
    // 禁止复制构造或直接创建（可选）
    Event(const Event&) = default;
    Event& operator=(const Event&) = default;

    // 静态工厂方法
    static Event create() {
        return {};
    }

    // 链式设置方法
    Event& setType(const std::string& type) {
        type_ = type;
        return *this;
    }

    Event& setProgress(double progress) {
        progress_ = progress;
        return *this;
    }

    // JSON 序列化方法
    [[nodiscard]] json toJson() const {
        return json{
                {"type", type_},
                {"progress", progress_}
        };
    }

    // 友元输出，便于调试
    friend std::ostream& operator<<(std::ostream& os, const Event& evt) {
        os << evt.toJson().dump();
        return os;
    }
};


#endif //EVENT_H
