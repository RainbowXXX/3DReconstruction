//
// Created by rainbowx on 25-5-11.
//

#ifndef MYPROGRESS_H
#define MYPROGRESS_H
#include "common/common.h"
#include "system/progressinterface.hpp"

class MyProgress: public openMVG::system::ProgressInterface {
public:
  explicit MyProgress(
      const std::string& name,
      const std::uint32_t expected_count = 1
      ) noexcept
      : ProgressInterface(expected_count)
  {
    name_ = name;
    MyProgress::Restart(expected_count, name);
  }

  void Restart(const std::uint32_t expected_count, const std::string& msg = {}) override
  {
    ProgressInterface::Restart(expected_count);
    event_queue.push(::Event::create().setType(name_).setProgress(0));
  }

  inline std::uint32_t operator+=(const std::uint32_t increment) override
  {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto res = ProgressInterface::operator+=(increment);
    const auto percentage = Percent();
    event_queue.push(::Event::create().setType(name_).setProgress(percentage));
    return res;
  }

  inline bool Increment(const int increment = 1)
  {
    std::ostringstream os;
    std::lock_guard<std::mutex> lock(mutex_);
    ProgressInterface::operator+=(increment);
    const auto percentage = Percent();
    return true;
  }

   inline std::uint32_t operator++() override
   {
     return this->operator+=(1);
   }

  [[nodiscard]] inline std::string PercentString() const
  {
    return name_ + " " + std::to_string(Percent()) + "%";
  }

 private:
  std::string name_;
  std::mutex mutex_;
};

#endif //MYPROGRESS_H
