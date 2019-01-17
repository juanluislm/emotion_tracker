#pragma once

#include <chrono>
#include <iostream>

using namespace std;

// performance profiling
class Timer {
  std::chrono::time_point<std::chrono::high_resolution_clock> time_last;
  string name;

public:
  Timer(): total_duration(0), name("default") {}
  Timer(const string &timer_name): total_duration(0), name(timer_name) {}
  void start_timer() {
    this->time_last = std::chrono::high_resolution_clock::now();
    this->last_duration = 0;
    this->total_duration = 0;
  }

  void update_timer(const string &event_name) {
    auto time_now = std::chrono::high_resolution_clock::now();
    auto duration =  (std::chrono::duration<double, std::milli>(time_now - this->time_last)).count();
    this->total_duration += duration;
    this->last_duration = duration;
    if (event_name.compare("") != 0) {
      cout << "Timer[" << name << "]: " << event_name << " took " << duration << " ms, " << this->total_duration << " ms since start" << endl;
    }
    this->time_last = std::chrono::high_resolution_clock::now();
  }

  double get_total_duration() {
    return this->total_duration;
  }

  double get_last_duration() {
    return this->last_duration;
  }

  double total_duration;
  double last_duration;
};
