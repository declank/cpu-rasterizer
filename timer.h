#pragma once

#include <algorithm>
#include <chrono>
#include <numeric>


template<typename T>
double getAverage(std::vector<T> const& v)
{
  if (v.empty())
  {
    return 0;
  }

  return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

struct TimerContext {
  std::map<std::string, std::vector<double> > timings;

  void printTimings()
  {
    std::cout << timings.size() << '\n';
    for (auto const& timingKvPair : timings)
    {
      auto runs = timingKvPair.second.size();
      auto minmax = std::minmax_element(timingKvPair.second.begin(), timingKvPair.second.end());
      double average = getAverage(timingKvPair.second);
      double total = std::accumulate(timingKvPair.second.begin(), timingKvPair.second.end(), 0.0);

      std::cout << "=== " << timingKvPair.first
#ifdef _DEBUG
        << " - DEBUG build"
#else
        << " - RELEASE build"
#endif
        << " ============\n";
      std::cout << "Runs:\t" << runs << "\tMin:\t" << *minmax.first * 1000.0 << "\tMax:\t" << *minmax.second * 1000.0
        << "\tAverage:\t" << average * 1000.0 << "\tTotal:\t" << total * 1000.0 << '\n';
    }
  }
};

class Timer {
public:
  Timer(std::string name, TimerContext& context) :
    name_(name),
    context_(context),
    clock_(std::chrono::high_resolution_clock::now()) {}

  ~Timer() {
    double time = std::chrono::duration_cast<std::chrono::duration<double>>(
      std::chrono::high_resolution_clock::now() - clock_
      ).count();
    context_.timings[name_].push_back(time);
    //context_.timings.insert()
  }

private:
  std::string name_;
  TimerContext& context_;
  std::chrono::high_resolution_clock::time_point clock_;
};
