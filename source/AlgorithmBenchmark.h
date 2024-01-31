#pragma once

#include <chrono>

using std::chrono::duration;
using std::chrono::high_resolution_clock;

class AlgorithmBenchmark
{
private:
	double totalDuration = 0;
	int repetitions = 0;

public:
	/// <summary>
	/// Returns the total duration of the current benchmark.
	/// </summary>
	/// <returns></returns>
	double GetTotalDuration() const {
		return totalDuration; 
	}

	/// <summary>
	/// Returns the average duration of the benchmark over the number of repetitions.
	/// </summary>
	/// <returns></returns>
	double GetAvgDuration() const {
		if (repetitions == 0) return 0;
		return totalDuration / repetitions;
	}

	/// <summary>
	/// Returns the repetitions of the current benchmark.
	/// </summary>
	/// <returns></returns>
	int GetRepetitions() const {
		return repetitions; 
	}

	/// <summary>
	/// Runs a benchmark of a given function with the specified number of repetitions.
	/// The higher the number of repetitions the more stable the benchmark results.
	/// </summary>
	/// <typeparam name="Func"></typeparam>
	/// <param name="func">Template function that should be run</param>
	/// <param name="repetitions">Number of repetitions that he benchmark should be run</param>
	template<typename Func>
	void RunBenchmark(Func func, int repetitions) {
		this->repetitions += repetitions;
		auto startTime = high_resolution_clock::now();
		for (int i = 0; i < repetitions; i++) {
			func();
		}
		auto endTime = high_resolution_clock::now();
		duration<double, std::milli> duration = endTime - startTime;
		this->totalDuration += duration.count();
	}

	/// <summary>
	/// Resets the repetitions and durations of the current benchmark.
	/// </summary>
	void ResetBenchmark() {
		this->totalDuration = 0;
		this->repetitions = 0;
	}
};
