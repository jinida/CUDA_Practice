#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <format>
#include <iostream>
#define MAX_TIMER 10

class Timer
{
public:
	Timer() = delete;
	Timer(std::string _title, int _numTimer = MAX_TIMER);
	~Timer();
	int getNumTimer() const { return numTimer; }
	void setTimerTitle(std::string _title) { timerTitle = _title; }
	void setTimerName(int idx, std::string _name);
	void setTimerContents(int idx, std::string _contents);
	void start(int idx);
	void end(int idx);
	void release();
	void printReport();

private:
	int numTimer;
	int numCompleteTimer;
	bool* pTimerStart;
	bool* pTimerEnd;
	std::string timerTitle;
	std::string* pTimerName;
	std::string* pTimerContent;
	std::chrono::system_clock::time_point* pTime_begin;
	std::chrono::microseconds* pElapsed_usec;
	std::chrono::milliseconds* pElapsed_msec;

private:
	bool isValid(int idx);
};

#endif TIMER_H