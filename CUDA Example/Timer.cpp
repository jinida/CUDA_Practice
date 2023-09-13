#include "Timer.h"

#define SAFE_DELETE(p) \
	do\
	{\
		if(p != nullptr)\
		{\
			delete[] p;\
			p = nullptr;\
		}\
	}while(0)\

Timer::Timer(std::string _title, int _numTimer)
{
	timerTitle = _title;
	numTimer = _numTimer;
	numCompleteTimer = 0;
	pTimerStart = new bool[numTimer];
	pTimerEnd = new bool[numTimer];
	pTimerName = new std::string[numTimer];
	pTimerContent = new std::string[numTimer];
	pTime_begin = new std::chrono::system_clock::time_point[numTimer];
	pElapsed_usec = new std::chrono::microseconds[numTimer];
	pElapsed_msec = new std::chrono::milliseconds[numTimer];

	for (int idx = 0; idx != numTimer; ++idx)
	{
		pTimerStart[idx] = false;
		pTimerEnd[idx] = false;
		pTimerName[idx] = std::format("Timer[{}]", idx);
		pTimerContent[idx] = "";
	}
}

Timer::~Timer()
{
	SAFE_DELETE(pTimerStart);
	SAFE_DELETE(pTimerEnd);
	SAFE_DELETE(pTimerName);
	SAFE_DELETE(pTimerContent);
	SAFE_DELETE(pTime_begin);
	SAFE_DELETE(pElapsed_msec);
	SAFE_DELETE(pElapsed_usec);
}

void Timer::setTimerName(int idx, std::string _name)
{
	if (isValid(idx))
	{
		pTimerName[idx] = _name;
	}
}

void Timer::setTimerContents(int idx, std::string _content)
{
	if (isValid(idx))
	{
		pTimerContent[idx] = _content;
	}
}

void Timer::start(int idx)
{
	if (isValid(idx) && !pTimerStart[idx] && !pTimerEnd[idx])
	{
		pTime_begin[idx] = std::chrono::system_clock::now();
		pTimerStart[idx] = true;
	}
	else if (!isValid(idx))
	{
		std::cout << "This Index is Not Valid\n";
	}
	else if (pTimerEnd[idx])
	{
		std::cout << "This timer has already been calculated\n";
	}
	else if (pTimerStart[idx])
	{
		std::cout << "This Timer has already started\n";
	}
	else
	{
		std::cout << "Uknown Error\n";
	}
}

void Timer::end(int idx)
{
	if (isValid(idx) && pTimerStart[idx] && !pTimerEnd[idx])
	{
		std::chrono::system_clock::time_point endTime = std::chrono::system_clock::now();
		pElapsed_msec[idx] = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - pTime_begin[idx]);
		pElapsed_usec[idx] = std::chrono::duration_cast<std::chrono::microseconds>(endTime - pTime_begin[idx]);
		pTimerEnd[idx] = true;
		numCompleteTimer++;
	}
	else if (!isValid(idx))
	{
		std::cout << "This Index is Not Valid\n";
	}
	else if (pTimerEnd[idx])
	{
		std::cout << "This timer has already been calculated\n";
	}
	else if (!pTimerStart[idx])
	{
		std::cout << "This timer did not start\n";
	}
	else
	{
		std::cout << "Uknown Error\n";
	}
}

void Timer::release()
{
	SAFE_DELETE(pTimerStart);
	SAFE_DELETE(pTimerEnd);
	SAFE_DELETE(pTimerName);
	SAFE_DELETE(pTimerContent);
	SAFE_DELETE(pTime_begin);
	SAFE_DELETE(pElapsed_msec);
	SAFE_DELETE(pElapsed_usec);
}

void Timer::printReport()
{
	std::cout << std::format("*** {} Timer Report ***\n", timerTitle);
	std::cout << std::format("Number of Timer: {}\n", numCompleteTimer);
	for (int idx = 0; idx < numTimer; ++idx)
	{
		if (pTimerStart[idx] && pTimerEnd[idx])
		{
			std::cout << format("{:>10} {:} - Time: {}(usec)\n", pTimerName[idx], pTimerContent[idx], pElapsed_usec[idx].count());
		}
	}
	std::cout << "*** End of report ***\n";
}

bool Timer::isValid(int idx)
{
	if (idx >= numTimer)
		return false;
	else
		return true;
}
