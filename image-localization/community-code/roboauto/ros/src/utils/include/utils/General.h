/*
 *  Copyright 2016 RoboAuto team, Artin
 *  All rights reserved.
 *
 *  This file is part of RoboAuto HorizonSlam.
 *
 *  RoboAuto HorizonSlam is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  RoboAuto HorizonSlam is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with RoboAuto HorizonSlam.  If not, see <http://www.gnu.org/licenses/>.
 */

/**
 * @author: RoboAuto Team
 * @brief: General utilities that didn't fall into other categories.
 */

#pragma once
#include <iostream>
#include <chrono>

namespace utils {
    /**
     * @brief: Custom NAN substitute we used to signalize default (non-set) value because of -Ofast (-ffast-math).
     */
    constexpr double FMATH_NAN = -11119.00;

    /**
     * @brief: Prints time elapsed since time point in microseconds.
     * @details: Declare a time point variable using some kind of std::chrono clock
     *           at the beginning of the code segment that you want to time and then
     *           pass the variable to this function at the end of that segment.
     */
    template<typename timePoint_t>
    inline void ElapsedTimeMicro(const timePoint_t &tp, const std::string& marker = "") {
        using namespace std::chrono;
        auto endTime = timePoint_t::clock::now();
        auto timeElapsed = duration_cast<microseconds>(endTime - tp).count();
        std::cout << marker << (marker == "" ? "" : ":\t") << "time elapsed: " << timeElapsed << " microseconds\n";
    }

    /**
     * @brief: Same as ElapsedTimeMicro, only in milliseconds.
     */
    template<typename timePoint_t>
    inline void ElapsedTimeMilli(const timePoint_t &tp, const std::string& marker = "") {
        using namespace std::chrono;
        auto endTime = timePoint_t::clock::now();
        auto timeElapsed = duration_cast<milliseconds>(endTime - tp).count();
        std::cout << marker << (marker == "" ? "" : ":\t") << "time elapsed: " << timeElapsed << " milliseconds\n";
    }

    /**
     * @brief: Same as ElapsedTimeMilli, but includes average elapsed time.
     * @details: Use thread_local/member/static variables for total and counter params.
     */
    template<typename timePoint_t>
    inline void ElapsedTimeMilliAvg(const timePoint_t &tp, uint64_t& total, uint32_t& counter, const std::string& marker = "") {
        using namespace std::chrono;
        auto endTime = timePoint_t::clock::now();
        auto timeElapsed = duration_cast<milliseconds>(endTime - tp).count();
        total += timeElapsed;
        std::cout << marker << (marker == "" ? "" : ":\t") << "time elapsed: " << timeElapsed << " milliseconds" << " | avg time: " << total/++counter << "\n";
    }

    /**
     * @brief: A shorthand for std::chrono::steady_clock::now(), because it's too long and bothersome to remember.
     */
    inline auto now()
    {
        return std::chrono::steady_clock::now();
    }

    template<typename T1, typename T2>
    inline bool between(const T1 __x, const T2 min, const T2 max)
    {
        return (__x >= min && __x <= max);
    }

    /**
     * @brief: A substitute for isnan function, because of our NAN substitute.
     */
    template <typename T>
    inline bool isfnan(T __x)
    {
        return between(__x, FMATH_NAN - 1, FMATH_NAN + 1);
    }

    template <typename T1>
    inline T1 clamp(T1 val, T1 min, T1 max)
    {
        return std::max(std::min(max, val), min);
    }
}