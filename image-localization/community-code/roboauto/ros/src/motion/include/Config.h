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

#pragma once

namespace Motion{
    // sampling step over the flow Mat
    const int SAMPLE_STEP_X = 15;
    const int SAMPLE_STEP_Y = 15;

    // scaling of the input image
    const double SCALE_X = 0.5;
    const double SCALE_Y = 0.5;

    // NN normalization
    //
    const double MAX_DIST = 3; // in meters, max distance between two frames
                                // depends on FPS and images sampling for optical flow
    const double MAX_FLOW = 30;
}