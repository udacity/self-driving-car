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

#include <utils/Topics.h>

#include "Motion.h"
#include "MotionLearner.h"
#include "MotionTest.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "motion");
    ros::NodeHandle node("motion");
    //Motion::MotionTest motion();
    Motion::MotionTest motion(argc, argv, node);
    //Motion::MotionLearner ml(argc, argv, node);
    ros::spin();
}