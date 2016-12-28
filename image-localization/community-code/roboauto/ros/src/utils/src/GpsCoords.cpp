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

#include <utils/GpsCoords.h>

namespace utils
{

    double GpsCoords::Azimuth(GpsCoords first, GpsCoords second)
    {
        double result = 0.0;

        double lat1 = first.GetLatitude();
        double lon1 = first.GetLongitude();
        double lat2 = second.GetLatitude();
        double lon2 = second.GetLongitude();

        int ilat1 = (int)(0.50 + lat1 * 360000.0);
        int ilat2 = (int)(0.50 + lat2 * 360000.0);
        int ilon1 = (int)(0.50 + lon1 * 360000.0);
        int ilon2 = (int)(0.50 + lon2 * 360000.0);

        lat1 *= DEG_2_RAD;
        lon1 *= DEG_2_RAD;
        lat2 *= DEG_2_RAD;
        lon2 *= DEG_2_RAD;

        if ((ilat1 == ilat2) && (ilon1 == ilon2))
        {
            return result;
        }
        else if (ilon1 == ilon2)
        {
            if (ilat1 > ilat2)
                result = 180.0;
        }
        else
        {
            double c = acos(sin(lat2)*sin(lat1) +
                            cos(lat2)*cos(lat1)*cos((lon2-lon1)));
            double A = asin(cos(lat2)*sin((lon2-lon1))/sin(c));
            result = (A * RAD_2_DEG);

            if ((ilat2 > ilat1) && (ilon2 > ilon1))
            {
            }
            else if ((ilat2 < ilat1) && (ilon2 < ilon1))
            {
                result = 180.0 - result;
            }
            else if ((ilat2 < ilat1) && (ilon2 > ilon1))
            {
                result = 180.0 - result;
            }
            else if ((ilat2 > ilat1) && (ilon2 < ilon1))
            {
                result += 360.0;
            }
        }

        return result;
    }

}

