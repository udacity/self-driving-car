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
 * @brief: A class that represents GPS coordinates along with some useful functions.
 */

#pragma once

#include <opencv2/core.hpp>

#include <gps.h>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/utility.hpp>

#undef STATUS_NO_FIX
#undef STATUS_FIX

namespace utils {

    class GpsCoords{

    public:
        GpsCoords(double lat = 0.0, double lon = 0.0, double altitude = 0.0, double azimuth = 0.0)  : latitude_(lat), longitude_(lon), altitude_(altitude), azimuth_(azimuth)  {
            latLonDegree_ =latLonDegreeLength();
        }

        bool isNull() {
            return latitude_ == 0.0 && longitude_ == 0.0 && azimuth_ == 0.0;
        }

        template <typename T>
        void Shift(const T& vector) {
            latitude_ += vector.x / latLonDegree_.first;
            longitude_ -= vector.y / latLonDegree_.second;
            latLonDegree_ =latLonDegreeLength();
        }

        // this -> coords
        cv::Point2d Distance2DTo(const GpsCoords& coords) const {
            double x = (coords.latitude_ - latitude_) * latLonDegree_.first;
            double y = (coords.longitude_ - longitude_) * latLonDegree_.second;

            return {x,-y};
        };

        // this -> coords
        cv::Point3d Distance3DTo(const GpsCoords& coords) const {
            double x = (coords.latitude_ - latitude_) * latLonDegree_.first;
            double y = (coords.longitude_ - longitude_) * latLonDegree_.second;
            double z = (coords.altitude_ - altitude_);

            return {x,-y,z};
        };

        double DistanceTo(const GpsCoords &c) const
        {
            return earth_distance(GetLatitude(), GetLongitude(),c.GetLatitude(), c.GetLongitude());
        }

        static double Azimuth(GpsCoords first, GpsCoords second);

        double GetLatitude() const {
            return latitude_;
        }

        double GetLongitude() const {
            return longitude_;
        }

        double GetAltitude() const {
            return altitude_;
        }

        double GetAzimuth() const {
            return azimuth_;
        }

        void SetLatitude(double latitude) {
            latitude_ = latitude;
            latLonDegree_ =latLonDegreeLength();
        }

        void SetLongitude(double longitude) {
            longitude_ = longitude;
        }

        void SetAltitude(double altitude) {
            altitude_ = altitude;
        }

        void SetAzimuth(double azimuth)
        {
            azimuth_ = azimuth;
        }

        bool operator!= (const GpsCoords &other) const
        {
            return !(latitude_ == other.latitude_ && longitude_ == other.longitude_ && azimuth_ == other.azimuth_);
        }

    private:
        double latitude_ = 0.0;
        double longitude_ = 0.0;
        double altitude_ = 0.0;
        double azimuth_ =0.0;
        std::pair<double,double> latLonDegree_;

        static constexpr double EARTH_EQUATORIAL_RADIUS = 6378137.0; //m
        static constexpr double EARTH_ECCENTRICITY_SQUARED = 0.00669437999014;
        //<lat_length, long_length> // <y,x>
        std::pair<double,double> latLonDegreeLength()
        {
            double lat_radians = latitude_ * M_PI /180;

            double lat_length = (M_PI * EARTH_EQUATORIAL_RADIUS * (1.0 - EARTH_ECCENTRICITY_SQUARED)) / (180.0 * pow(1 - (EARTH_ECCENTRICITY_SQUARED * pow(sin(lat_radians), 2)), 1.5));
            double long_length = (M_PI * EARTH_EQUATORIAL_RADIUS * cos(lat_radians)) /  (180.0 * pow(1 - (EARTH_ECCENTRICITY_SQUARED * pow(sin(lat_radians), 2)), 0.5));

            return std::pair<double,double>(lat_length, long_length);
        }

        // Boost serialization
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {
            ar & latitude_;
            ar & longitude_;
            ar & altitude_;
            ar & azimuth_;
            ar & latLonDegree_;
        }
    };
}
