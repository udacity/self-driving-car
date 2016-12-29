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
 * @brief: A set of functions we use for serialization of some of the non-OpenCV types for the purpose of saving maps we use for localization.
 */

#ifndef PROJECT_SERIALIZATIONHELPER_H
#define PROJECT_SERIALIZATIONHELPER_H

#include <cstdio>
#include <opencv2/core/core.hpp>
#include <boost/serialization/split_free.hpp>
#include <tf2/LinearMath/Transform.h>

namespace boost {
    namespace serialization {

        /**
         * Serialization support for std::tuple
         */
        template<uint N>
        struct Serialize
        {
            template<class Archive, typename... Args>
            static void serialize(Archive & ar, std::tuple<Args...> & t, const unsigned int version)
            {
                ar & std::get<N-1>(t);
                Serialize<N-1>::serialize(ar, t, version);
            }
        };

        template<>
        struct Serialize<0>
        {
            template<class Archive, typename... Args>
            static void serialize(Archive & ar, std::tuple<Args...> & t, const unsigned int version)
            {
            }
        };

        template<class Archive, typename... Args>
        void serialize(Archive & ar, std::tuple<Args...> & t, const unsigned int version)
        {
            Serialize<sizeof...(Args)>::serialize(ar, t, version);
        }

    }
}

BOOST_SERIALIZATION_SPLIT_FREE(tf2::Vector3)
namespace boost {
    namespace serialization {

        /**
         * Serialization support for tf2::Vector3
         */
        template<class Archive>
        void save(Archive &ar, const tf2::Vector3 &v, const unsigned int version) {
            ar & v.m_floats[0];
            ar & v.m_floats[1];
            ar & v.m_floats[2];
            ar & v.m_floats[3];
        }

        template<class Archive>
        void load(Archive &ar, tf2::Vector3 &v, const unsigned int version) {
            ar & v.m_floats[0];
            ar & v.m_floats[1];
            ar & v.m_floats[2];
            ar & v.m_floats[3];
        }
    }
}
BOOST_SERIALIZATION_SPLIT_FREE(tf2::Transform)
namespace boost {
    namespace serialization {

        /**
         * Serialization support for tf2::Transform
         */
        template<class Archive>
        void save(Archive &ar, const tf2::Transform &t, const unsigned int version) {

            ar & t.getBasis()[0];
            ar & t.getBasis()[1];
            ar & t.getBasis()[2];
            ar & t.getOrigin();
        }

        template<class Archive>
        void load(Archive &ar, tf2::Transform &t, const unsigned int version) {

            ar & t.getBasis()[0];
            ar & t.getBasis()[1];
            ar & t.getBasis()[2];
            ar & t.getOrigin();
        }
    }
}

#endif //PROJECT_SERIALIZATIONHELPER_H
