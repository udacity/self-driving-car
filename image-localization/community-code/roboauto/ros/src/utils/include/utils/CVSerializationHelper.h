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
 * @brief: A set of functions we use for serialization of some of the OpenCV types for the purpose of saving maps we use for localization.
 */

#pragma once

#include <opencv2/core/core.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>

BOOST_SERIALIZATION_SPLIT_FREE(cv::Mat)
namespace boost {
    namespace serialization {

        /**
         * Serialization support for cv::Mat
         */
        template<class Archive>
        void save(Archive &ar, const cv::Mat &m, const unsigned int version) {
            size_t elem_size = m.elemSize();
            size_t elem_type = m.type();

            ar & m.cols;
            ar & m.rows;
            ar & elem_size;
            ar & elem_type;

            const size_t data_size = m.cols * m.rows * elem_size;
            ar & boost::serialization::make_array(m.ptr(), data_size);
        }

        template<class Archive>
        void load(Archive &ar, cv::Mat &m, const unsigned int version) {
            int cols, rows;
            size_t elem_size, elem_type;

            ar & cols;
            ar & rows;
            ar & elem_size;
            ar & elem_type;

            m.create(rows, cols, elem_type);

            size_t data_size = m.cols * m.rows * elem_size;
            ar & boost::serialization::make_array(m.ptr(), data_size);
        }
    }
}

BOOST_SERIALIZATION_SPLIT_FREE(cv::KeyPoint)
namespace boost {
    namespace serialization {
        /**
         * Serialization support for cv::KeyPoint
         */
        template<class Archive>
        void save(Archive &ar, const cv::KeyPoint &k, const unsigned int version) {
            ar & k.size;
            ar & k.angle;
            ar & k.response;
            ar & k.octave;
            ar & k.class_id;
            ar & k.pt.x;
            ar & k.pt.y;
        }

        template<class Archive>
        void load(Archive &ar, cv::KeyPoint &k, const unsigned int version) {
            ar & k.size;
            ar & k.angle;
            ar & k.response;
            ar & k.octave;
            ar & k.class_id;
            ar & k.pt.x;
            ar & k.pt.y;
        }
    }
}

BOOST_SERIALIZATION_SPLIT_FREE(cv::Point3d)
namespace boost {
    namespace serialization {

        /**
         * Serialization support for cv::Point3d
         */
        template<class Archive>
        void save(Archive &ar, const cv::Point3d &p, const unsigned int version) {
            ar & p.x;
            ar & p.y;
            ar & p.z;
        }

        template<class Archive>
        void load(Archive &ar, cv::Point3d &p, const unsigned int version) {
            ar & p.x;
            ar & p.y;
            ar & p.z;
        }
    }
}

BOOST_SERIALIZATION_SPLIT_FREE(cv::Point)
namespace boost {
    namespace serialization {

        /**
         * Serialization support for cv::Point
         */
        template<class Archive>
        void save(Archive &ar, const cv::Point &p, const unsigned int version) {
            ar & p.x;
            ar & p.y;
        }

        template<class Archive>
        void load(Archive &ar, cv::Point &p, const unsigned int version) {
            ar & p.x;
            ar & p.y;
        }
    }
}

BOOST_SERIALIZATION_SPLIT_FREE(cv::Point2f)
namespace boost {
    namespace serialization {

        /**
         * Serialization support for cv::Point2f
         */
        template<class Archive>
        void save(Archive &ar, const cv::Point2f &p, const unsigned int version) {
            ar & p.x;
            ar & p.y;
        }

        template<class Archive>
        void load(Archive &ar, cv::Point2f &p, const unsigned int version) {
            ar & p.x;
            ar & p.y;
        }
    }
}

BOOST_SERIALIZATION_SPLIT_FREE(cv::Size2f)
namespace boost {
    namespace serialization {

        /**
         * Serialization support for cv::Size2f
         */
        template<class Archive>
        void save(Archive &ar, const cv::Size2f &s, const unsigned int version) {
            ar & s.width;
            ar & s.height;
        }

        template<class Archive>
        void load(Archive &ar, cv::Size2f &s, const unsigned int version) {
            ar & s.width;
            ar & s.height;
        }
    }
}

BOOST_SERIALIZATION_SPLIT_FREE(cv::RotatedRect)
namespace boost {
    namespace serialization {

        /**
         * Serialization support for cv::RotatedRect
         */
        template<class Archive>
        void save(Archive &ar, const cv::RotatedRect &r, const unsigned int version) {
            ar & r.center;
            ar & r.size;
            ar & r.angle;
        }

        template<class Archive>
        void load(Archive &ar, cv::RotatedRect &r, const unsigned int version) {
            ar & r.center;
            ar & r.size;
            ar & r.angle;
        }
    }
}

