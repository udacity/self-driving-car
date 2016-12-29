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
 * @brief: A class that represents the map we are to localize ourselves in. The map is divided into segments, which are in turn divided
 * into map points.
 */

#pragma once

#include <utils/GpsCoords.h>
#include <ImageDescriptor/ImageDescriptor.h>

#include "DetectedObjects.h"
#include <tuple>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include "CVSerializationHelper.h"
#include "SerializationHelper.h"


namespace utils {
namespace Map {

    class MapPoint {
    public:
        MapPoint() {}

        MapPoint(const DetectedObjects &obj) : detectedObjecsts(obj) {

        }

        DetectedObjects detectedObjecsts;
    private:
        // Boost serialization
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {
            ar & detectedObjecsts;
        }
    };

    class Segment {
    public:
        void SetBeginning(const utils::GpsCoords &beginning) {
            beginning_=beginning;
        }

        utils::GpsCoords& GetBeginning() {
            return beginning_;
        }

        const utils::GpsCoords& GetBeginning() const {
            return beginning_;
        }

        void SetDir(const cv::Point3d &dir) {
            dir_=dir;
            size_ = cv::norm(dir);
        }

        cv::Point3d& GetDir() {
            return dir_;
        }

        const cv::Point3d& GetDir() const {
            return dir_;
        }

        void AddMapPoint(MapPoint && point) {
            points_.push_back(point);
        }

        const std::vector<MapPoint>& GetMapPoints() const {
            return points_;
        }

        std::vector<MapPoint>& GetMapPoints() {
            return points_;
        }

        const std::vector<std::tuple<double,std::size_t,double>>& getNextSegments() const {
            return nextSegments;
        };

        void AddNextSegment(std::tuple<double,std::size_t,double> t) {
            nextSegments.push_back(t);
        }

        float GetSize() const {
            return size_;
        }

    protected:
        // 1 = percentage of segment, size_t is id of segment, 2 is percentage of next
        std::vector<std::tuple<double,std::size_t,double>> nextSegments;
        utils::GpsCoords beginning_;
        cv::Point3d dir_;
        float size_;
        std::vector<MapPoint> points_;
    private:
        // Boost serialization
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {
            ar & nextSegments;
            ar & beginning_;
            ar & dir_;
            ar & points_;
            ar & size_;
        }
    };

    class Map {
    public:
        void AddSegment(const Segment& segment_) {
            map_.push_back(segment_);
        }

        std::size_t GetSegmentsSize() const {
            return map_.size();
        }

        float GetMapDistance() {
            if(mapDistance != 0)
                return mapDistance;

            for(const Segment& segment : map_){
                mapDistance += segment.GetSize();
            }

            return mapDistance;
        }

        Segment& GetSegment(std::size_t index) {
            return map_[index];
        }

        const Segment& GetSegment(std::size_t index) const {
            return map_[index];
        }

        std::vector<std::size_t> GetNearSegments(utils::GpsCoords &coords, float maxDiff) {
            std::vector<std::size_t> nearest;
            for(std::size_t i=0;i<map_.size();i++) {
                const auto& segment = map_[i];
                if(segment.GetBeginning().DistanceTo(coords) < maxDiff) {
                    nearest.push_back(i);
                }
            }
            return nearest;
        }

        double ShiftByMeters(double pos, double dist) const {
            std::size_t segment = static_cast<std::size_t>(floor(pos));
            double segmentPercentage = pos - segment;

            double leftMeters = (1.0-segmentPercentage)*map_[segment].GetSize();

            while(dist > leftMeters && segment < map_.size()-2) {
                dist-=leftMeters;
                segmentPercentage = 0.0;
                segment++;
                leftMeters = map_[segment].GetSize();
            }

            double percentageToShift = segmentPercentage+dist/leftMeters;
            pos = segment + percentageToShift;

            if(pos >= map_.size()) {
                pos = map_.size() -1.1;
            }

            return pos;
        }

        std::pair<double, double> GetSegmentsBounds(double pos, double distance) const {
            std::pair<double, double> ret;
            int segment = pos;
            double percent = pos - segment;

            double sumdist = percent * map_[segment].GetSize();

            int offset;
            for(offset = segment; sumdist < distance && offset >= 0; offset--){
                sumdist += map_[offset].GetSize();
            }

            ret.first = static_cast<double>(offset);

            sumdist = map_[segment].GetSize() - percent * map_[segment].GetSize();

            for(offset = segment; sumdist < distance && static_cast<size_t>(offset) < map_.size(); offset++){
                sumdist += map_[offset].GetSize();
            }

            ret.second = static_cast<double>(offset);

            return ret;
        }

        double GetDistBetweenSegments(double pos1, double pos2) const{
            if(pos1 > pos2){
                std::swap(pos1, pos2);
            }

            if(floor(pos1) - floor(pos2) == 0) {
                return map_[floor(pos1)].GetSize()*(pos2-pos1);
            } else {
                double dist = 0.0;
                int start = floor(pos1)+1;
                dist += map_[start-1].GetSize()*(start - pos1);
                for(int i = start; i < floor(pos2); i++){
                    dist += map_[i].GetSize();
                }
                dist += map_[floor(pos2)].GetSize()*(pos2 - floor(pos2));
                return dist;
            }
        }

    protected:
        std::vector<Segment> map_;

    private:
        float mapDistance = 0;
        // Boost serialization
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {
            ar & map_;
        }
    };
}
};