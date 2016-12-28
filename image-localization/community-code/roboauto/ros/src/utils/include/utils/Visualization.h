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
 * @brief: A class that contains ROS publishers for various visualizations along with the methods to draw those.
 */

#pragma once

#include "GpsCoords.h"
#include "Map.h"
#include "ParticleFilter.h"

#include <ros/time.h>
#include <ros/ros.h>
#include <string>
#include <map>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/NavSatFix.h>

namespace utils {
    class Visualization {
    public:
        sensor_msgs::NavSatFix gps_;
        double coef = 1.0;
        Visualization(ros::NodeHandle& rosNode, const std::string& topic) {
            pathPub_ = rosNode.advertise<nav_msgs::Path>(topic+"/map", 1);
            gpsStartPub_ = rosNode.advertise<sensor_msgs::NavSatFix>(topic+"/pos", 1);
            markerPub_ = rosNode.advertise<visualization_msgs::MarkerArray>(topic+"/markers", 1);
            particlesPub_ = rosNode.advertise<visualization_msgs::MarkerArray>(topic+"/particles", 1);
            arrowPub_ = rosNode.advertise<visualization_msgs::Marker>(topic + "/arrow", 1);
            positionPub_ = rosNode.advertise<visualization_msgs::Marker>(topic + "/position", 1);
        }

        void DrawMap(const Map::Map& map);

        void DrawMap() {
            sensor_msgs::NavSatFix gps = gps_;
            gps.header.stamp = ros::Time::now();
            gpsStartPub_.publish(gps);
            pathPub_.publish(path_);
            markerPub_.publish(markers_);
        }

        void DrawPath(const std::vector<std::pair<utils::GpsCoords,tf2::Transform>> transforms) {

            nav_msgs::Path path;
            path.header.stamp = ros::Time::now();
            path.header.frame_id = "map";

            for (size_t i = 0; i < transforms.size(); ++i)
            {
                geometry_msgs::PoseStamped msg;
                msg.header.stamp = ros::Time::now();
                msg.header.frame_id = "car_origin";
                tf2::toMsg(transforms[i].second, msg.pose);
                path.poses.push_back(msg);
            }

        }

        void DrawArrow(const tf2::Transform &t)
        {
            visualization_msgs::Marker msg;
            msg.header.frame_id = "map";
            msg.header.stamp = ros::Time::now();
            msg.ns = "arrow";
            msg.id = 0;
            msg.type = visualization_msgs::Marker::ARROW;
            msg.action = visualization_msgs::Marker::ADD;
            msg.lifetime = ros::Duration();
            msg.scale.x = 3.0;
            msg.scale.y = 0.2;
            msg.scale.z = 0.2;

            double roll,pitch,yaw;
            t.getBasis().getRPY(roll,pitch,yaw);

            msg.pose.orientation.x = roll;
            msg.pose.orientation.y = pitch;
            msg.pose.orientation.z = yaw;
            msg.pose.orientation.w = 1.0;

            msg.pose.position.x = t.getOrigin().x();
            msg.pose.position.y = t.getOrigin().y();
            msg.pose.position.z = t.getOrigin().z();

            msg.color.r = 255;
            msg.color.g = 0;
            msg.color.b = 255;
            msg.color.a = 1;

            arrowPub_.publish(msg);
        }

        void DrawParticles(const Map::Map& map, const utils::ParticleFilter::ParticleFilter<1,0>& filter) {

            visualization_msgs::MarkerArray msg;

            for (size_t i = 0; i < filter.Particles().size(); ++i) {
                visualization_msgs::Marker particleMarker;
                particleMarker.header.frame_id = "map";
                particleMarker.header.stamp = ros::Time::now();
                particleMarker.ns = "particle";
                particleMarker.id = i;
                particleMarker.type = visualization_msgs::Marker::ARROW;
                particleMarker.action = visualization_msgs::Marker::ADD;
                particleMarker.lifetime = ros::Duration(2.0);

                const auto& particle = filter.Particles()[i];
                auto coords = map.GetSegment(floor(particle[0])).GetBeginning();
                auto vec = map.GetSegment(floor(particle[0])).GetDir();

                auto position = map.GetSegment(0).GetBeginning().Distance2DTo(coords);

                double yaw = vec.x==0? 0 : M_PI+atan(vec.y/vec.x);

                particleMarker.pose.position.x = (position.x+vec.x*(particle[0]-floor(particle[0]))) * coef;
                particleMarker.pose.position.y = (position.y+vec.y*(particle[0]-floor(particle[0]))) * coef;
                particleMarker.pose.position.z = 0.5 * coef;

                tf2::Quaternion R;
                R.setRPY(0,0, yaw);

                particleMarker.pose.orientation.x = R.x();
                particleMarker.pose.orientation.y = R.y();
                particleMarker.pose.orientation.z = R.z();
                particleMarker.pose.orientation.w = R.w();

                particleMarker.scale.x = 2.0 ;
                particleMarker.scale.y = 0.5 ;
                particleMarker.scale.z = 0.5 ;
                particleMarker.color.a = 1.0; // Don't forget to set the alpha!
                particleMarker.color.r = 1.0;
                particleMarker.color.g = 0.0;
                particleMarker.color.b = 0.0;
                msg.markers.push_back(particleMarker);
            }

            particlesPub_.publish(msg);
        }

        void DrawClusters(const Map::Map& map, const std::vector<std::pair<double,std::size_t>> &clusters) {

            visualization_msgs::MarkerArray msg;

            for (size_t i = 0; i < clusters.size(); ++i) {
                visualization_msgs::Marker particleMarker;
                particleMarker.header.frame_id = "map";
                particleMarker.header.stamp = ros::Time::now();
                particleMarker.ns = "clusters";
                particleMarker.id = i;
                particleMarker.type = visualization_msgs::Marker::CUBE;
                particleMarker.action = visualization_msgs::Marker::ADD;
                particleMarker.lifetime = ros::Duration(2.0);

                const auto& segmentPosition = clusters[i].first;
                auto coords = map.GetSegment(floor(segmentPosition)).GetBeginning();
                auto vec = map.GetSegment(floor(segmentPosition)).GetDir();

                auto position = map.GetSegment(0).GetBeginning().Distance2DTo(coords);

                double yaw = vec.x==0? 0 : M_PI+atan(vec.y/vec.x);

                double zPos = clusters[i].second/3;
                particleMarker.pose.position.x = (position.x+vec.x*(segmentPosition-floor(segmentPosition))) * coef;
                particleMarker.pose.position.y = (position.y+vec.y*(segmentPosition-floor(segmentPosition))) * coef;
                particleMarker.pose.position.z = (-zPos/2) * coef;

                tf2::Quaternion R;
                R.setRPY(0,0, yaw);

                particleMarker.pose.orientation.x = R.x();
                particleMarker.pose.orientation.y = R.y();
                particleMarker.pose.orientation.z = R.z();
                particleMarker.pose.orientation.w = R.w();

                particleMarker.scale.x = 2 * coef;
                particleMarker.scale.y = 2 * coef;
                particleMarker.scale.z = zPos * coef;

                particleMarker.color.a = 1.0; // Don't forget to set the alpha!
                particleMarker.color.r = 0.0;
                particleMarker.color.g = 0.0;
                particleMarker.color.b = 1.0;
                msg.markers.push_back(particleMarker);
            }

            particlesPub_.publish(msg);
        }

        void DrawPosition(const Map::Map& map, const utils::GpsCoords&coords)
        {
            visualization_msgs::Marker msg;
            msg.header.frame_id = "map";
            msg.header.stamp = ros::Time::now();
            msg.ns = "Position";
            msg.id = 0;
            msg.type = visualization_msgs::Marker::ARROW;
            msg.action = visualization_msgs::Marker::ADD;
            msg.lifetime = ros::Duration();
            msg.scale.x = 200.0 * coef;
            msg.scale.y = 200.5 * coef;
            msg.scale.z = 200.5 * coef;

            auto position = map.GetSegment(0).GetBeginning().Distance2DTo(coords);

            msg.points.resize(2);

            msg.points[0].x = position.x * coef;
            msg.points[0].y = position.y * coef;
            msg.points[0].z = 500 * coef;

            msg.points[1].x = position.x * coef;
            msg.points[1].y = position.y * coef;
            msg.points[1].z = 0;


            msg.color.a = 1.0; // Don't forget to set the alpha!
            msg.color.r = 0.0;
            msg.color.g = 0.0;
            msg.color.b = 1.0;

            positionPub_.publish(msg);
        }

        void DrawAugmentedBorders(const Map::Map& map, double from, double to)
        {
            visualization_msgs::Marker msg;
            msg.header.frame_id = "map";
            msg.header.stamp = ros::Time::now();
            msg.ns = "AGBorder";
            msg.id = 0;
            msg.type = visualization_msgs::Marker::ARROW;
            msg.action = visualization_msgs::Marker::ADD;
            msg.lifetime = ros::Duration();
            msg.scale.x = 20.0 * coef;
            msg.scale.y = 20.5 * coef;
            msg.scale.z = 20.5 * coef;

            to = std::min(map.GetSegmentsSize()-1., to);
            auto frompos = map.GetSegment(0).GetBeginning().Distance2DTo(map.GetSegment(from).GetBeginning());
            auto topos   = map.GetSegment(0).GetBeginning().Distance2DTo(map.GetSegment(to).GetBeginning());

            msg.points.resize(2);

            msg.points[0].x = frompos.x * coef;
            msg.points[0].y = frompos.y * coef;
            msg.points[0].z = 500 * coef;

            msg.points[1].x = frompos.x * coef;
            msg.points[1].y = frompos.y * coef;
            msg.points[1].z = 0;


            msg.color.a = 1.0; // Don't forget to set the alpha!
            msg.color.r = 1.0;
            msg.color.g = 0.0;
            msg.color.b = 1.0;

            positionPub_.publish(msg);

            msg.id = 1;
            msg.points[0].x = topos.x *coef;
            msg.points[0].y = topos.y *coef;
            msg.points[1].x = topos.x *coef;
            msg.points[1].y = topos.y *coef;

            positionPub_.publish(msg);
        }

    protected:

        ros::Publisher pathPub_;
        ros::Publisher particlesPub_;
        ros::Publisher arrowPub_;
        ros::Publisher positionPub_;
        ros::Publisher markerPub_;
        ros::Publisher gpsStartPub_;

        const std::string topic_;

        nav_msgs::Path path_;
        visualization_msgs::MarkerArray markers_;
    };
}