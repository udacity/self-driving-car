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

#include <utils/Visualization.h>

void utils::Visualization::DrawMap(const Map::Map &map) {
    if(path_.poses.size() <= map.GetSegmentsSize() && map.GetSegmentsSize() > 0) {
        if(path_.poses.size() > 0) {
            path_.poses.pop_back();
        } else {
            gps_.latitude = map.GetSegment(0).GetBeginning().GetLatitude();
            gps_.longitude = map.GetSegment(0).GetBeginning().GetLongitude();
        }

        for(std::size_t i=path_.poses.size(); i <= map.GetSegmentsSize(); i++) {
            geometry_msgs::PoseStamped msg;
            msg.header.stamp = ros::Time::now();
            msg.header.frame_id = "map";
            cv::Point3d distance;

            if(i==map.GetSegmentsSize()) { // Draw end of last segment!
                distance = map.GetSegment(0).GetBeginning().Distance3DTo(map.GetSegment(i-1).GetBeginning()) + map.GetSegment(i-1).GetDir();
            } else {
                distance = map.GetSegment(0).GetBeginning().Distance3DTo(map.GetSegment(i).GetBeginning());
            }

            tf2::Transform transform;
            tf2::Vector3 T(distance.x * coef, distance.y * coef,distance.z);
            tf2::Quaternion R;
            R.setRPY(0,0, 0);

            transform = tf2::Transform(R,T);
            tf2::toMsg(transform, msg.pose);
            path_.poses.push_back(msg);

            if(i!=map.GetSegmentsSize()) {
                cv::Point2d vec = {distance.y,-distance.x};
                vec =vec / cv::norm(vec);

                for (auto &yellowSign:map.GetSegment(i).GetMapPoints()[0].detectedObjecsts.yellowSigns) {
                    visualization_msgs::Marker sign;
                    sign.header.frame_id = "map";
                    sign.header.stamp = ros::Time::now();
                    sign.ns = "yellowSign";
                    sign.id = markers_.markers.size();
                    sign.type = visualization_msgs::Marker::CUBE;
                    sign.action = visualization_msgs::Marker::ADD;
                    sign.lifetime = ros::Duration();
                    sign.scale.x = 0.4 * coef;
                    sign.scale.y = 0.4 * coef;
                    sign.scale.z = 0.4 * coef;

                    sign.pose.orientation.x = 0;
                    sign.pose.orientation.y = 0;
                    sign.pose.orientation.z = 0;
                    sign.pose.orientation.w = 1.0;

                    auto pose = vec * yellowSign.first.x /300;

                    sign.pose.position.x = (transform.getOrigin().x() + pose.x) * coef;
                    sign.pose.position.y = (transform.getOrigin().y() + pose.y)  * coef;
                    sign.pose.position.z = transform.getOrigin().z();

                    sign.color.r = 255;
                    sign.color.g = 255;
                    sign.color.b = 0;
                    sign.color.a = 1;
                    markers_.markers.push_back(sign);
                }

                for (auto &yellowSign:map.GetSegment(i).GetMapPoints()[0].detectedObjecsts.darkYellowSigns) {
                    visualization_msgs::Marker sign;
                    sign.header.frame_id = "map";
                    sign.header.stamp = ros::Time::now();
                    sign.ns = "yellowSign";
                    sign.id = markers_.markers.size();
                    sign.type = visualization_msgs::Marker::CUBE;
                    sign.action = visualization_msgs::Marker::ADD;
                    sign.lifetime = ros::Duration();
                    sign.scale.x = 0.4 * coef;
                    sign.scale.y = 0.4 * coef;
                    sign.scale.z = 0.4 * coef;

                    sign.pose.orientation.x = 0;
                    sign.pose.orientation.y = 0;
                    sign.pose.orientation.z = 0;
                    sign.pose.orientation.w = 1.0;

                    auto pose = vec * yellowSign.first.x /300;

                    sign.pose.position.x = (transform.getOrigin().x() + pose.x )  * coef;
                    sign.pose.position.y = (transform.getOrigin().y() + pose.y)  * coef;
                    sign.pose.position.z = (transform.getOrigin().z());

                    sign.color.r = 125;
                    sign.color.g = 125;
                    sign.color.b = 0;
                    sign.color.a = 1;
                    markers_.markers.push_back(sign);
                }

                for (auto &yellowSign:map.GetSegment(i).GetMapPoints()[0].detectedObjecsts.darkGreenSigns) {
                    visualization_msgs::Marker sign;
                    sign.header.frame_id = "map";
                    sign.header.stamp = ros::Time::now();
                    sign.ns = "yellowSign";
                    sign.id = markers_.markers.size();
                    sign.type = visualization_msgs::Marker::CUBE;
                    sign.action = visualization_msgs::Marker::ADD;
                    sign.lifetime = ros::Duration();
                    sign.scale.x = 0.4 * coef;
                    sign.scale.y = 0.4 * coef;
                    sign.scale.z = 0.4 * coef;

                    sign.pose.orientation.x = 0;
                    sign.pose.orientation.y = 0;
                    sign.pose.orientation.z = 0;
                    sign.pose.orientation.w = 1.0;

                    auto pose = vec * yellowSign.first.x /300;

                    sign.pose.position.x = (transform.getOrigin().x() + pose.x)  * coef;
                    sign.pose.position.y = (transform.getOrigin().y() + pose.y )  * coef;
                    sign.pose.position.z = (transform.getOrigin().z());

                    sign.color.r = 0;
                    sign.color.g = 255;
                    sign.color.b = 0;
                    sign.color.a = 1;
                    markers_.markers.push_back(sign);
                }
            }
        }
    }

    path_.header.stamp = ros::Time::now();
    path_.header.frame_id = "map";
    DrawMap();
}