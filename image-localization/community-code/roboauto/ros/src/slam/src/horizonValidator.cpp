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
 * @author: RoboAuto team
 * @brief: A statistical tool for evaluation of the horizons comparison.
 *
 * This tool shows the score statistics of all horizons in the current map compared to the current horizon (from running bag).
 * The tool also evaluates the current frame and how often the horizons from similar map segment (close horizons) score in the top 10 most similar horizons.
 */

#include "SlamImpl.h"

#include <utils/Topics.h>
#include <sensor_msgs/NavSatFix.h>

#include <std_msgs/String.h>

class DataSetCreator : public Slam::SlamImpl {

public:

    DataSetCreator(ros::NodeHandle &node) {

        subCamImage_ = node.subscribe(utils::Topics::CAMERA, 1, &DataSetCreator::OnCameraImage, this);
        subGPS_ = node.subscribe(utils::Topics::GPS, 1, &DataSetCreator::OnGPS, this);
        subCommand_ = node.subscribe("/command", 1, &DataSetCreator::OnCommand, this);

        DSCreatorLogFile = std::ofstream("horizons.txt", std::ios::out | std::ios::trunc);

        std::fill(DSCreatorPositionHits.begin(), DSCreatorPositionHits.end(), 0);
    }



    void DrawHor(const ImageDescriptor::Horizont::Description &d, std::string name) {
        cv::Mat out ({640,480}, CV_8UC3,{0,0,0});
        for(std::size_t i=0;i<d.horizont.size();i++) {
            cv::line(out,{static_cast<int>(i),0},{static_cast<int>(i),480-d.horizont[i]},cv::Scalar(255,255,255),1);
        }
        cv::imshow(name,out);
    }



    void OnCameraImage(const sensor_msgs::ImageConstPtr& msg) {
        // do not evaluate before first GPS update
        if(!DSCreatorGpsUpdated)
            return;

        cv::Mat img;
        utils::Camera::ConvertImage(msg, img);

        auto horizont = descriptor_.DescribeImage(img);
        std::vector<std::pair<std::size_t, float>> bestSegments;
        std::vector<float> score(map_.GetSegmentsSize());
        std::vector<int> scoreIndex(map_.GetSegmentsSize());

        for (std::size_t i = 0; i < map_.GetSegmentsSize(); i++) {
            scoreIndex[i] = i;
            score[i] = descriptor_.Compare(horizont, map_.GetSegment(i).GetMapPoints()[0].detectedObjecsts.horizont);
        }

        std::sort(scoreIndex.begin(), scoreIndex.end(),
                  [score](std::size_t l, std::size_t r) { return score[l] < score[r]; });

        std::stringstream log;


        /// GPS HORIZONT
        log << "GPS_segment: " << DSCreatorNearestIndex << std::endl;
        log << "Vals: ";
        for (float val : map_.GetSegment(DSCreatorNearestIndex).GetMapPoints()[0].detectedObjecsts.horizont.horizont) {
            log << val << ",";
        }
        log << std::endl;

        /// CURRENT HORIZONT
        log << "Current_horizont: " << std::endl;
        log << "Vals: ";
        for (float val : horizont.horizont) {
            log << val << ",";
        }
        log << std::endl;

        /// 10 BEST HORIZONTS
        bool hit = false;
        for(std::size_t i=0;i<10;i++) {

            double score = descriptor_.Compare(horizont,map_.GetSegment(scoreIndex[i]).GetMapPoints()[0].detectedObjecsts.horizont);

            log << (i + 1) << ". best  segment: " << scoreIndex[i] << "  score: " << score <<  "\n";
            log << "Vals: ";

            for (float val : map_.GetSegment(scoreIndex[i]).GetMapPoints()[0].detectedObjecsts.horizont.horizont){
                log << val << ",";
            }
            log << std::endl;

            // the horizon is in the range of GPS
            if(scoreIndex[i] > DSCreatorNearestBoundsIndex.first && scoreIndex[i] < DSCreatorNearestBoundsIndex.second){
                DSCreatorPositionHits[i]++;
                hit = true;
            }
        }

        if(hit)
            DSCreatorTopHits++;
        DSCreatorFrameCnt++;

        int topPosition = std::find_if(scoreIndex.begin(), scoreIndex.end(),
                                        [&](int seg)
                                        { return (seg > DSCreatorNearestBoundsIndex.first
                                          && seg < DSCreatorNearestBoundsIndex.second); }) - scoreIndex.begin() + 1;
        DSCreatorTopPositionSum += topPosition;

        log << "99. best  segment: " << topPosition << " score: 0" << std::endl;
        log << "Vals: ";
        for (float val : map_.GetSegment(scoreIndex[topPosition]).GetMapPoints()[0].detectedObjecsts.horizont.horizont){
            log << val << ",";
        }
        log << std::endl;
        log << "\n\n=========================================\n\n";

        printStatistics(topPosition);

        if(!hit && (DSCreatorFrameCnt-DSCreatorTopHits) < DSCreatorMaxFramesToLog)
            DSCreatorLogFile << log.str();
    }


    void printStatistics(int topPosition){
        std::cout << "==== Statistics  =======================\n";
        std::cout << "Top 10 hits: " << DSCreatorTopHits << "/" << DSCreatorFrameCnt << "  => "
                  << ((static_cast<double>(DSCreatorTopHits)/DSCreatorFrameCnt)*100) << "%\n";
        std::cout << "Top position: " << topPosition << " Avg. top position: " << (static_cast<double>(DSCreatorTopPositionSum)/DSCreatorFrameCnt) << std::endl;
        std::cout << "\nHistogram: \n";

        int sum = 0;
        for(std::size_t i=0; i<DSCreatorPositionHits.size(); i++){
            sum += DSCreatorPositionHits[i];
            std::cout << (i+1) << ". : " << DSCreatorPositionHits[i] << std::endl;
        }
        std::cout << "Sum of all top 10: " << sum << std::endl;

        std::cout << std::endl;
    }

    void OnGPS(const sensor_msgs::NavSatFix& msg) {
        DSCreatorGpsUpdated = true;

        utils::GpsCoords actCoords(msg.latitude, msg.longitude, msg.altitude);
        double nearestPos = map_.GetSegment(0).GetBeginning().DistanceTo(actCoords);
        DSCreatorNearestIndex = 0;

        for(std::size_t i = 1; i < map_.GetSegmentsSize();i++) {
            double dist = map_.GetSegment(i).GetBeginning().DistanceTo(actCoords);
            if(dist < nearestPos) {
                nearestPos=dist;
                DSCreatorNearestIndex = i;
                DSCreatorNearestBoundsIndex = map_.GetSegmentsBounds(i, DSCreatorMaxRange);
            }
        }
    }

    void OnCommand(const std_msgs::String& msg) {
        if(msg.data[0] == 'l') {
            map_ = utils::Map::Map();
            LoadMap(std::string({msg.data.begin()+2, msg.data.end()}));
            std::cout << "Loaded map from " <<  std::string({msg.data.begin()+2, msg.data.end()}) << std::endl;
        }
    }

protected:
    std::shared_ptr<sensor_msgs::NavSatFix> gps_;
    std::map<int, std::tuple<CurrentState, utils::GpsCoords>> runSamples_;
    std::map<int, std::tuple<CurrentState, double>> batchSamples_;

    Motion::Motion motion_;

    ros::Subscriber subCamImage_;
    ros::Subscriber subGPS_;
    ros::Subscriber subCommand_;

    int DSCreatorNearestIndex = 0;
    int DSCreatorMaxRange = 100;  // m
    std::pair<int, int> DSCreatorNearestBoundsIndex;
    std::array<int, 10> DSCreatorPositionHits;
    int DSCreatorTopHits = 0;
    int DSCreatorFrameCnt = 0;
    bool DSCreatorGpsUpdated = false;
    int DSCreatorMaxFramesToLog = 10;
    int DSCreatorTopPositionSum = 0;
    std::ofstream DSCreatorLogFile;
};



int main(int argc, char** argv)
{
    ros::init(argc, argv, "slam");
    ros::NodeHandle node("slam");
    DataSetCreator slam(node);
    ros::spin();
}
