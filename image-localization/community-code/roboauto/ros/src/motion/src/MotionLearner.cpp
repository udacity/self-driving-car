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

#include "MotionLearner.h"

void Motion::MotionLearner::OnCameraUpdateDataset(const sensor_msgs::ImageConstPtr& msg){
    if((frameCnt++)%2 == 0)
        return;

    cv::Mat orig;
    utils::Camera::ConvertImage(msg, orig);

    resize(orig, orig, cv::Size(orig.cols*SCALE_X, orig.rows*SCALE_Y));

    images_.push_back(orig);
    imgStamps_.push_back(msg->header.stamp.toSec());

    cv::imshow("orig", orig);
    cv::waitKey(1);

    if(images_.size() <= 1) {
        return;
    }

    cv::Mat flow = utils::Motion::getOptFlow(images_[images_.size()-2], images_[images_.size()-1], SAMPLE_STEP_X, SAMPLE_STEP_Y);

    cv::imshow("flow", utils::Motion::DrawOpticalFlow(images_[images_.size()-2],flow));

//    std::cout << flow.at<cv::Point2f>(0,0) << std::endl;
//    std::cout << flow.at<cv::Point2f>(5,0) << std::endl;
//    std::cout << flow.at<cv::Point2f>(10,10) << std::endl;
//    std::cout << flow.at<cv::Point2f>(100,100) << std::endl;
//    std::cout << flow.at<cv::Point2f>(50,50) << std::endl;

    flows_.push_back(flow);

    double dt = (imgStamps_[imgStamps_.size()-1]) - (imgStamps_[imgStamps_.size()-2]);
    double dist = speeds_[speeds_.size()-1] * dt;

    // time jump
    if(dt < 0)
        return;

    addSample(flow, dist);

    // delete first N elements
    if(images_.size() > 1000){
        std::vector<cv::Mat>(images_.begin()+900, images_.end()).swap(images_);
    }
};

void Motion::MotionLearner::OnCameraUpdatePredict(const sensor_msgs::ImageConstPtr& msg) {
    if((frameCnt++)%2 == 0)
        return;

    cv::Mat orig;
    utils::Camera::ConvertImage(msg, orig);

    resize(orig, orig, cv::Size(orig.cols*SCALE_X, orig.rows*SCALE_Y));

    images_.push_back(orig);

    if(images_.size() <= 1)
        return;


    cv::Mat flow = utils::Motion::getOptFlow(images_[images_.size()-2], images_[images_.size()-1], SAMPLE_STEP_X, SAMPLE_STEP_Y);

    float dist = predictDistance(flow);

    std::cout << "Predicted dist: " << dist << std::endl;
}

void Motion::MotionLearner::OnGpsUpdate(const sensor_msgs::NavSatFixPtr& msg){

};

void Motion::MotionLearner::OnGpsSpeedUpdate(const geometry_msgs::TwistStampedPtr& msg){
    double vx = msg->twist.linear.x;
    double vy = msg->twist.linear.y;
//    std::cout << "speed: " << sqrt(vx*vx + vy*vy) << std::endl;
//    std::cout << "stamp: " << msg->header.stamp.toSec() << std::endl;
//    std::cout << "stamp2: " << speedStamps_[speedStamps_.size()-1] << std::endl;

    double speed = sqrt(vx*vx + vy*vy);

    // second update? (warning: speeds_ have one more value than the speedStamps_)
    if(speeds_.size() == 2){

        // TODO: possible problems with sample values
        samples_.clear();


//        double dts = msg->header.stamp.toSec() - speedStamps_[speedStamps_.size()-1];
//        for(Sample::Sample &sample : samples_){
            // update the speed values of the samples without velocity fix
            //(sample.second)[0] = speed * dts;
//        }
    }

    speeds_.push_back(speed);
    speedStamps_.push_back(msg->header.stamp.toSec());
};

void Motion::MotionLearner::addSample(cv::Mat flow, double speed){
    Sample::Sample sample = Sample::createSample(flow, speed);
    samples_.push_back(sample);

    std::cout << "Adding sample!\n";
    std::cout << "Samples total: " << samples_.size() << std::endl;
    std::cout << "In size: " << sample.first.size() << std::endl;
    std::cout << "In prev: " << sample.first[0] << "," << sample.first[1] << "," << sample.first[2] << std::endl;
    std::cout << "Dist prev: " << sample.second[0] << std::endl << std::endl;
};

void Motion::MotionLearner::CommandReceived(const std_msgs::StringPtr& msg){
    std::cout << "Command received! \"" << msg->data << "\"\n";

    // learn
    if(msg->data[0] == 'l') {
        if(msg->data.size() > 3){
            std::string num = msg->data.substr(2);
            nn_learning_iterations_ = std::stoi(num);
        }

        std::cout << "Number of learning iterations set to: " << nn_learning_iterations_ << std::endl;

        cameraSub_.shutdown();
        gpsSub_.shutdown();
        gpsSpeedSub_.shutdown();

        learn(net_);

        cameraSub_ = n_.subscribe(utils::Topics::CAMERA, 1, &MotionLearner::OnCameraUpdateDataset, this);
        gpsSub_ = n_.subscribe(utils::Topics::GPS, 1, &MotionLearner::OnGpsUpdate, this);
        gpsSpeedSub_ = n_.subscribe(utils::Topics::GPS_SPEED, 1, &MotionLearner::OnGpsSpeedUpdate, this);
    } // preview distances
    else if(msg->data[0] == 'p'){
        cameraSub_.shutdown();
        gpsSub_.shutdown();
        gpsSpeedSub_.shutdown();

        cameraSub_ = n_.subscribe(utils::Topics::CAMERA, 100, &MotionLearner::OnCameraUpdatePredict, this);

        //learning = false;
        //cameraSub_ = n_.subscribe(utils::Topics::CAMERA, 1, &MotionLearner::OnCameraUpdate, this);

    // save
    }else if(msg->data[0] == 's'){
        saveNetToFile(saveFile);
    // validate
    }else if(msg->data[0] == 'v'){
        cameraSub_.shutdown();
        gpsSub_.shutdown();
        gpsSpeedSub_.shutdown();

        std::cout << "Validating: ...\n";
        validate(net_, samples_);

        cameraSub_ = n_.subscribe(utils::Topics::CAMERA, 1, &MotionLearner::OnCameraUpdateDataset, this);
        gpsSub_ = n_.subscribe(utils::Topics::GPS, 1, &MotionLearner::OnGpsUpdate, this);
        gpsSpeedSub_ = n_.subscribe(utils::Topics::GPS_SPEED, 1, &MotionLearner::OnGpsSpeedUpdate, this);
    }
};

// weights are randomized during initialization of a new NN
void Motion::MotionLearner::learn(NeuralNetwork::FeedForward::Network& n){
    if(samples_.size() < 10){
        std::cout << "There is too few learning samples, skipping learning...\n";
    }

    std::cout << "Preparing BP...\n";

    NeuralNetwork::Learning::BackPropagation prop(n);

    std::cout << "Learning NN...\n";
    learning = true;

    std::random_shuffle(samples_.begin(), samples_.end());

    int delim = samples_.size()/3;
    std::vector<Sample::Sample> train(samples_.begin(), samples_.begin()+(2*delim));
    std::vector<Sample::Sample> valid(samples_.begin()+(2*delim), samples_.end());

    std::cout << "Training DS size: " << train.size() << std::endl;
    std::cout << "Validation DS size: " << valid.size() << std::endl;

    for(int i=0;i<=nn_learning_iterations_ && learning;i++) {
        std::cout << "Learning iteration nr.: " << i << std::endl;
        for(auto sample : samples_){
            prop.teach(sample.first, sample.second);
        }

        if(i % 10 == 0)
            validate(n, train);
    }

    std::cout << "NN learning done...\n";
    validate(n, valid);
};


void Motion::MotionLearner::validate(NeuralNetwork::FeedForward::Network &n, std::vector<Sample::Sample> &samples){
    float sum = 0;
    float sqrErr = 0;
    for(auto sample : samples) {
        sum += fabs(sample.second[0] - (n.computeOutput(sample.first))[0]);
        sqrErr += pow(fabs(sample.second[0] - (n.computeOutput(sample.first))[0]), 2);
//        std::cout << "NN out: " << (n.computeOutput(sample.first))[0] << std::endl;
//        std::cout << "out: " << (n.computeOutput(sample.second))[0] << std::endl;
    }

    std::cout << "Mean Error: " << (sum/samples.size()) << std::endl;
    std::cout << "Mean SqrError: " << (sqrErr/samples.size()) << std::endl;
    std::cout << "Mean Error in meters: " << (Motion::MAX_DIST * sum/samples.size()) << std::endl;
}

void Motion::MotionLearner::initNN(NeuralNetwork::FeedForward::Network& n,const std::string& file) {
    // we have a file
    if (!file.empty()) {
        loadNetFromFile(file);
    } else {
        NeuralNetwork::ActivationFunction::Sigmoid a(-1);
        n.appendLayer(100, a);
        //n.appendLayer(100, a);
        n.appendLayer(1, a);
        n.randomizeWeights();
    }
};

void Motion::MotionLearner::saveNetToFile(const std::string& file){
    std::cout << "Saving NN into the file: \"" << file << "\"\n";

    std::ofstream myfile(file);
    if (myfile.is_open()) {
        myfile << net_.serialize().serialize();
        myfile.close();
        std::cout << "done...\n";
    } else std::cout << "Unable to open file\n\n";
};

void Motion::MotionLearner::loadNetFromFile(const std::string &file){
    std::cout << "Loading NN from the file: \"" << file << "\"\n";

    std::ifstream myfile(file, std::ifstream::in);
    if (myfile.is_open()) {
        std::string json((std::istreambuf_iterator<char>(myfile)),
                         std::istreambuf_iterator<char>());

        auto nPtr = NeuralNetwork::FeedForward::Network::deserialize(SimpleJSON::JSONParser::parseObject(json)).release();
        net_ = *nPtr;
        std::cout << "done...\n";
    } else std::cout << "Unable to open file";
};

double Motion::MotionLearner::predictDistance(cv::Mat flow){
    return net_.computeOutput(Sample::createSample(flow))[0] * Motion::MAX_DIST;
};