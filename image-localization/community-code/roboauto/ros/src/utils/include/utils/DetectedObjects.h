/**
 * @author: RoboAuto Team
 * @brief: A struct that aggregates descriptors we use that were detected in an image.
 */

#pragma once

#include <ImageDescriptor/Horizont.h>
#include "./SignDetection.h"
#include "PoleDetector.h"
#include <opencv2/core.hpp>

namespace utils {
    struct DetectedObjects {
        std::vector<cv::Point> lights;
        bool semaphore = false;
        bool bridge = false;
        ImageDescriptor::Horizont::Description horizont;
        std::vector<PoleDetector::Pole> poles;
        std::vector<SignDetection::Sign> yellowSigns;
        std::vector<SignDetection::Sign> darkYellowSigns;
        std::vector<SignDetection::Sign> darkGreenSigns;

    private:
        // Boost serialization
        friend class boost::serialization::access;

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {
            ar & lights;
            ar & horizont;

            std::vector<float> __fHorizont;
            ar & __fHorizont;

            ar & poles;
            ar & yellowSigns;
            ar & darkYellowSigns;
            ar & darkGreenSigns;
            ar & bridge;
        }
    };
}
