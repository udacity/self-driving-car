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

#include "../include/ImageDescriptor/Horizont.h"

#include <opencv2/opencv.hpp>

ImageDescriptor::Horizont::Config ImageDescriptor::Horizont::config_;

std::vector<uint16_t> ImageDescriptor::Horizont::detectMaxPeak(const std::vector<uint16_t> &d) {
    std::size_t steps = 8;
    std::vector<uint16_t> max(steps);
    std::size_t stepSize = (d.size()/steps);
    for(std::size_t i =0;i<steps;i++) {
        auto element = std::max_element(d.begin()+i*stepSize,d.begin()+(i+1)*stepSize);
        auto distance = element - d.begin();
        max[i] = static_cast<uint16_t>(distance);
    }
    return max;
}

std::vector<uint16_t> ImageDescriptor::Horizont::detectMaxDifference(const std::vector<uint16_t> &d) {
    std::size_t steps = 8;
    std::vector<uint16_t> max(steps);

    std::vector<std::pair<int, int>> diffVec(d.size());

    diffVec[0].first = 0;
    diffVec[0].second = 0;
    for(std::size_t i=1; i<d.size(); i++){
        diffVec[i].first = std::abs(static_cast<int>(d[i]) - d[i-1]);
        diffVec[i].second = static_cast<int>(i);
    }

    std::sort(diffVec.begin(), diffVec.end(),
              [](const std::pair<int, int> & a, const std::pair<int, int> & b) {
                  return a.first > b.first;
              }
    );

    for(std::size_t i=0; i<steps; i++){
        max[i] = static_cast<uint16_t>(diffVec[i].second);
    }

    return max;
}

std::vector<uint16_t> ImageDescriptor::Horizont::detectMax(const std::vector <uint16_t> &d) {
    return detectMaxDifference(d);
}

ImageDescriptor::Horizont::Description ImageDescriptor::Horizont::DescribeImage(const Img &img) {
    cv::Mat tresholded = ExtractHorizontByOtsu(img);

    std::vector<uint16_t> desc(static_cast<unsigned int>(tresholded.cols));

    for(int i=0;i<tresholded.cols;++i) {
        for(int j=1;j< tresholded.rows;++j) {
            if(tresholded.at<uchar>(j,i) != 255 || j == tresholded.rows-1 ) {
                desc[i]=static_cast<uint16_t >(tresholded.rows-j);
                break;
            }
        }
    }

    //inverse
    std::vector<uint16_t> descInv(static_cast<unsigned int>(tresholded.cols));

    cv::Mat floodFillInv;
    cv::bitwise_not(tresholded,floodFillInv);

    for(int i=0;i<floodFillInv.cols;++i) {
        if(floodFillInv.at<uchar>(1,i) == 0) {
            cv::floodFill(floodFillInv,cv::Point(i,1),cv::Scalar(255));
        }
    }

    cv::Mat filledHoles = (tresholded & floodFillInv);

    for(int i = filledHoles.cols - 1;i >= 0; --i) {
        for(int j = filledHoles.rows -1 ;j >= 1; --j) {
            if(filledHoles.at<uchar>(j,i) == 255 || j == 1) {
                descInv[i]=static_cast<uint16_t >(filledHoles.rows-j);
                break;
            }
        }
    }

    return {desc, descInv, detectMaxDifference(desc)};
}

int ImageDescriptor::Horizont::getVerticalShift(const Description& l, const Description &curr, int shift){
    // shift by mean difference
    int shiftY = 0;
    int overlap = 0;
    std::size_t max = l.horizont.size() - std::max(shift,0);
    for(std::size_t i = static_cast<size_t>(std::max(0, -shift));i<max;i++) {
        if(curr.horizont[i] + l.horizont[i + shift] < 479*2) {
            overlap++;
            shiftY += curr.horizont[i] - l.horizont[i + shift];
        }
    }

    shiftY = overlap ? shiftY / overlap : 0;

    return shiftY;
}

/**
 * Computes a difference between two horizons. The difference is composition of two components.
 * The first component is a mean difference between points in the same position.
 * The second component represents percentige of horizon that is close together (difference is below threshold).
 * For last part are included only the values that are not in the top of the image (<479).
 * @param l - horizon1
 * @param curr - horizon from current frame
 * @param shift - horizontal shift
 * @param shiftY - vertical shift
 * @return difference of the horizons l and curr
 */
double ImageDescriptor::Horizont::CompareShifted(const Description& l, const Description &curr, int shift, int shiftY)  {
    std::size_t distance = 0;

    const std::size_t max = l.horizont.size() - std::max(shift,0);
    int nontopOverlap = 0;

    // maximal vertical shift
    const int maxUp = 50;
    const int maxDown = 0;

    // maximal difference to be considered "close"
    const int maxDiff = 25;

    std::size_t diffCnt = 0;
    const uint16_t preClamp = static_cast<uint16_t>(clamp(std::abs(shiftY), maxDown, maxUp));

    // divided into separate branches because of optimalizations
    if(shiftY > 0) {
        for(std::size_t i = static_cast<size_t>(std::max(0,-shift));i<max;i++) {
            // equivalent to
            // if ((curr.horizont[i] < 479) && (l.horizont[i + shift] < 479)) ++nontopOverlap;
            // but not producing a jmp instruction and thus providing a much needed speed up
            nontopOverlap += ((curr.horizont[i] < 479) & (l.horizont[i + shift] < 479));

            uint16_t tmpDist = std::abs(std::min(l.horizont[i + shift] + preClamp, 480) - curr.horizont[i])
                      + std::abs(std::min(l.horizontInverse[i + shift] + preClamp, 480) - curr.horizontInverse[i]);
            distance += tmpDist;
            diffCnt += tmpDist > maxDiff;
        }
    } else if (shiftY < 0) {
        for(std::size_t i = static_cast<size_t>(std::max(0,-shift));i<max;i++) {
            nontopOverlap += ((curr.horizont[i] < 479) & (l.horizont[i + shift] < 479));

            uint16_t tmpDist = std::abs(std::min(curr.horizont[i] + preClamp, 480) - l.horizont[i + shift])
                      + std::abs(std::min(curr.horizontInverse[i] + preClamp, 480) - l.horizontInverse[i + shift]);
            distance += tmpDist;
            diffCnt += tmpDist > maxDiff;
        }
    } else { // shiftY == 0
        for(std::size_t i = static_cast<size_t>(std::max(0,-shift));i<max;i++) {
            nontopOverlap += ((curr.horizont[i] < 479) & (l.horizont[i + shift] < 479));

            uint16_t tmpDist = std::abs(l.horizont[i + shift] - curr.horizont[i])
                      + std::abs(l.horizontInverse[i + shift] - curr.horizontInverse[i]);
            distance += tmpDist;
            diffCnt += tmpDist > maxDiff;
        }
    }

    return (static_cast<double>(distance)/(l.horizont.size()-abs(shift))) + 100*(static_cast<double>(diffCnt) / nontopOverlap);
}

double ImageDescriptor::Horizont::Compare(const Description& l, const Description &curr) {
    double bestScore = std::numeric_limits<float>::max();
    for(std::size_t x=0;x<l.maximums.size();x++) {
        for(std::size_t y=0;y<curr.maximums.size();y++) {
            int shift = l.maximums[x] - curr.maximums[y];
            if(shift < 250 && shift > -250) {
                int shiftY = getVerticalShift(l, curr, shift);
                double score = CompareShifted(l, curr, shift, shiftY);
                if(score < bestScore) {
                    bestScore = score;
                }
            }
        }
    }

    return bestScore;
}

void ImageDescriptor::Horizont::descToIMG(cv::Mat &img, const Description &desc) {
    for(std::size_t i=0;i< desc.horizont.size()-1;i++) {
        if(i % 50 == 0)
            cv::line(img,{static_cast<int>(i),0},{static_cast<int>(i),640},cv::Scalar(200,100,255),1);
        cv::line(img,{static_cast<int>(i),img.rows-desc.horizont[i]},{static_cast<int>(i+1),img.rows- (desc.horizont[i+1])},cv::Scalar(0,50,255),2);
        cv::line(img,{static_cast<int>(i),img.rows-desc.horizontInverse[i]},{static_cast<int>(i+1),img.rows- (desc.horizontInverse[i+1])},cv::Scalar(0,0,255),2);
    }
}

cv::Mat ImageDescriptor::Horizont::compareToIMG(const Description& l, const Description& r, const int shift, const int shiftY) {
    cv::Mat img(480,l.horizont.size(),CV_8UC3,cv::Scalar(0));

    std::size_t max = l.horizont.size() - std::max(shift,0);

    for(std::size_t i= static_cast<std::size_t>(std::max(0,-shift));i<max;i++) {
        cv::line(img,{static_cast<int>(i),img.rows-(l.horizont[i+shift] + shiftY)},{static_cast<int>(i+1),img.rows- (l.horizont[i+1+shift] + shiftY)},cv::Scalar(0,50,255),2);
        cv::line(img,{static_cast<int>(i),img.rows-r.horizont[i]},{static_cast<int>(i+1),img.rows- (r.horizont[i+1])},cv::Scalar(255,50,25),2);
    }

    return img;
}