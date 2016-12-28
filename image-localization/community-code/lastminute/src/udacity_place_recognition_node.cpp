#include <iostream>
#include <vector>
#include <string>
#include <stdlib.h>

// DLoopDetector and DBoW2
#include <DBoW2/DBoW2.h> // defines BriefVocabulary
#include <DVision/DVision.h> // Brief

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// ROS
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include "udacity_place_recognition/PlaceDetector.h"

using namespace DLoopDetector;
using namespace DBoW2;
using namespace DVision;
using namespace std;


/// This functor extracts BRIEF descriptors in the required format
class BriefExtractor: public FeatureExtractor<FBrief::TDescriptor>
{
public:
  /**
   * Extracts features from an image
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(const cv::Mat &im,
    vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const;

  /**
   * Creates the brief extractor with the given pattern file
   * @param pattern_file
   */
  BriefExtractor(const std::string &pattern_file);

private:

  /// BRIEF descriptor extractor
  DVision::BRIEF m_brief;
};



/// This functor extracts BRIEF descriptors in the required forma
class PlaceRecognitionNode
{
public:
  PlaceRecognitionNode()//:it_(nh_)
  {
    int32_t img_width, img_height;
    std::string brief_pattern_file, voc_file, image_dir, database_file, pose_file, test_image_dir;
    ros::param::param<int32_t>("~img_width", img_width, 640);
    ros::param::param<int32_t>("~img_height", img_height, 480);
    ros::param::param<std::string>("~brief_pattern_file", brief_pattern_file, "/tmp/brief_pattern_file.yml");
    ros::param::param<std::string>("~voc_file", voc_file, "/tmp/voc_file.tar.gz");
    ros::param::param<std::string>("~image_dir", image_dir, "/tmp/images");
    ros::param::param<std::string>("~test_image_dir", test_image_dir, "");
    ros::param::param<std::string>("~pose_file", pose_file, "/tmp/pose.txt");

    ROS_INFO_STREAM("brief_pattern_file: " << brief_pattern_file);
    extractor_.reset(new BriefExtractor(brief_pattern_file));
    ROS_INFO_STREAM("voc_file: " << voc_file);
    ROS_INFO_STREAM("image_dir: " << image_dir);
    ROS_INFO_STREAM("test_image_dir: " << test_image_dir);
    ROS_INFO_STREAM("pose_file: " << pose_file);

    place_detector_.reset(new PlaceDetector<BriefVocabulary, BriefLoopDetector, FBrief::TDescriptor>(
        voc_file, image_dir, pose_file, img_width, img_height));

    try
    {
      // run the demo with the given functor to extract features
      place_detector_->load(*extractor_, detector_);
    }
    catch(const std::string &ex)
    {
      ROS_INFO_STREAM("Error: " << ex);
      throw std::runtime_error("exception caught");
    }

    // run the demo with the given functor to extract features
    if (!test_image_dir.empty())
    {
      place_detector_->run(test_image_dir, *extractor_, *detector_);
    }

    //img_sub_ = it_.subscribe("/center_camera/image_color", 10, &PlaceRecognitionNode::imgCb, this);
  }

  void imgCb(const sensor_msgs::ImageConstPtr& img)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(img, "mono8");
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }

    place_detector_->detect(cv_ptr->image, *extractor_, *detector_);
  }

private:
  std::shared_ptr<BriefExtractor> extractor_;
  std::shared_ptr<PlaceDetector<BriefVocabulary, BriefLoopDetector, FBrief::TDescriptor> > place_detector_;
  ros::NodeHandle nh_;
  //image_transport::Subscriber img_sub_;
  std::shared_ptr<BriefLoopDetector> detector_;
};

// ----------------------------------------------------------------------------

BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
  // The DVision::BRIEF extractor computes a random pattern by default when
  // the object is created.
  // We load the pattern that we used to build the vocabulary, to make
  // the descriptors compatible with the predefined vocabulary

  // loads the pattern
  cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
  if(!fs.isOpened()) throw string("Could not open file ") + pattern_file;

  vector<int> x1, y1, x2, y2;
  fs["x1"] >> x1;
  fs["x2"] >> x2;
  fs["y1"] >> y1;
  fs["y2"] >> y2;

  m_brief.importPairs(x1, y1, x2, y2);
}

// ----------------------------------------------------------------------------

void BriefExtractor::operator() (const cv::Mat &im,
  vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const
{
  // extract FAST keypoints with opencv
  const int fast_th = 20; // corner detector response threshold
  cv::FAST(im, keys, fast_th, true);

  // compute their BRIEF descriptor
  m_brief.compute(im, keys, descriptors);
}

// ----------------------------------------------------------------------------

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

int main(int argc, char** argv)
{
  ros::init(argc, argv, "place_recognition_node");

  PlaceRecognitionNode prn;
  ros::spin();

  return 0;
}


