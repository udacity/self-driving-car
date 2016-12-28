/**
 * File: PlaceDetector.h
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DLoopDetector
 * License: see the LICENSE.txt file
 */

#ifndef __PLACE_DETECTOR__
#define __PLACE_DETECTOR__

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// DBoW2
#include <DBoW2/DBoW2.h>
#include <DUtils/DUtils.h>
#include <DUtilsCV/DUtilsCV.h>
#include <DVision/DVision.h>

#include "TemplatedLoopDetector.h"
#include "FBrief.h"

using namespace DBoW2;
using namespace DLoopDetector;
using namespace std;

/// BRIEF Loop Detector
typedef DLoopDetector::TemplatedLoopDetector
  <FBrief::TDescriptor, FBrief> BriefLoopDetector;

/// Generic class to create functors to extract features
template<class TDescriptor>
class FeatureExtractor
{
public:
  /**
   * Extracts features
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(const cv::Mat &im,
    vector<cv::KeyPoint> &keys, vector<TDescriptor> &descriptors) const = 0;
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/// @param TVocabulary vocabulary class (e.g: Surf64Vocabulary)
/// @param TDetector detector class (e.g: Surf64LoopDetector)
/// @param TDescriptor descriptor class (e.g: vector<float> for SURF)
template<class TVocabulary, class TDetector, class TDescriptor>

/// Class to run the demo
class PlaceDetector
{
public:

  /**
   * @param vocfile vocabulary file to load
   * @param imagedir directory to read images from
   * @param posefile pose file
   * @param width image width
   * @param height image height
   */
  PlaceDetector(const std::string &vocfile, const std::string &imagedir,
    const std::string &posefile, int width, int height);

  ~PlaceDetector(){}

  /**
   * Runs the demo
   * @param name demo name
   * @param extractor functor to extract features
   */
  void run(const std::string &test_image_dir,
    const FeatureExtractor<TDescriptor> &extractor, TDetector& detector);
  void load(const FeatureExtractor<TDescriptor> &extractor, std::shared_ptr<TDetector>& detector);
  void detect(const cv::Mat& im, const FeatureExtractor<TDescriptor> &extractor, TDetector& detector);

protected:

  /**
   * Reads the robot poses from a file
   * @param filename file
   * @param xs
   * @param ys
   */
  void readPoseFileLLA(const char *filename, std::vector<double> &xs,
     std::vector<double> &ys) const;

protected:

  std::string m_vocfile;
  std::string m_imagedir;
  std::string m_posefile;
  int m_width;
  int m_height;
  vector<double> xs, ys;
  int m_last_match;
  int m_last_match_index;
};

// ---------------------------------------------------------------------------
template<class TVocabulary, class TDetector, class TDescriptor>
PlaceDetector<TVocabulary, TDetector, TDescriptor>::PlaceDetector
  (const std::string &vocfile, const std::string &imagedir,
  const std::string &posefile, int width, int height)
  : m_vocfile(vocfile), m_imagedir(imagedir), m_posefile(posefile),
    m_width(width), m_height(height), m_last_match(0), m_last_match_index(0)
{
}



template<class TVocabulary, class TDetector, class TDescriptor>
void PlaceDetector<TVocabulary, TDetector, TDescriptor>::load
  (const FeatureExtractor<TDescriptor> &extractor,
   std::shared_ptr<TDetector>& detector)
{
  // Set loop detector parameters
  typename TDetector::Parameters params(m_height, m_width);

  // We are going to change these values individually:
  params.use_nss = true; // use normalized similarity score instead of raw score
  params.alpha = 0.3; // nss threshold
  params.k = 3; // a loop must be consistent with 1 previous matches
  params.max_neighbor_ratio = 0.75;
  params.max_ransac_iterations = 200;
  params.geom_check = GEOM_DI; // use direct index for geometrical checking
  params.di_levels = 4; // use four direct index levels

  // Load the vocabulary to use
  ROS_INFO_STREAM("Loading vocabulary...");
  TVocabulary voc(m_vocfile);

  // Initiate loop detector with the vocabulary
  ROS_INFO_STREAM("Processing sequence...");
  detector.reset(new TDetector(voc, params));

  // Process images
  vector<cv::KeyPoint> keys;
  vector<TDescriptor> descriptors;

  // load image filenames
  vector<string> filenames =
    DUtils::FileFunctions::Dir(m_imagedir.c_str(), ".jpg", true);
  ROS_INFO("Found %lu files in %s", filenames.size(), m_imagedir.c_str());

  // load robot poses
  ROS_INFO("Loading LLA...");
  readPoseFileLLA(m_posefile.c_str(), xs, ys);
  ROS_INFO("Read %lu poses from file %s", xs.size(), m_posefile.c_str());

  // we can allocate memory for the expected number of images
  detector->allocate(filenames.size());

  ROS_INFO_STREAM("loading " << filenames.size() << " images!");
  int count = 0;
  for(unsigned int i = 0; i < filenames.size(); i++)
  {
    ROS_INFO_STREAM("Adding image " << i << ": " << filenames[i] << "... ");

    // get image
    cv::Mat im = cv::imread(filenames[i].c_str(), 0); // grey scale

    // get features
    extractor(im, keys, descriptors);

    // add image to the collection and check if there is some loop
    detector->add(keys, descriptors);
  }
}

template<class TVocabulary, class TDetector, class TDescriptor>
void PlaceDetector<TVocabulary, TDetector, TDescriptor>::detect
    (const cv::Mat& im, const FeatureExtractor<TDescriptor> &extractor, TDetector& detector)
{
  // Process images
  vector<cv::KeyPoint> keys;
  vector<TDescriptor> descriptors;
  int count = 0;

  // go
  ROS_INFO("Detect image... ");

  // get features
  extractor(im, keys, descriptors);

  // add image to the collection and check if there is some loop
  DetectionResult result;

  detector.detectWithoutAdd(keys, descriptors, result);

  if(result.detection())
  {
    ROS_INFO_STREAM("- Loop found with image " << result.match << "!");
    ++count;
  }
  else
  {
    ROS_INFO("- No loop: ");
    switch(result.status)
    {
      case CLOSE_MATCHES_ONLY:
        ROS_INFO("All the images in the database are very recent");
        break;

      case NO_DB_RESULTS:
        ROS_INFO("There are no matches against the database (few features in the image?)");
        break;

      case LOW_NSS_FACTOR:
        ROS_INFO("Little overlap between this image and the previous one");
        break;

      case LOW_SCORES:
        ROS_INFO("No match reaches the score threshold: ");
        break;

      case NO_GROUPS:
        ROS_INFO_STREAM("Not enough close matches to create groups. Best candidate: " << result.match);
        break;

      case NO_TEMPORAL_CONSISTENCY:
        ROS_INFO_STREAM("No temporal consistency. Best candidate: " << result.match);
        break;

      case NO_GEOMETRICAL_CONSISTENCY:
        ROS_INFO_STREAM("No geometrical consistency. Best candidate: " << result.match);
        break;

      default:
        break;
    }
  }

  if(count == 0)
  {
    ROS_INFO("No loops found in this image");
  }
  else
  {
    ROS_INFO_STREAM(count << " loops found in this image!");
  }
}


// ---------------------------------------------------------------------------

template<class TVocabulary, class TDetector, class TDescriptor>
void PlaceDetector<TVocabulary, TDetector, TDescriptor>::run(const std::string &test_image_dir,
    const FeatureExtractor<TDescriptor> &extractor, TDetector& detector)
{
  // Process images
  vector<cv::KeyPoint> keys;
  vector<TDescriptor> descriptors;

  // load image filenames
  vector<string> filenames =
    DUtils::FileFunctions::Dir(test_image_dir.c_str(), ".png", true);

  // we can allocate memory for the expected number of images
  detector.allocate(filenames.size());

  // prepare profiler to measure times
  DUtils::Profiler profiler;
  int count = 0;

  std::ofstream output_file("/tmp/result.csv", std::ofstream::out);

  // go
  for(unsigned int i = 0; i < filenames.size(); i+=1)
  {
    ROS_INFO_STREAM(filenames[i]);

    // get image
    cv::Mat im = cv::imread(filenames[i].c_str(), 0); // grey scale

    // get features
    profiler.profile("features");
    extractor(im, keys, descriptors);
    profiler.stop();

    // add image to the collection and check if there is some loop
    DetectionResult result;

    profiler.profile("detection");
    detector.detectWithoutAdd(keys, descriptors, result);
    profiler.stop();

    if(result.detection() && result.match<xs.size())
    {
      ROS_INFO("Matched. %.10f, %.10f", xs[result.match], ys[result.match]);
      output_file << filenames[i] << "," << xs[result.match] << "," << ys[result.match] << std::endl;
      m_last_match = result.match;
      m_last_match_index = i;

      ++count;
    }
    else
    {
      switch(result.status)
      {
        case CLOSE_MATCHES_ONLY:
        case NO_DB_RESULTS:
        case LOW_NSS_FACTOR:
        case LOW_SCORES:
        case NO_GROUPS:
        case NO_TEMPORAL_CONSISTENCY:
        case NO_GEOMETRICAL_CONSISTENCY:
          ROS_INFO("Failed, %.10f, %.10f",xs[m_last_match],ys[m_last_match]);
          output_file << filenames[i] << "," << xs[m_last_match] << "," << ys[m_last_match] << std::endl;
          break;

        default:
          break;
      }
    }
  }
  output_file.close();

  if(count == 0)
  {
    ROS_INFO("No loops found in this image sequence");
  }
  else
  {
    ROS_INFO_STREAM(count << " loops found in this image sequence!");
  }
}


template<class TVocabulary, class TDetector, class TDescriptor>
void PlaceDetector<TVocabulary, TDetector, TDescriptor>::readPoseFileLLA
  (const char *filename, std::vector<double> &xs, std::vector<double> &ys)
  const
{
  xs.clear();
  ys.clear();

  ifstream infile(filename);
  while (infile)
  {
    string s;
    if (!getline( infile, s )) break;

    istringstream ss( s );
    vector <string> record;

    while (ss)
    {
      string s;
      if (!getline( ss, s, ',' )) break;
      record.push_back( s );
    }
    if(record.size()==3)
    {
      double x = stod(record[1]);
      double y = stod(record[2]);
      xs.push_back(x);
      ys.push_back(y);
    }
  }
  infile.close();
}

// ---------------------------------------------------------------------------

#endif
