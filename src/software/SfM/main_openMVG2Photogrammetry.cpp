// This file is part of OpenMVG, an Open Multiple View Geometry C++ library.

/* v.0.17 October 14th, 2015
 * Kevin CAIN, www.insightdigital.org
 * Adapted from the openMVG libraries,
 * Copyright (c) 2012-2015 Pierre MOULON.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include "openMVG/cameras/Camera_Pinhole.hpp"
#include "openMVG/cameras/Camera_undistort_image.hpp"
#include "openMVG/features/feature.hpp"
#include "openMVG/image/image_io.hpp"
#include "openMVG/image/image_resampling.hpp"
#include "openMVG/sfm/sfm_data.hpp"
#include "openMVG/sfm/sfm_data_io.hpp"

using namespace openMVG;
using namespace openMVG::cameras;
using namespace openMVG::geometry;
using namespace openMVG::image;
using namespace openMVG::sfm;
using namespace openMVG::features;

#include "third_party/cmdLine/cmdLine.h"
#include "third_party/progress/progress_display.hpp"
#include "third_party/stlplus3/filesystemSimplified/file_system.hpp"
#include <cstdlib>
#include <cmath>
#include <iterator>
#include <iomanip>
#include <atomic>

/// Naive image bilinear resampling of an image for thumbnail generation
template <typename ImageT>
ImageT
create_thumbnail
(
  const ImageT & image,
  int thumb_width,
  int thumb_height
);

//convert to photogrammetry
void getOPKFromRotation(const Mat3 & rotation, 
  double &omega, double &phi, double &kappa)
{
  omega = atan2(-rotation(2,1), rotation(2,2));
  kappa = atan2(-rotation(1,0), rotation(0,0));
  phi   = atan2( rotation(2,0), rotation(2,2)/cos(omega));
}

bool exportToPhotogrammetryFormat(
  const SfM_Data & sfm_data,
  const std::string & sImageDir,
  const std::string & sOutDirectory // Output files directory
  )
{

  // Make Dir
  std::string photogrammetry_images_path = sOutDirectory + "/images";
  if ( !stlplus::folder_exists( photogrammetry_images_path ) )
  {
    if ( !stlplus::folder_create( photogrammetry_images_path ))
    {
      std::cerr << "\nCannot create output directory" << std::endl;
      return EXIT_FAILURE;
    }
  }

  // Export the SfM_Data scene to the Photogrammetry format
  {
    // Prepare to write bundle file
    // Get cameras and features from OpenMVG
    const Views & views = sfm_data.GetViews();
    const size_t cameraCount = views.size();
    // Tally global set of feature landmarks
    const Landmarks & landmarks = sfm_data.GetLandmarks();
    const size_t featureCount = landmarks.size();
    const std::string bundle_filename = "bundle.txt";
    const std::string pointcloud_filename = "point_sparse.xyz";
    std::cout << "Writing bundle (" << cameraCount << " cameras ): to " << bundle_filename << "...\n";
    std::ofstream out(stlplus::folder_append_separator(sOutDirectory) + bundle_filename);
    out.setf(std::ios::fixed, std::ios::floatfield);  
    out.precision(6);  
    for (const auto & views_it : views)
    {
        const View * view = views_it.second.get();
        if (sfm_data.IsPoseAndIntrinsicDefined(view))
        {
            Intrinsics::const_iterator iterIntrinsic = sfm_data.GetIntrinsics().find(view->id_intrinsic);
            const IntrinsicBase * cam = iterIntrinsic->second.get();
            const Pose3 pose = sfm_data.GetPoseOrDie(view);
            const Pinhole_Intrinsic * pinhole_cam = static_cast<const Pinhole_Intrinsic *>(cam);
            Mat3 trans_mat;
            trans_mat(0,0) = 1;trans_mat(0,1) = 0;trans_mat(0,2) = 0;
            trans_mat(1,0) = 0;trans_mat(1,1) =-1;trans_mat(1,2) = 0;
            trans_mat(2,0) = 0;trans_mat(2,1) = 0;trans_mat(2,2) =-1;
            Mat3  rotation = trans_mat*pose.rotation();
          
            double omega,phi,kappa;
            getOPKFromRotation(rotation,omega,phi,kappa);
            const Vec3 center = pose.center();

            out
              << view->s_Img_path << "\t" << pinhole_cam->focal() << "\t"
              << center[0] << "\t" << center[1] << "\t" << center[2] << "\t"
              << omega<< "\t" << phi << "\t" << kappa << "\n";             
        }
    }
    out.close();

    std::cout << "Writing Point Cloud (" << featureCount << " points ): to " << pointcloud_filename << "...\n";
    std::ofstream out2(stlplus::folder_append_separator(sOutDirectory) + pointcloud_filename);
    out2<<"#X,Y,Z"<<std::endl;
    out2.setf(std::ios::fixed, std::ios::floatfield);  
    out2.precision(6);  
    // For each feature, write to bundle: position XYZ[0-3]
    for (const auto & landmarks_it : landmarks)
    {
      const Vec3 & exportPoint = landmarks_it.second.X;
      out2 << exportPoint.x() << "," << exportPoint.y() << "," << exportPoint.z() << "\n";
    }
    out2.close();

    // Export (calibrated) views as undistorted images in parallel
    std::cout << "Exporting views..." << std::endl;

    C_Progress_display my_progress_bar(views.size());

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(views.size()); ++i)
    {
      auto views_it = views.begin();
      std::advance(views_it, i);
      const View * view = views_it->second.get();

      if (!sfm_data.IsPoseAndIntrinsicDefined(view))
      {
        continue;
      }
          
    
      // We have a valid view with a corresponding camera & pose
      const std::string srcImage = stlplus::create_filespec(sImageDir, view->s_Img_path);
      const std::string dstImage = stlplus::create_filespec(photogrammetry_images_path, view->s_Img_path);


      Image<RGBColor> image, image_ud;
      Intrinsics::const_iterator iterIntrinsic = sfm_data.GetIntrinsics().find(view->id_intrinsic);
      const IntrinsicBase * cam = iterIntrinsic->second.get();
      if (cam->have_disto())
      {
        // Undistort and save the image
        if (!ReadImage(srcImage.c_str(), &image))
        {
          std::cerr
            << "Unable to read the input image as a RGB image:\n"
            << srcImage << std::endl;
          continue;
        }
        UndistortImage(image, cam, image_ud, BLACK);
        if (!WriteImage(dstImage.c_str(), image_ud))
        {
          std::cerr
            << "Unable to write the output image as a RGB image:\n"
            << dstImage << std::endl;
          continue;
        }
      }
      else // (no distortion)
      {
        // copy the PNG image
        stlplus::file_copy(srcImage, dstImage);
      }

      ++my_progress_bar;
    }
  }
  return true;
}

int main(int argc, char *argv[])
{

  CmdLine cmd;
  std::string sSfM_Data_Filename;
  std::string sImageDir = "";
  std::string sOutDir = "";
  cmd.add( make_option('i', sSfM_Data_Filename, "sfmdata") );
  cmd.add( make_option('p',sImageDir,"imagedir"));
  cmd.add( make_option('o', sOutDir, "outdir") );
  std::cout << "Note:  this program writes output in Photogrammetry format.\n";

  try {
      if (argc == 1) throw std::string("Invalid command line parameter.");
      cmd.process(argc, argv);
  } catch (const std::string& s) {
      std::cerr << "Usage: " << argv[0] << '\n'
      << "[-i|--sfmdata] filename, the SfM_Data file to convert\n"
      << "[-p|--imagedir] image dir \n"
      << "[-o|--outdir] path\n"
      << std::endl;

      std::cerr << s << std::endl;
      return EXIT_FAILURE;
  }

  // Create output dir
  if (!stlplus::folder_exists(sOutDir))
    stlplus::folder_create(sOutDir);

  // Read the input SfM scene
  SfM_Data sfm_data;
  if (!Load(sfm_data, sSfM_Data_Filename, ESfM_Data(ALL))) {
    std::cerr << std::endl
      << "The input SfM_Data file \""<< sSfM_Data_Filename << "\" cannot be read." << std::endl;
    return EXIT_FAILURE;
  }

  if (exportToPhotogrammetryFormat(sfm_data, sImageDir, sOutDir))
    return EXIT_SUCCESS;
  else
    return EXIT_FAILURE;
}

/// Naive image bilinear resampling of an image for thumbnail generation
/// Inspired by create_thumbnail from MVE (cropping is here ignored)
template <typename ImageT>
ImageT
create_thumbnail
(
  const ImageT & image,
  int thumb_width,
  int thumb_height
)
{
  const int width = image.Width();
  const int height = image.Height();
  const float image_aspect = static_cast<float>(width) / height;
  const float thumb_aspect = static_cast<float>(thumb_width) / thumb_height;

  int rescale_width, rescale_height;
  if (image_aspect > thumb_aspect)
  {
    rescale_width = std::ceil(thumb_height * image_aspect);
    rescale_height = thumb_height;
  }
  else
  {
    rescale_width = thumb_width;
    rescale_height = std::ceil(thumb_width / image_aspect);
  }

  // Generation of the sampling grid
  std::vector<std::pair<float,float>> sampling_grid;
  sampling_grid.reserve(rescale_height * rescale_width);
  for ( int i = 0; i < rescale_height; ++i )
  {
    for ( int j = 0; j < rescale_width; ++j )
    {
      const float dx = static_cast<float>(j) * width / rescale_width;
      const float dy = static_cast<float>(i) * height / rescale_height;
      sampling_grid.push_back( std::make_pair( dy , dx ) );
    }
  }

  const Sampler2d<SamplerLinear> sampler;
  ImageT imageOut;
  GenericRessample(image, sampling_grid, rescale_width, rescale_height, sampler, imageOut);
  return imageOut;
}
