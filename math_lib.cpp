#pragma once

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <math.h>

using namespace cv;
using namespace std;

void rotate_face_no_crop(const Mat & src, Mat & dst, double roll)
{
  // Rotate a face contained in src and fill the rotate image in dst.
  // The size of dst increases depending on the angle, and the "empty" spots
  // are filled with black pixels

  cv::Point2f center((src.cols-1)/2.0, (src.rows-1)/2.0);
  Mat rot = cv::getRotationMatrix2D(center, roll, 1.0);
  // determine bounding rectangle, center not relevant
  Rect2f bbox = cv::RotatedRect(cv::Point2f(), src.size(), roll).boundingRect2f();
  // adjust transformation matrix
  rot.at<double>(0,2) += bbox.width/2.0 - src.cols/2.0;
  rot.at<double>(1,2) += bbox.height/2.0 - src.rows/2.0;

  cv::warpAffine(src, dst, rot, bbox.size() );
  // smooth_background(src, dst);
}

void rotate_face_crop(const Mat & src, Mat & dst, double roll){

  // Rotate a face contained in src and fill the rotate image in dst.
  // The size of dst is the same as src, so there may be the sensation of "cropping"
  cv::Point2f center((src.cols-1)/2.0, (src.rows-1)/2.0);
  Mat rot = cv::getRotationMatrix2D(center, roll, 1.0);

  cv::Size dsize( src.rows, src.cols );
  cv::warpAffine(src, dst, rot, dsize);

}

Mat rotate_face(const Mat & src, double roll, int width, int height, float extend_h=0.3, float extend_v=0.3)
{
  // The input to this function is the src image, the rotation angle, the dimensions of the final image.
  // The image is rotated, and to crop it around the center, we need the ratio between the original dimensions
  // and the dimensions of the crop area. It is 0.3 by default, which means that for a 128x128, the cropped area is
  // 2*0.3*128 ~ 78x78 pixels. This intermediate image is then resized to the expected width and height, and returned.
  // Note: the assumption is that src is 2x bigger than width and height. This will prevent unusual "croppings" from showing
  // and will also help us avoid using artificial or complex ways of filling black patches in the output matrix
  Mat dst;
  rotate_face_crop(src, dst, roll);

  Point center((src.cols)/2, (src.rows)/2);

  Point top_left( center.x - extend_v*src.cols, center.y-extend_h*src.rows );
  Point bottom_right(  center.x + extend_v*src.cols,  center.y+extend_h*src.rows);
  // Point top_left( center.x - src.cols/4, center.y-src.rows/4 );
  // Point bottom_right(  center.x + src.cols/4,  center.y+src.rows/4);
  Rect fr(  top_left, bottom_right );

  Mat out;
  resize(dst(fr), out, Size(width, height));

  return out;
}

Mat extend_face_to_raw_array(const Mat & bgr_image,
                             const cv::Rect & face_rect, const float * h_ext_ratio, const float * w_ext_ratio,
                             const int & selfie_width, const int & selfie_height,
                             float * facePatchOut, bool is_output_bgr = false) {
  int width = bgr_image.cols;
  int height = bgr_image.rows;
  int x = face_rect.x;
  int y = face_rect.y;
  int w = face_rect.width;
  int h = face_rect.height;
  // worth noting, here we extend face region in order to get
  // hair, also to preserve the aspect ratio of a square image
  int w_ext_left = int(w * w_ext_ratio[0]);
  int w_ext_right = int(w * w_ext_ratio[1]);
  int h_ext_top = int(h * h_ext_ratio[0]);
  int h_ext_bottom = int(h * h_ext_ratio[1]);
  int w_full = w + w_ext_left + w_ext_right;
  int h_full = h + h_ext_top + h_ext_bottom;
  // make sure face is in the middle, if goes out of bound,
  // fill the color black
  Mat facePatch(h_full, w_full, CV_8UC3, Scalar(0, 0, 0));
  int x_from     = max(x-w_ext_left, 0);
  int x_from_out = max(w_ext_left-x, 0);
  int y_from     = max(y-h_ext_top, 0);
  int y_from_out = max(h_ext_top-y, 0);
  int x_to = min(x+w+w_ext_right, width - 1);
  int y_to = min(y+h+h_ext_bottom, height - 1);
  bgr_image(Range(y_from, y_to), Range(x_from, x_to)).copyTo(facePatch(Range(y_from_out, y_to-y_from+y_from_out),
                                                                       Range(x_from_out, x_to-x_from+x_from_out)));
  resize(facePatch, facePatch, cv::Size(selfie_width, selfie_height));
#ifdef DEBUGCV
  imshow("facePatch", facePatch);
#endif
  // write to raw array format for attribute classifier
  if (is_output_bgr) {
    for (size_t i = 0; i < selfie_height; i++) {
      for (size_t j = 0; j < selfie_width; j++) {
        // facePatch is in BGR; one need to store the array as BGR
        facePatchOut[(i*selfie_width+j)*3]   = (facePatch.at<Vec3b>(i, j).val[0]);
        facePatchOut[(i*selfie_width+j)*3+1] = (facePatch.at<Vec3b>(i, j).val[1]);
        facePatchOut[(i*selfie_width+j)*3+2] = (facePatch.at<Vec3b>(i, j).val[2]);
      }
    }
  } else {
    for (size_t i = 0; i < selfie_height; i++) {
      for (size_t j = 0; j < selfie_width; j++) {
        // facePatch is in BGR; one need to store the array as RGB
        facePatchOut[(i*selfie_width+j)*3+2]   = (facePatch.at<Vec3b>(i, j).val[0]);
        facePatchOut[(i*selfie_width+j)*3+1] = (facePatch.at<Vec3b>(i, j).val[1]);
        facePatchOut[(i*selfie_width+j)*3] = (facePatch.at<Vec3b>(i, j).val[2]);
      }
    }
  }
  return facePatch;
}

Mat convertToFloatMat(vector<vector<float> > source) {
  Mat dst(source.size(), source[0].size(), CV_32F);
  for(int i = 0; i < source.size(); ++i)
    dst.row(i) = Mat(source[i]).t();
  return dst;
}

void procrustes(const Mat & data1, const Mat & data2,
                float & translateX, float & translateY, float & scaling, float & rotation) {
  Mat shapeMean, templateMean;
  Mat fromShape = data2.clone();
  Mat toTemplate = data1.clone();
  reduce(fromShape, shapeMean, 0, REDUCE_AVG);
  fromShape -= Mat::ones(fromShape.rows, 1, CV_32F) * shapeMean;

  reduce(toTemplate, templateMean, 0, REDUCE_AVG);
  toTemplate -= Mat::ones(toTemplate.rows, 1, CV_32F) * templateMean;

  scaling = norm(toTemplate)/norm(fromShape);
  fromShape = fromShape * scaling;

  float top = 0;
  float bottom = 0;
  for (size_t i=0; i<fromShape.rows; i++) {
    top += (fromShape.at<float>(i, 0) * toTemplate.at<float>(i, 1) - fromShape.at<float>(i, 1) * toTemplate.at<float>(i, 0));
    bottom += (fromShape.at<float>(i, 0) * toTemplate.at<float>(i, 0) + fromShape.at<float>(i, 1) * toTemplate.at<float>(i, 1));
  }
  rotation = atan(top/bottom);
  translateX = templateMean.at<float>(0,0)-scaling*(cos(rotation)*shapeMean.at<float>(0,0)+\
                                                    sin(-rotation)*shapeMean.at<float>(0,1));
  translateY = templateMean.at<float>(0,1)-scaling*(sin(rotation)*shapeMean.at<float>(0,0)+\
                                                    cos(rotation)*shapeMean.at<float>(0,1));
}

// http://euclideanspace.com/maths/geometry/rotations/conversions/angleToEuler/index.htm
void axisAngleToEuler(double angle, double x, double y, double z, double * yaw_pitch_roll) {
  double s=sin(angle);
	double c=cos(angle);
	double t=1-c;
	if ((x*y*t + z*s) > 0.998) { // north pole singularity detected
		yaw_pitch_roll[0] = 2*atan2(x*sin(angle/2), cos(angle/2));
		yaw_pitch_roll[2] = M_PI/2;
		yaw_pitch_roll[1] = 0;
		return;
	}
	if ((x*y*t + z*s) < -0.998) { // south pole singularity detected
		yaw_pitch_roll[0] = -2*atan2(x*sin(angle/2), cos(angle/2));
		yaw_pitch_roll[2] = -M_PI/2;
		yaw_pitch_roll[1] = 0;
		return;
	}
	yaw_pitch_roll[0] = atan2(y * s- x * z * t , 1 - (y*y+ z*z ) * t);
	yaw_pitch_roll[2] = asin(x * y * t + z * s) ;
	yaw_pitch_roll[1] = atan2(x * s - y * z * t , 1 - (x*x + z*z) * t);
}

// not prefered, use axisAngleToEuler
// http://euclideanspace.com/maths/geometry/rotations/conversions/matrixToEuler/index.htm
void rotationMatrixToEuler(const Mat & m, double * yaw_pitch_roll) {
	if (m.at<double>(1, 0) > 0.998) { // singularity at north pole
		yaw_pitch_roll[0] = atan2(m.at<double>(0, 2), m.at<double>(2, 2));
		yaw_pitch_roll[2] = M_PI/2;
		yaw_pitch_roll[1] = 0;
		return;
	}
	if (m.at<double>(1, 0) < -0.998) { // singularity at south pole
		yaw_pitch_roll[0] = atan2(m.at<double>(0, 2), m.at<double>(2, 2));
		yaw_pitch_roll[2] = -M_PI/2;
		yaw_pitch_roll[1] = 0;
		return;
	}
	yaw_pitch_roll[0] = atan2(-m.at<double>(2, 0), m.at<double>(0, 0));
	yaw_pitch_roll[1] = atan2(-m.at<double>(1, 2), m.at<double>(1, 1));
	yaw_pitch_roll[2] = asin(m.at<double>(1, 0));
}

void eulerToRotationMatrix(const double * yaw_pitch_roll, Mat & m) {
  // Assuming the angles are in radians.
  double ch = cos(yaw_pitch_roll[0]);
  double sh = sin(yaw_pitch_roll[0]);
  double ca = cos(yaw_pitch_roll[2]);
  double sa = sin(yaw_pitch_roll[2]);
  double cb = cos(yaw_pitch_roll[1]);
  double sb = sin(yaw_pitch_roll[1]);

  m.at<double>(0, 0) = ch * ca;
  m.at<double>(0, 1) = sh*sb - ch*sa*cb;
  m.at<double>(0, 2) = ch*sa*sb + sh*cb;
  m.at<double>(1, 0) = sa;
  m.at<double>(1, 1) = ca*cb;
  m.at<double>(1, 2) = -ca*sb;
  m.at<double>(2, 0) = -sh*ca;
  m.at<double>(2, 1) = sh*sa*cb + ch*sb;
  m.at<double>(2, 2) = -sh*sa*sb + ch*cb;
}
// find out median of a matrix
double medianMat(Mat input){
  input = input.reshape(0,1); // spread Input Mat to single row
  std::vector<double> vecFromMat;
  input.copyTo(vecFromMat); // Copy Input Mat to vector vecFromMat
  std::nth_element(vecFromMat.begin(), vecFromMat.begin() + vecFromMat.size() / 2, vecFromMat.end());
  return vecFromMat[vecFromMat.size() / 2];
}

// data is assembled as:
// left eye   | x0 y0 |
// right eye  | x1 y1 |
// nose       | x2 y2 |
// left mouth corner
// right mouth corner
double calculateYawRatio(const vector<Point2d> & image_points) {
  double x1_x0 = image_points[1].x - image_points[0].x;
  double y1_y0 = image_points[1].y - image_points[0].y;
  double x0_x2 = image_points[0].x - image_points[2].x;
  double y0_y2 = image_points[0].y - image_points[2].y;
  return (- y0_y2 * y1_y0 - x0_x2 * x1_x0) / (x1_x0 * x1_x0 + y1_y0 * y1_y0);
}

// data is assembled as:
// left eye   | x0 y0 |
// right eye  | x1 y1 |
// nose       | x2 y2 |
float calculateYawRatio2(const Mat & data) {
  float x1_x0 = data.at<float>(1, 0) - data.at<float>(0, 0);
  float y1_y0 = data.at<float>(1, 1) - data.at<float>(0, 1);
  float x0_x2 = data.at<float>(0, 0) - data.at<float>(2, 0);
  float y0_y2 = data.at<float>(0, 1) - data.at<float>(2, 1);
  return (- y0_y2 * y1_y0 - x0_x2 * x1_x0) / (x1_x0 * x1_x0 + y1_y0 * y1_y0);
}
