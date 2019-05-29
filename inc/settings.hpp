#ifndef SETTINGS_H
#define SETTINGS_H

#include <cstdint>
#include "general_includes.hpp"

class Settings
{
public:
	// Atributos
	// ---------
	int32_t num_trees_,
		max_tree_depth_,
		num_frames_per_tree_,
		num_pixels_per_frame_,
		image_width_,
		image_height_,
		depth_factor_; // for TUM data sets

	// intrinsics
	double fx, fy, cx, cy;
	cv::Mat K_ref;


	
	// Constructores
	Settings() 
		: num_trees_(5), max_tree_depth_(16), 
		num_frames_per_tree_(500), num_pixels_per_frame_(5000),
		image_width_(640), image_height_(480),
		depth_factor_(5000),
		fx(525.0f), fy(525.0f), cx(319.5f), cy(239.5f)
		{
			double k[3][3] = {{ fx,   0.0,    cx},
			                 { 0.0,    fy,    cy},
			                 { 0.0,   0.0,   1.0}};

		    //Expresados en cv::Mat
		    K_ref = cv::Mat(cv::Size(3,3),CV_64F,k).clone();
		}

	Settings(int32_t num_trees, int32_t width, int32_t height, int32_t factor, double fx_, double fy_, double cx_, double cy_)
		: num_trees_(num_trees), max_tree_depth_(16), 
		num_frames_per_tree_(500), num_pixels_per_frame_(5000),
		image_width_(width), image_height_(height),
		depth_factor_(factor),
		fx(fx_), fy(fy_), cx(cx_), cy(cy_)
		{
			double k[3][3] = {{ fx,   0.0,    cx},
			                 { 0.0,    fy,    cy},
			                 { 0.0,   0.0,   1.0}};

		    //Expresados en cv::Mat
		    K_ref = cv::Mat(cv::Size(3,3),CV_64F,k).clone(); // usamos la funcion clone para q no se pierda la asignacion
		}
}; // FIN DE LA CLASE SETTINGS

#endif // SETTINGS_H