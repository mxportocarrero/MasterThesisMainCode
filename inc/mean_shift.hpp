#ifndef MEAN_SHIFT_HPP
#define MEAN_SHIFT_HPP

/*
 * MeanShift implementation created by Matt Nedrich.
 * project: https://github.com/mattnedrich/MeanShift_cpp
 *
 */

#include <stdio.h>
#include <vector>
#include <math.h>
#include <bitset>
#include "general_includes.hpp"

//using namespace std;

// Some required functions

//#define EPSILON 0.0000001
#define EPSILON 0.0001


double euclidean_distance(const Eigen::Vector3d &point_a, const Eigen::Vector3d &point_b);

double gaussian_kernel(double distance, double kernel_bandwidth);

// Clase MeanShift
// ---------------

class MeanShift
{
public:
    MeanShift();
    MeanShift(double (*_kernel_func)(double,double));

    std::vector<Eigen::Vector3d> cluster(std::vector<Eigen::Vector3d> points, double kernel_bandwidth);

private:
	// Este es uno de los atributos de la clase
    double (*kernel_func)(double,double);

    void set_kernel(double (*_kernel_func)(double,double));

    Eigen::Vector3d shift_point(const Eigen::Vector3d &point, const std::vector< Eigen::Vector3d > &points, double kernel_bandwidth);
    
}; // Fin de la Clase MeanShift

#endif // MEAN_SHIFT_HPP