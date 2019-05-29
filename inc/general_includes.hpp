#ifndef GENERAL_INCLUDES_HPP
#define GENERAL_INCLUDES_HPP

// Import Standar Libraries
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <tuple>
#include <iomanip>
#include <stdio.h>

// Import Intrinsic Directives
#include <immintrin.h>

// Import OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

// Librerias usadas en la visualizacion
#include <opencv2/viz.hpp>
#include <opencv2/calib3d.hpp>


// Import Eigen
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions> // Para hacer uso de esta libreria
// tuve que modificar algunos includes en el source code que utiliza por q referenciaba
// #include <Eigen/Core> pero deberia ser en mi caso #include <eigen3/Eigen/Core>

// Algunos typedefs importantes
// necesito redeclarar este vector por que el default usado por Eigen presenta issues de alocacion cuando
// se compila con parametros optimizados (-O3)
typedef Eigen::Matrix<double,6,1,Eigen::DontAlign> myVector6d;

/** MACROS **/

#define FOR(i,n) for(int i = 0; i < n; ++i)

#endif // GENERAL_INCLUDES_HPP