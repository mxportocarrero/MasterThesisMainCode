#ifndef DIRECT_ODOMETRY_HPP
#define DIRECT_ODOMETRY_HPP

#include "settings.hpp"
#include "general_includes.hpp"
#include "linear_algebra_functions.hpp"
#include "utilities.hpp"

// Downscale Class
//-----------------
// Clase para encontrar las imagenes reducidas para el algoritmo de alineamiento
enum image_type {intensity, depth, intrinsics};
class DownscaleBaseInterface
{
public:
	// Esta funcion se encarga de calcular los valores para el reescalado en un nivel
	// Es decir, reduciendo la imagen a la mitad
	void doDownscale(const cv::Mat& input, cv::Mat& output,int type)
	{
		switch(type)
		{
			case intensity:
				DownscaleImg(input,output);
				break;
			case depth:
				DownscaleDepth(input,output);
				break;
			case intrinsics:
				DownscaleIntrinsics(input,output);
				break;
			default:
				break;
		}
	}
	// AbstractFunctions
	virtual void DownscaleImg(const cv::Mat& input, cv::Mat& output) = 0;
	virtual void DownscaleDepth(const cv::Mat& input, cv::Mat& output) = 0;
	virtual void DownscaleIntrinsics(const cv::Mat& input, cv::Mat& output) = 0;
}; // Fin de DownscaleBaseInterface

class Downscale1 : public DownscaleBaseInterface
{
public:
	void DownscaleImg(const cv::Mat& input, cv::Mat& output)
	{
	    int rows = input.rows;
        int cols = input.cols;

        // creamos una matriz que calcula la mitad de la imagen
        output = cv::Mat::zeros(cv::Size(cols/2,rows/2),input.type());

        double num;
        for(int j = 0; j < rows/2; j++){
            for(int i = 0; i < cols/2; i++){
                num = input.at<double>(2*j,2*i) +
	                     input.at<double>(2*j+1,2*i) +
	                     input.at<double>(2*j,2*i+1) +
	                     input.at<double>(2*j+1,2*i+1);
                output.at<double>(j,i) = num / 4.0;
             }
        }
	} // Fin de DownscaleImg


	void DownscaleDepth(const cv::Mat& input, cv::Mat& output)
	{
	    int rows = input.rows;
        int cols = input.cols;

        // creamos una matriz que calcula la mitad de la imagen
        output = cv::Mat::zeros(cv::Size(cols/2,rows/2),input.type());

        double num;
        double a,b,c,d;
        double cont;
        for(int j = 0; j < rows/2; j++){
            for(int i = 0; i < cols/2; i++){
                cont = 0.0;
                //Contamos la cantidad de pixeles no nulos en la vecindad del pixel
                a = input.at<double>(2*j,2*i);
                b = input.at<double>(2*j+1,2*i);
                c = input.at<double>(2*j,2*i+1);
                d = input.at<double>(2*j+1,2*i+1);
                if(a)
                    cont += 1.0;
                if(b)
                    cont += 1.0;
                if(c)
                    cont += 1.0;
                if(d)
                    cont += 1.0;

                num = a + b + c + d;

                if(!cont) output.at<double>(j,i) = 0.0;
                else output.at<double>(j,i) = num / cont;
            }
        }
	} // Fin de DownscaleDepth

	void DownscaleIntrinsics(const cv::Mat& input, cv::Mat& output)
	{
		output = cv::Mat::eye(cv::Size(3,3),input.type());

        output.at<double>(0,0) = input.at<double>(0,0) / 2.0;
        output.at<double>(1,1) = input.at<double>(1,1) / 2.0;
        output.at<double>(0,2) = (input.at<double>(0,2) + 0.5) / 2.0 - 0.5;
        output.at<double>(1,2) = (input.at<double>(1,2) + 0.5) / 2.0 - 0.5;
	} // Fin de DownscaleIntrinsics

}; // Fin de Downscale1


/*
Clase DirectVisualAlignment
--------------------------
Clase Base para diferentes variaciones de DirectOdometry
*/

class DirectOdometryBase
{
protected:
	Settings *settings_;
	DownscaleBaseInterface *downscaler;
public:
	virtual void doAlignment(const cv::Mat& i0, const cv::Mat& d0, const cv::Mat& i1, myVector6d &xi, double& err) = 0;
};

class DirectOdometryA : public DirectOdometryBase
{
public:
	DirectOdometryA(Settings *settings)
	{
		settings_ = settings;
		downscaler = new Downscale1();
	}

	// doAlignment : esta funcion calcula un vector xi optimo que satisface el principio de fotoconsistencia
	// Input: 1 par rgbd de referencia y una imagen rgb consecutiva, parametros intrinsecos, referencia a un vector xi
	// Output: Vector xi con valores actualizados
	void doAlignment(const cv::Mat& i0_ref, const cv::Mat& d0_ref, const cv::Mat& i1_ref, myVector6d& xi, double& err);

	// Funcion que calcula los residuales entre imágenes
	void CalcDiffImage(const cv::Mat & i0, const cv::Mat & d0, const cv::Mat & i1,const cv::Mat& XGradient, const cv::Mat& YGradient, const myVector6d &xi, const cv::Mat& K, Eigen::VectorXd &Res, Eigen::MatrixXd &Jac);

	// Funcion para visualizar los residuales entre imágenes
	void CalcDiffImage(const cv::Mat & i0, const cv::Mat & d0, const cv::Mat & i1, const myVector6d &xi, const cv::Mat &K);

	// Estas funciones toman como referencia una imagen raw y forma el vector de mats piramidales
	// usando el downscaler
	void prepare_pyramidal_rgbs(const cv::Mat& in, std::vector<cv::Mat>& vec, int level);

	void prepare_pyramidal_depths(const cv::Mat& in, std::vector<cv::Mat>& vec , int level);

	void prepare_pyramidal_intrinsics(const cv::Mat& in, std::vector<cv::Mat>& vec, int level);

	// Funcion para calcular la gradiente en direccion X e Y
	void Gradient(const cv::Mat & InputImg, cv::Mat & OutputXImg, cv::Mat & OutputYImg);

	void interpolate(const cv::Mat& InputImg, cv::Mat& OutputImg, const cv::Mat& map_x, const cv::Mat& map_y, int padding);

}; // Fin del primer algoritmo



#endif // DIRECT_ODOMETRY_HPP






















