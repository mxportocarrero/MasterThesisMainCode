#ifndef MAIN_SYSTEM_HPP
#define MAIN_SYSTEM_HPP

#include "direct_odometry.hpp"
#include "forest.hpp"

class MainSystemBase
{
protected:
	Dataset *data_;
	Settings *settings_;

	// Direct Odometry Algorithms
	DirectOdometryBase *direct_odometry_;

	// Regression Forest Algorithms
	Forest<ushort,cv::Vec3b> *forest_;
	std::string forest_file_name_;

	// Aqui guardaremos las correspondecias en puntos 3D
	// Para las comparaciones
	std::vector<Eigen::Vector3d> matched_coord_data_;
	std::vector<Eigen::Vector3d> matched_coord_gt_;

	// Vectores para guardar y comparar las rotaciones
	std::vector<Eigen::Matrix3d> matched_rot_data_;
	std::vector<Eigen::Matrix3d> matched_rot_gt_;

public:
	//Constructor
	MainSystemBase(Dataset *data, Settings *settings, DirectOdometryBase *direct_odometry, std::string forest_file_name)
		:data_(data), settings_(settings), direct_odometry_(direct_odometry), forest_file_name_(forest_file_name)
	{
		forest_ = new Forest<ushort,cv::Vec3b>(data_,settings_,forest_file_name_);
	}

	// Virtual Functions
	virtual void execute() = 0;

	// Esta funcion nos permitira previsualizar la posible reconstrucción
	// de la escena basandonos en las RBM calculadas por el algoritmo
	// Resultaria similar a la hecha en Forest.VerifyLabelData
	void displayReconstructionPreview(const cv::viz::Viz3d *window){
		
	}

	// Funcion para evaluar las correspondencias entre el sistema y el algoritmo
	void EvalSystem(std::string file_output)
	{
		// Revisar si ambos vectores tienen la misma cantidad de elementos
		if (matched_coord_data_.size() != matched_coord_gt_.size())
		{
			std::cout << "coord data size: " << matched_coord_data_.size() << std::endl;
			std::cout << "coord gt size: " << matched_coord_gt_.size() << std::endl;
			std::cout << "Error: Vectores desiguales" << std::endl;
			return;
		}

		std::vector<double> vec_err(matched_coord_data_.size());

		std::ofstream myfile;
		myfile.open(file_output);
		// Calcular los errores
		for (int i = 0; i < matched_coord_gt_.size(); ++i)
		{
			// Evaluando las Posiciones
			// ------------------------
			vec_err[i] = (matched_coord_data_[i] - matched_coord_gt_[i]).norm();
			myfile << vec_err[i] << " ";

			// Evaluando las Rotaciones
			// ------------------------
			// Error rotacional // Este error no se calcula directamente como en los vectores
		    // Calculado segun https://math.stackexchange.com/questions/2113634/comparing-two-rotation-matrices
		    //std::cout << "matched_rot_gt:" << std::endl << matched_rot_gt_[i] << std::endl;
		    //std::cout << "matched_rot_data:" << std::endl << matched_rot_data_[i] << std::endl;
			Eigen::Matrix3d mat = matched_rot_gt_[i].transpose() * matched_rot_data_[i];
	    	
			double rot_error = mat.trace();

			// Debido a falta de precision, a veces eigen calcula valores para la traza mayores a 3.0, lo que genera inconsistencias en las operaciones posteriores
			if (rot_error > 3.0)
				rot_error = 2.9999;

	   		rot_error = ( rot_error - 1.0 ) / 2.0;
	   		rot_error = acos(rot_error) * 180.0 / M_PI;

	   		myfile << rot_error << "\n";

		}

		myfile.close();

		double sum = 0.0;
		for(auto err : vec_err)
			sum += err;
		sum /= (double) matched_coord_gt_.size();

		std::cout << "Avg Err: " << sum << std::endl;



	}
};

// Esta Variacion solo tomara en cuenta el Algoritmo de Visual Odometry
class MainSystem_A : public MainSystemBase
{
public:
	MainSystem_A(Dataset *data, Settings *settings, DirectOdometryBase *direct_odometry, std::string forest_file_name)
		:MainSystemBase(data, settings, direct_odometry, forest_file_name){};

	void execute();	
};

// Esta variacion sera en Base al Random Forest
class MainSystem_B : public MainSystemBase
{
public:
	MainSystem_B(Dataset *data, Settings *settings, DirectOdometryBase *direct_odometry, std::string forest_file_name)
		:MainSystemBase(data, settings, direct_odometry, forest_file_name){};

	void execute();	
};

// Esta variacion es la primera prueba de tratar de juntar ambos algoritmos
class MainSystem_C : public MainSystemBase
{
public:
	MainSystem_C(Dataset *data, Settings *settings, DirectOdometryBase *direct_odometry, std::string forest_file_name)
		:MainSystemBase(data, settings, direct_odometry, forest_file_name){};

	void execute();	
};

// Descrito en el main.cpp
class MainSystem_D : public MainSystemBase
{
public:
	MainSystem_D(Dataset *data, Settings *settings, DirectOdometryBase *direct_odometry, std::string forest_file_name)
		:MainSystemBase(data, settings, direct_odometry, forest_file_name){};

	void execute();	
};

#endif