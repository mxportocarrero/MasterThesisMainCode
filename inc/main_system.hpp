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

	// Vector para previsualizar la reconstruccion
	// Llenado con los RBM calculados por el sistema
	std::vector<cv::Affine3d> cvAbsPoses;

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
	void displayReconstructionPreview(cv::viz::Viz3d &window){
		std::vector<LabeledPixel> labeled_data;

		// Objeto random para la toma de muestras
		Random *random = new Random();

		std::vector<cv::Point3d> tmp_pos;
		std::vector<cv::Vec3b> tmp_color;
		int num_pixel_per_frame = 2000;
		cv::viz::Viz3d pc("temp pointcloud");
		for (int curr_frame = 0; curr_frame < data_->getNumFrames(); ++curr_frame)
		{
			//std::cout << "frame" << curr_frame << std::endl;
			//std::cout << "Transformacion\n" << cvAbsPoses[curr_frame].matrix << std::endl;
			//std::vector<cv::Point3d> tmp_pos1;
			//std::vector<cv::Vec3b> tmp_color1;
			//cv::Mat controller1 = cv::imread(data_->dataset_path_ + "/" + data_->rgb_filenames_.at(curr_frame));
			//cv::Mat controller2 = data_->getRgbImage(curr_frame);

			cv::Mat depth_image = cv::imread(data_->dataset_path_ + "/" + data_->depth_filenames_.at(curr_frame), cv::IMREAD_ANYDEPTH);
			cv::Mat image = cv::imread(data_->dataset_path_ + "/" + data_->rgb_filenames_.at(curr_frame));

			for (int j = 0; j < num_pixel_per_frame; ++j)
			{
				int row = random->Next(0, settings_->image_height_);
				int col = random->Next(0, settings_->image_width_);

				double Z = (double) depth_image.at<ushort>(row,col);

				if (Z != 0.0)
				{
					Z = Z / (double)settings_->depth_factor_;
					if(Z < 4.0){
					double Y = (row - settings_->cy) * Z / settings_->fy;
					double X = (col - settings_->cx) * Z / settings_->fx;

					cv::Point3d base_point(X,Y,Z);
					cv::Point3d label = cvAbsPoses[curr_frame] * base_point;

					tmp_pos.push_back(label);
					tmp_color.push_back(image.at<cv::Vec3b>( cv::Point2i(col,row) ));
					//tmp_pos1.push_back(label);
					//tmp_color1.push_back(image.at<cv::Vec3b>( cv::Point2i(col,row) ));					
					}
				} else { // Pixel invalido de valor 0
					--j;
				}
			} // Fin de iterar sobre los pixeles

			// Agregando la nube de puntos a la referencia del visualizador
			// ---------------------------
			// pc.showWidget("pc"+std::to_string(curr_frame),cv::viz::WCloud(tmp_pos1,tmp_color1));
			// pc.spinOnce(1,true);
			//pc.spin();

			// cv::imshow("controller1",controller1);
			// cv::imshow("controller2",controller2);
			// cv::waitKey();

		} // Fin de Iterar sobre los frame

		//pc.spin();

		window.showWidget("pointcloud",cv::viz::WCloud(tmp_pos,tmp_color));
		window.spin();

	} // Fin de la funcion displayReconstructionPreview

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
	   		rot_error = acos(rot_error) * 180.0 / M_PI; // conversion from radians to degrees

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