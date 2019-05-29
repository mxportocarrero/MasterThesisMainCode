#include "../inc/forest.hpp"

//#define PRINT_CREATE_HYPOTHESIS_INFO
//#define PRINT_HYPOTHESIS_OPTIMIZATION_INFO

// Other Functions
uint tophat_error(double val)
{
	return !(val > 0.0 && val < 0.1);
}

std::vector<cv::Point3d> getFrameCoordPairs(Pose pose, Settings *settings)
{
	int width = 640, height = 480;

	std::vector<cv::Point3d> pairs;
	std::vector<cv::Point2i> tmp_cv;

	// Establishing the corners (Img Space)
	tmp_cv.push_back(cv::Point2i(0,0));
	tmp_cv.push_back(cv::Point2i(width,0));
	tmp_cv.push_back(cv::Point2i(0,height));
	tmp_cv.push_back(cv::Point2i(width,height));

	double Z = 0.3; // Depth 0.3 meters

	// Tranforming to Camera Space
	// then using pose info tranforming to World Space
	for(auto coord : tmp_cv){
		double X = (coord.x - settings->cx) * Z / settings->fx;
		double Y = (coord.y - settings->cy) * Z / settings->fy;

		Eigen::Vector3d p_camera(X, Y, Z);
		Eigen::Vector3d label_e = poseRotation(pose) * p_camera + posePosition(pose);
		cv::Point3d label(label_e.x(), label_e.y(), label_e.z());
		
		Eigen::Vector3d c(0.0,0.0,0.0);
		Eigen::Vector3d label_c = poseRotation(pose) * c + posePosition(pose);
		cv::Point3d label_t(label_c.x(), label_c.y(), label_c.z());

		pairs.push_back(label_t);
		pairs.push_back(label);
	}

	// 0,0 -> w,0
	pairs.push_back(pairs[1]);
	pairs.push_back(pairs[3]);

	// 0,0 -> 0,h
	pairs.push_back(pairs[1]);
	pairs.push_back(pairs[5]);

	// w,h -> w,0
	pairs.push_back(pairs[7]);
	pairs.push_back(pairs[3]);

	// w,h -> 0,h
	pairs.push_back(pairs[7]);
	pairs.push_back(pairs[5]);

	return pairs;	
} // Fin de la Funcion

// Initializing empty forest
template<typename D, typename RGB>
Forest<D,RGB>::Forest(Dataset *data, Settings *settings)
	:data_(data), settings_(settings)
{
	for (int i = 0; i < settings_->num_trees_; ++i)
	{
		forest_.push_back(new Tree<D,RGB>());
	}

	random_ = new Random();
}

template<typename D, typename RGB>
Forest<D,RGB>::Forest(Dataset *data, Settings *settings, const std::string &features_path, int i)
	:data_(data), settings_(settings)
{
	for (int i = 0; i < settings_->num_trees_; ++i)
	{
		forest_.push_back(new Tree<D,RGB>());
	}

	random_ = new Random();

	candidate_features_ = this->read_candidate_features(features_path);
}

// Initializaing from binary file
template<typename D, typename RGB>
Forest<D,RGB>::Forest(Dataset *data, Settings *settings, const std::string &path)
	:data_(data), settings_(settings)
{
	random_ = new Random();

	std::ifstream i(path, std::ios_base::binary);
	if (!i.is_open())
	{
		throw std::runtime_error("No se puede abrir el archivo");
	}

	// Deserializando los arboles
	std::vector<char> buffer(strlen(binaryFileHeader_) + 1);
	i.read(&buffer[0], strlen(binaryFileHeader_));
	buffer[ buffer.size()-1 ] = '\0';

	if ( strcmp(&buffer[0], binaryFileHeader_) != 0 )
		throw std::runtime_error("Formato de arbol no soportado");

	const int majorVersion = 0, minorVersion =0;
	i.read((char*)(&majorVersion), sizeof(majorVersion));
	i.read((char*)(&minorVersion), sizeof(minorVersion));

	int treeCount = 0;
	i.read((char*)(&treeCount), sizeof(treeCount)); // Leemos la cantidad de arboles

	for (int j = 0; j < treeCount; ++j)
	{
		std::cout << "Leyendo arbol " << j << std::endl;

		// Deserializando el arbol
		Tree<D,RGB> *t = new Tree<D,RGB>();
		t->Deserialize(i,settings_);
		forest_.push_back(t);

		//forest_[j]->printBTree("",forest_[j]->getRoot(),false);

	} // Fin de Leer los arboles

	// Validating Forest
	std::cout << "Validating forest\n";
	if (this->IsValid())
		std::cout << "Forest is Valid" << std::endl;
	else
		throw "Forest is NOT Valid";

} // Fin del Constructora partir


// Funcion para validar los arboles
template<typename D, typename RGB>
bool Forest<D,RGB>::IsValid()
{
	bool ret = true;
	for (auto tree : forest_) {
		ret = ret && tree->IsValid();
	}

	return ret;
}

// ------------------
// TRAINING FUNCTIONS
// ------------------

template<typename D, typename RGB>
std::vector< DepthAdaptiveRGB<D, RGB> > Forest<D,RGB>::read_candidate_features(const std::string &file)
{
	std::vector< DepthAdaptiveRGB<D, RGB> > candidate_features;
	std::ifstream myfile(file);

	std::string line;
	if (myfile.is_open())
	{
		std::vector<std::string> v(7);
		int cont = 0;
		while ( myfile >> line )
		{
			v[cont] = line;
			cont++;

			// terminamos de leer 7 datos consecutivos
			if(cont == 7)
			{
				DepthAdaptiveRGB<D,RGB> feature(cv::Point2i(std::stoi(v[0]),std::stoi(v[1])),
												cv::Point2i(std::stoi(v[2]),std::stoi(v[3])),
												std::stoi(v[4]),
												std::stoi(v[5]),
												std::stof(v[6]));

				candidate_features.push_back(feature);

				cont = 0;
			}
		}

		myfile.close();
	}

	return candidate_features;
} // Fin de read_candidate_features

template<typename D, typename RGB>
std::vector<LabeledPixel> Forest<D,RGB>::VerifyLabelData()
{
	std::vector<LabeledPixel> labeled_data;

	std::vector<cv::Point3d> tmp_pos;
	std::vector<cv::Vec3b> tmp_color;
	std::vector<cv::Point3d> cam_coords;
	cv::viz::Viz3d window("Coordinate Frame");

	// randomly choose frames
	//for (int i = 0; i < settings_->num_frames_per_tree_; ++i)
	for (int i = 0; i < data_->getNumFrames(); i+=5)
	{
		// Activar desactivar las siguientes lineas de codigo para usar el 7 scenes
		//------------------------------------------------------
		//int curr_frame = random_->Next(0, data_->getNumFrames());
		int curr_frame = i;
		//------------------------------------------------------
		/*
		int seq = (data_->train_sequences[ random_->Next(0,data_->train_sequences.size()) ] - 1) * 1000;
		int curr_frame = random_->Next(0,1000) + seq;
		if (3920 < curr_frame && curr_frame < 3950)
		{
			//std::cout << "?" << curr_frame << std::endl;
			--i;
			continue;
		}
		*/
		//std::cout << "testing frame: " << curr_frame << std::endl;

		///*
		cv::Mat depth_image = data_->getDepthImage(curr_frame); // Conseguimos la imagen
		Pose pose = data_->getPose(curr_frame); // Conseguimos el Groundtruth pose

		int invalid_pixels = 0; std::vector<LabeledPixel> labeled_data_tmp;
		for (int j = 0; j < settings_->num_pixels_per_frame_; ++j) {
			//int row = random_->Next(0, settings_->image_height_);
			//int col = random_->Next(0, settings_->image_width_);

			int row = random_->Next(0, settings_->image_height_);
			int col = random_->Next(0, settings_->image_width_ );

			double Z = (double)depth_image.at<D>(row, col);
			//std::cout << Z << std:: endl;

			if (Z != 0.0)
			{
				Z = Z / (double)settings_->depth_factor_;

				double Y = (row - settings_->cy) * Z / settings_->fy;
				double X = (col - settings_->cx) * Z / settings_->fx;

				Eigen::Vector3d p_camera(X, Y, Z);
				Eigen::Vector3d label_e = poseRotation(pose) * p_camera + posePosition(pose);

				/* // Activar para evaluar el 7 scenes
				// Limitamos los pixeles para que esten dentro de un area de interes
				if ((-1.5 < label_e.x() && label_e.x() < 1.5) &&
					(-1.5 < label_e.y() && label_e.y() < 1.5) &&
					( 0.0 < label_e.z() && label_e.z() < 3.0) )
				{
					cv::Point3d label(label_e.x(), label_e.y(), label_e.z());

					// store labeled pixel
					LabeledPixel pixel(curr_frame, cv::Point2i(col, row), label);
					labeled_data_tmp.push_back(pixel);
				}
				else
					--j;
				*/

				
				///* // Algunos limites fr1_room
				// Limitamos los pixeles para que esten dentro de un area de interes
				if ((-2.7 < label_e.x() && label_e.x() < 2.8) &&
					(-3.4 < label_e.y() && label_e.y() < 3.0) &&
					( 0.0 < label_e.z() && label_e.z() < 3.5) )
				{
					cv::Point3d label(label_e.x(), label_e.y(), label_e.z());

					// store labeled pixel
					LabeledPixel pixel(curr_frame, cv::Point2i(col, row), label);
					labeled_data_tmp.push_back(pixel);
				}
				else
					--j;
				//*/

				/*
				cv::Point3d label(label_e.x(), label_e.y(), label_e.z());

				// store labeled pixel
				LabeledPixel pixel(curr_frame, cv::Point2i(col, row), label);
				labeled_data_tmp.push_back(pixel);
				*/

			}
			// Filtramos los pixeles invalidos, por que igual no aportaran
			// Y nos apoyamos para filtrar frames que tienen gran cantidad de pixeles invalidos
			else
			{
				--j;
				++invalid_pixels;
			}

			if (invalid_pixels > 10000)
			{
				break;
			}

		} // Fin de FOR de insertar todos los pixeles

		if (invalid_pixels > 10000) {
			--i;
		} else {
			// Copiar los labeled pixeles temporales al vector
			//std::vector<cv::Point3d> tmp_pos;
			for (auto pixel : labeled_data_tmp){
				labeled_data.push_back(pixel);

				tmp_pos.push_back(pixel.label_);
				tmp_color.push_back(data_->getRgbImage(pixel.frame_).at<cv::Vec3b>(pixel.pos_));
			}

			///*
			// mostrando los pixeles seleccionados
			std::cout << "Frame " << curr_frame << std::endl;
			std::cout << "timestamp " <<  data_->getTimestamp(curr_frame) << std::endl;

			cv::Mat m = data_->getRgbImage(curr_frame).clone();
			cv::namedWindow("Display RGB",cv::WINDOW_AUTOSIZE);
		    cv::namedWindow("Display Depth",cv::WINDOW_AUTOSIZE);

		    cv::imshow("Display RGB",m);
			show_depth_image("Display Depth", data_->getDepthImage(curr_frame));

			cv::waitKey(10);


			printMat44(pose,"groundtruth pose");

			//cv::viz::Viz3d window("Coordinate Frame");
			window.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());
			if (tmp_pos.size() > 0)
				window.showWidget("points", cv::viz::WCloud(tmp_pos, tmp_color));

			// Construct a cube widget
		    cv::viz::WCube cube_widget(cv::Point3f(1.5,1.5,3.0), cv::Point3f(-1.5,-1.5,0.0), true, cv::viz::Color::blue());
		    cube_widget.setRenderingProperty(cv::viz::LINE_WIDTH, 4.0);
		    window.showWidget("Cube Widget", cube_widget);


		    // Displaying Camera Pose
		    std::vector<cv::Point3d> cam_coords_tmp = getFrameCoordPairs(pose,settings_);
		    std::vector<cv::viz::WLine> vec_lines;
		    for (int k = 0; k < cam_coords_tmp.size(); k += 2)
		    {
		    	vec_lines.push_back( cv::viz::WLine( cam_coords_tmp[k],cam_coords_tmp[k+1],cv::viz::Color::red() ) );
		    	vec_lines[vec_lines.size()-1].setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
		    	window.showWidget("a" + std::to_string(k),vec_lines[vec_lines.size()-1]);
		    }

		    // Displaying camera path
		    cam_coords.push_back(cam_coords_tmp[0]);

		    if (cam_coords.size() > 1)
		    {
		    	for (int k = 1; k < cam_coords.size(); ++k)
		    	{
		    		cv::viz::WSphere s(cam_coords[k],0.005,10,cv::viz::Color::blue());
		    		cv::viz::WLine l( cam_coords[k],cam_coords[k-1],cv::viz::Color::red() );
		    		l.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
		    		window.showWidget("s"+std::to_string(k),s);
		    		window.showWidget("b"+std::to_string(k),l);
		    	}
		    }

			window.spinOnce(100);
			//window.spin();
			//*/
		}
		//*/
	} // Fin de Primer FOR
	return labeled_data;
} // Fin de LabelData

template<typename D, typename RGB>
std::vector<LabeledPixel> Forest<D,RGB>::LabelData()
{
	std::vector<LabeledPixel> labeled_data;

	// randomly choose frames
	for (int i = 0; i < settings_->num_frames_per_tree_; ++i)
	{
		// Activar desactivar las siguientes lineas de codigo para usar el 7 scenes
		//------------------------------------------------------
		int curr_frame = random_->Next(0, data_->getNumFrames());
		//------------------------------------------------------
		/*
		int seq = (data_->train_sequences[ random_->Next(0,data_->train_sequences.size()) ] - 1) * 1000;
		int curr_frame = random_->Next(0,1000) + seq;
		if (3920 < curr_frame && curr_frame < 3950)
		{
			//std::cout << "?" << curr_frame << std::endl;
			--i;
			continue;
		}
		*/
		//std::cout << "testing frame: " << curr_frame << std::endl;

		cv::Mat depth_image = data_->getDepthImage(curr_frame); // Conseguimos la imagen
		Pose pose = data_->getPose(curr_frame); // Conseguimos el Groundtruth pose

		int invalid_pixels = 0; std::vector<LabeledPixel> labeled_data_tmp;
		for (int j = 0; j < settings_->num_pixels_per_frame_; ++j) {
			//int row = random_->Next(0, settings_->image_height_);
			//int col = random_->Next(0, settings_->image_width_);

			int row = random_->Next(0, settings_->image_height_);
			int col = random_->Next(0, settings_->image_width_ );

			double Z = (double)depth_image.at<D>(row, col);
			//std::cout << Z << std:: endl;

			if (Z != 0.0)
			{
				Z = Z / (double)settings_->depth_factor_;

				double Y = (row - settings_->cy) * Z / settings_->fy;
				double X = (col - settings_->cx) * Z / settings_->fx;

				Eigen::Vector3d p_camera(X, Y, Z);
				Eigen::Vector3d label_e = poseRotation(pose) * p_camera + posePosition(pose);

				/* // Activar para evaluar el 7 scenes (chess)
				// Limitamos los pixeles para que esten dentro de un area de interes
				if ((-1.5 < label_e.x() && label_e.x() < 1.5) &&
					(-1.5 < label_e.y() && label_e.y() < 1.5) &&
					( 0.0 < label_e.z() && label_e.z() < 3.0) )
				{
					cv::Point3d label(label_e.x(), label_e.y(), label_e.z());

					// store labeled pixel
					LabeledPixel pixel(curr_frame, cv::Point2i(col, row), label);
					labeled_data_tmp.push_back(pixel);
				}
				else
					--j;
				*/

				
				///* // Algunos limites
				// Limitamos los pixeles para que esten dentro de un area de interes
				if ((-2.7 < label_e.x() && label_e.x() < 2.8) &&
					(-3.4 < label_e.y() && label_e.y() < 3.0) &&
					( 0.0 < label_e.z() && label_e.z() < 3.5) )
				{
					cv::Point3d label(label_e.x(), label_e.y(), label_e.z());

					// store labeled pixel
					LabeledPixel pixel(curr_frame, cv::Point2i(col, row), label);
					labeled_data_tmp.push_back(pixel);
				}
				else
					--j;
				//*/

				/*
				cv::Point3d label(label_e.x(), label_e.y(), label_e.z());

				// store labeled pixel
				LabeledPixel pixel(curr_frame, cv::Point2i(col, row), label);
				labeled_data_tmp.push_back(pixel);
				*/

			}
			// Filtramos los pixeles invalidos, por que igual no aportaran
			// Y nos apoyamos para filtrar frames que tienen gran cantidad de pixeles invalidos
			else
			{
				--j;
				++invalid_pixels;
			}

			if (invalid_pixels > 10000)
			{
				break;
			}

		} // Fin de FOR de insertar todos los pixeles

		if (invalid_pixels > 10000) {
			--i;
		} else {
			// Copiar los labeled pixeles temporales al vector
			for (auto pixel : labeled_data_tmp)
				labeled_data.push_back(pixel);

		}
	} // Fin de Primer FOR
	return labeled_data;
} // Fin de LabelData


// Training Function
template<typename D, typename RGB>
void Forest<D,RGB>::Train(const std::string &path)
{
	// Imprimir Info importante del dataset
	std::cout << "RGBD pairs: " << data_->timestamp_rgbd_.size() << std::endl;
	std::cout << "Groundtruth ticks: " << data_->timestamp_groundtruth_.size() << std::endl;
	std::cout << "Valid RGBD pairs with Gt: " << data_->getNumFrames() << std::endl;
	int index = 0;

	for(auto tree : forest_)
	{
		std::cout << "[Tree " << index << "] " << "Generating Training Data.\n";
		VerifyLabelData();
		std::vector<LabeledPixel> labeled_data = LabelData();
		std::cout << "LabelData Size: " << labeled_data.size() << std::endl;

		///*
		// Parte para visualizar los pixeles ---------------------
		std::vector<cv::Point3d> tmp_pos;
		std::vector<cv::Vec3b> tmp_color;
		for(auto labeled_pixel : labeled_data){
			tmp_pos.push_back(labeled_pixel.label_);
			tmp_color.push_back(data_->getRgbImage(labeled_pixel.frame_).at<cv::Vec3b>(labeled_pixel.pos_));
		}

		cv::viz::Viz3d window("Coordinate Frame");
		window.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());
		//window.showWidget("points", cv::viz::WCloud(tmp_pos, cv::viz::Color::green()));
		window.showWidget("points", cv::viz::WCloud(tmp_pos, tmp_color));
		/// Construct a cube widget
	    cv::viz::WCube cube_widget(cv::Point3f(2.8,3.0,3.5), cv::Point3f(-2.7,-3.4,0.0), true, cv::viz::Color::blue());
	    cube_widget.setRenderingProperty(cv::viz::LINE_WIDTH, 4.0);
	    window.showWidget("Cube Widget", cube_widget);

		window.spin();
		// ---------------------
		//*/

		// train tree with set of pixels recursively
		std::cout << "[Tree " << index << "] " << "Training.\n";

		std::clock_t start;
		double duration;
		start = std::clock();

		tree->Train(data_, labeled_data, random_, settings_, candidate_features_);

		duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

		// Desactivar la opcion para no imprimir el arbol
		//tree->printBTree("",tree->getRoot(),false);

		std::cout << "[Tree " << index << "] "
		        << "Train time: " << duration << " Seconds \n";
		index++;
	}

	// Serializando los arboles
	std::ofstream o(path, std::ios::binary);

	const int majorVersion = 0, minorVersion =0;

	o.write(binaryFileHeader_, strlen(binaryFileHeader_));
	o.write((const char*)(&majorVersion), sizeof(majorVersion));
	o.write((const char*)(&minorVersion), sizeof(minorVersion));

	int treeCount = settings_->num_trees_;
	o.write((const char*)(&treeCount), sizeof(treeCount));
	std::cout << "Serializing\n";

	for(auto tree: forest_)
		tree->Serialize(o);

	if (o.bad())
	{
		throw std::runtime_error("Forest serialization failed");
	}

	// Validating Forest
	std::cout << "Validating forest\n";
	if (this->IsValid())
		std::cout << "Forest is Valid" << std::endl;
	else
		throw "Forest is NOT Valid";

} // Fin de la funcion Train

// ------------------
// TESTING FUNCTIONS
// ------------------
template<typename D, typename RGB>
std::vector<Hypothesis> Forest<D,RGB>::CreateHypothesis(int K_init, cv::Mat rgb_frame, cv::Mat depth_frame)
{
	std::vector<Hypothesis> hypotheses;

	// Extraemos K_init hipotesis iniciales
	for (int i = 0; i < K_init; ++i)
	{
		Hypothesis h;
		// 3 points
		h.input_.resize(3, 3);
		h.output_.resize(3, 3);

		for (int j = 0; j < 3; ++j)
		{
			/* code */
		}


	}
}


template<typename D, typename RGB>
std::vector<Eigen::Vector3d> Forest<D,RGB>::Eval(int col, int row, cv::Mat rgb_image, cv::Mat depth_image)
{
	std::vector<Eigen::Vector3d> modes;

	int cont = 0;
	for (auto t : forest_) {
		//std::cout << "Arbol " << cont << std::endl;
		bool valid = true;
		Eigen::Vector3d mode = t->Eval(col, row, rgb_image, depth_image, valid);
		if (valid){
			//std::cout << "Tree " << cont << " , estimated mode: " << mode(0) << "," << mode(1) << "," << mode(2) << std::endl;

			modes.push_back(mode);
		} // Si el pixel fue correctamente validado

		cont++;
	}

	if (modes.empty())
	{
		//std::cout << "No se pudo evaluar el pixel en el Arbol\n";
	}

	return modes;
} // Fin de la Funcion

template<typename D, typename RGB>
void Forest<D,RGB>::Test()
{
	int correct_predictions = 0;
	for (int frame = 0, cont = 1; frame < 1000; frame += 1, ++cont)
	{
		// Seleccionamos un frame al azar
		//int frame = random_->Next(0,1000);
		//int frame = 72;
		std::cout << "Evaluating frame: " << frame << std::endl;

		// Calculating groundtruth pixel coordinates
		Pose pose = data_->getPose(frame);

		///**
		std::vector<Hypothesis> hypotheses;

#ifdef PRINT_CREATE_HYPOTHESIS_INFO
		std::cout << "Creando Hipotesis\n";
		std::cout << "==================================================\n";
#endif

		// Extraemos K_init hipotesis iniciales
		int K = 1024;
		int offset = 50;
		for (int i = 0; i < K; ++i)
		{
			Hypothesis h;
			// 3 points
			h.input_.resize(3, 3);
			h.output_.resize(3, 3);

			int no_inliers = 0, no_outliers = 0;
			for (int j = 0; j < 3; ++j)
			{
				int col = random_->Next( 0 + offset, settings_->image_width_ - offset);
				int row = random_->Next( 0 + offset, settings_->image_height_ - offset);
				//int col = random_->Next( 0 , settings_->image_width_);
				//int row = random_->Next( 0 , settings_->image_height_);

				ushort depth = data_->getDepthImage(frame).at<ushort>(cv::Point2i(col,row));
#ifdef PRINT_CREATE_HYPOTHESIS_INFO
				std::cout << "depth: " << depth << std::endl;
#endif
				auto modes = this->Eval(col, row, data_->getRgbImage(frame), data_->getDepthImage(frame));

				if (depth == 0 || modes.empty())
				{
					j--;
				}
				else
				{
					// Calculamos las coordenadas en Camera Space
					// ------------------------------------------
					double X,Y,Z;
					Z = (double)depth / (double)settings_->depth_factor_;

					X = (col - settings_->cx) * Z / settings_->fx;
					Y = (row - settings_->cy) * Z / settings_->fy;

					Eigen::Vector3d p_camera(X,Y,Z); // Declaramos un vector temporal

					//h.camera_space_point = p_camera; // No se si esto sea necesario
					// Agregamos un punto para el input!
					h.input_.col(j) = p_camera;

					// Calculamos las coordenadas en World Space
					// ------------------------------------------

					// Agregamos el punto del camera space a las propiedad de la hipotesis
					Eigen::Vector3d selected_mode;
					if (modes.size() > 1)
						selected_mode = modes.at(random_->Next(0, modes.size()));
					else
						selected_mode = modes.front();

					// Agregamos un punto para el output!
					h.output_.col(j) = selected_mode;

#ifdef PRINT_CREATE_HYPOTHESIS_INFO
					// Verificando si se empleo algun inlier en las hipotesis
					// ------------------------------------------------------
					std::cout << "==================================================\n";
					//std::cout << "Camera space Coordinate: " << p_camera(0) << "," << p_camera(1) << "," << p_camera(2) << std::endl;

					Eigen::Vector3d world_coord = poseRotation(pose) * p_camera + posePosition(pose);
					std::cout << "World Coord Coordinate: " << world_coord(0) << "," << world_coord(1) << "," << world_coord(2) << std::endl;


					Eigen::Vector3d e = selected_mode - world_coord;
					double e_norm = e.norm();

					if (e_norm < 0.2)
					{
						no_inliers++;
						std::cout << "Inlier, estimated mode: " << selected_mode(0) << "," << selected_mode(1) << "," << selected_mode(2) << std::endl;
					}
					else
					{
						no_outliers++;
						std::cout << "Outlier, estimated mode: " << selected_mode(0) << "," << selected_mode(1) << "," << selected_mode(2) << std::endl;
					}
					std::cout << "error " << e_norm << std::endl;
					std::cout << "==================================================\n";
#endif

				} // Fin de la Condicional de pixeles invalidos				

			} // Terminando de recolector los tres puntos iniciales

			// Usando el algoritmo de Kabsch y refistrando la Hipotesis
			// --------------------------------------------------------
			Eigen::Affine3d transform = Find3DAffineTransform(h.input_, h.output_);
			h.pose_ = transform;
			h.energy_ = 0;
			hypotheses.push_back(h);


			
			// Imprimiendo info importante
			// ---------------------------
			std::cout << "==================================================\n";
			std::cout << "frame "<< frame  << " hyp: " << i << std::endl;
			std::cout << "==================================================\n";
			printMat44(pose,"groundtruth pose");

#ifdef PRINT_CREATE_HYPOTHESIS_INFO
			Eigen::IOFormat stdFormat(4);
			std::cout << "Estimated Pose:\n" << h.pose_.matrix().format(stdFormat) << std::endl;

			std::cout << "No Inliers en la hipotesis: " << no_inliers << std::endl;

			Eigen::Vector3d rot_gt = poseRotation(pose).eulerAngles(0, 1, 2) * 180.0 / M_PI;
			//std::cout << "rot_gt" << std::endl << rot_gt << std::endl;
			Eigen::Vector3d pos_gt = posePosition(pose);

			Eigen::Vector3d rot_eval = h.pose_.rotation().eulerAngles(0, 1, 2) * 180.0 / M_PI;
			//std::cout << "rot_eval" << std::endl << rot_eval << std::endl;
		    Eigen::Vector3d pos_eval = h.pose_.translation();

		    Eigen::Vector3d verror; double error_pos;

		    // Error rotacional // Este error no se calcula directamente como en los vectores
		    // Calculado segun https://math.stackexchange.com/questions/2113634/comparing-two-rotation-matrices
		    double rot_error = acos(((poseRotation(pose).transpose() * h.pose_.rotation()).trace() - 1.0) / 2.0) * 180.0 / M_PI;

		    // Error traslacional
		    verror = pos_eval - pos_gt;
		    error_pos = verror.norm();

		    printf("Rotational Error: %.4f Traslational Error: %.4f\n", rot_error, error_pos );

			std::cout << "==================================================\n";

			// Si el error traslacional es menor a 20 cm paramos para revisarlo
			cv::Mat m = data_->getRgbImage(frame).clone();
			cv::namedWindow("Display RGB",cv::WINDOW_AUTOSIZE);
		    cv::namedWindow("Display Depth",cv::WINDOW_AUTOSIZE);

		    cv::imshow("Display RGB",m);
			show_depth_image("Display Depth", data_->getDepthImage(frame));

			cv::waitKey();
#endif

		} // Fin de Calcular K = 1024 hipotesis

		//**/



		///***
		// Hypothesis Optimization
		// ------------------------

		while( K > 1)
		{
			std::vector<cv::Point2i> test_pixels;
			int batch_size = 500;
			for (int i = 0; i < batch_size; ++i)
			{
				int col = random_->Next( 0 + offset, settings_->image_width_ - offset);
				int row = random_->Next( 0 + offset, settings_->image_height_ - offset);
				//int col = random_->Next( 0 , settings_->image_width_);
				//int row = random_->Next( 0 , settings_->image_height_);

				ushort depth = data_->getDepthImage(frame).at<ushort>(cv::Point2i(col,row));

				// Filtramos los pixeles con depth no validos
				if (depth == 0)
					--i;
				else
					test_pixels.push_back(cv::Point2i(col, row));
			} // Fin de Samplear el Batch Size

			// Evaluamos el Batch de los pixeles
			// ---------------------------------
			for(auto p : test_pixels)
			{
				auto modes = this->Eval(p.x, p.y, data_->getRgbImage(frame), data_->getDepthImage(frame));

				if (!modes.empty())
				{
					D test = data_->getDepthImage(frame).at<D>(p);
					double Z = (double)test / (double)settings_->depth_factor_;
					double X = (p.x - settings_->cx) * Z / settings_->fx;
					double Y = (p.y - settings_->cy) * Z / settings_->fy;

					Eigen::Vector3d p_camera(X, Y, Z);
					//std::cout << "p_camera " << p_camera(0) << "," << p_camera(1) << "," << p_camera(2) << std::endl;

					for (int i = 0; i < K; ++i)
					{
						// Estimacion de la hipotesis
						Eigen::Vector3d h_pcamera = hypotheses.at(i).pose_ * p_camera;

#ifdef PRINT_HYPOTHESIS_OPTIMIZATION_INFO
						Eigen::IOFormat stdFormat(4);
						printMat44(pose,"groundtruth pose");
						std::cout << "Hypothesis:\n" << hypotheses.at(i).pose_.matrix().format(stdFormat) << std::endl;
						std::cout << "hypothesis coord estimation " << h_pcamera(0) << "," << h_pcamera(1) << "," << h_pcamera(2) << std::endl;
#endif //------------------------------//

						double e_min = DBL_MAX; // Valor maximo del double
						Eigen::Vector3d best_mode;
						Eigen::Vector3d best_camera_p;

						for (auto mode : modes) {
							Eigen::Vector3d e = mode - h_pcamera;
							double e_norm = e.norm();

							if (e_norm < e_min) {
#ifdef PRINT_HYPOTHESIS_OPTIMIZATION_INFO
								std::cout << "mode " << mode(0) << "," << mode(1) << "," << mode(2) << " error " << e_norm << std::endl;
#endif
								e_min = e_norm;
								best_mode = mode;
								best_camera_p = p_camera;
							}
						} // Fin del bucle de las modas
#ifdef PRINT_HYPOTHESIS_OPTIMIZATION_INFO
						std::cout << "==================================================\n";
#endif


						// update energy
						hypotheses.at(i).energy_ += tophat_error(e_min);

						///*
						// inlier
						if (tophat_error(e_min) == 0) {
							//std::cout << "*************Inlier!!\n";
							// add to kabsch matrices
							hypotheses.at(i).input_.conservativeResize(3, hypotheses.at(i).input_.cols() + 1);
							hypotheses.at(i).input_.col(hypotheses.at(i).input_.cols() - 1) = best_camera_p;

							hypotheses.at(i).output_.conservativeResize(3, hypotheses.at(i).output_.cols() + 1);
							hypotheses.at(i).output_.col(hypotheses.at(i).output_.cols() - 1) = best_mode;
						} // Fin del Condicional

					} // Revisando todas las hipotesis

					std::cout << "==================================================\n";

					// pequeña pausa
					//cv::Mat m = data_->getRgbImage(frame).clone();
					//cv::namedWindow("Display RGB",cv::WINDOW_AUTOSIZE);
				    //cv::namedWindow("Display Depth",cv::WINDOW_AUTOSIZE);

				    //cv::imshow("Display RGB",m);
					//show_depth_image("Display Depth", data_->getDepthImage(frame));

					//cv::waitKey();

				} // Fin de condicional de revisar si hay modas para evaluar

			} // Fin de Testear todo un batch!

			// sort hypotheses (de forma ascendente)
			std::sort(hypotheses.begin(), hypotheses.begin() + K); // Solo ordena los K primeras hipotesis

		    K = K / 2; // discard half

		    // Refinando las hipotesis
	    	for (int i = 0; i < K; ++i){
		    	// Printing ordered energies
		    	std::cout << hypotheses.at(i).energy_ << ", ";
				hypotheses.at(i).pose_ = Find3DAffineTransform(hypotheses.at(i).input_, hypotheses.at(i).output_);

				// Resetting hypothesis energies
		    	hypotheses.at(i).energy_ = 0;

		    	// Resetting inliers
		    	hypotheses.at(i).input_.resize(3, 0);
		    	hypotheses.at(i).output_.resize(3, 0);
	    	} std::cout << std::endl;
	    	std::cout << "==================================================\n";
			std::cout << "frame "<< frame << std::endl;
			std::cout << "==================================================\n";
			printMat44(pose,"groundtruth pose");

			std::cout << "Best Hypothesis\n";

			Eigen::Affine3d estimated_pose = hypotheses.front().pose_;
			Eigen::IOFormat stdFormat(4);
			
			std::cout << "Estimated Pose:\n" << estimated_pose.matrix().format(stdFormat) << std::endl;

			Eigen::Vector3d rot_gt = poseRotation(pose).eulerAngles(0, 1, 2) * 180.0 / M_PI;
			//std::cout << "rot_gt" << std::endl << rot_gt << std::endl;
			Eigen::Vector3d pos_gt = posePosition(pose);

			Eigen::Vector3d rot_eval = estimated_pose.rotation().eulerAngles(0, 1, 2) * 180.0 / M_PI;
			//std::cout << "rot_eval" << std::endl << rot_eval << std::endl;
		    Eigen::Vector3d pos_eval = estimated_pose.translation();

		    Eigen::Vector3d verror; double error_pos;

		    // Error rotacional // Este error no se calcula directamente como en los vectores
		    // Calculado segun https://math.stackexchange.com/questions/2113634/comparing-two-rotation-matrices
		    Eigen::Matrix3d mat = poseRotation(pose).transpose() * estimated_pose.rotation();
			Eigen::IOFormat std2Format(8);
		    std::cout << "mat: "  << std::endl << mat.format(std2Format) << std::endl;
		    double rot_error = ( poseRotation(pose).transpose() * estimated_pose.rotation() ).trace();
		    		// Debido a falta de precision, a veces eigen calcula valores para la traza mayores a 3.0, lo que genera inconsistencias en las operaciones posteriores
		    		if (rot_error > 3.0)
		    			rot_error = 2.9999;
		    printf("Rotational Error: %.8f\n", rot_error);
		    		rot_error = ( rot_error - 1.0 ) / 2.0;
		    printf("Rotational Error: %.8f\n", rot_error);
		    		rot_error = acos(rot_error) * 180.0 / M_PI;
		    //printf("Rotational Error: %.8f\n", rot_error);

		    // Error traslacional
		    verror = pos_eval - pos_gt;
		    error_pos = verror.norm();

		    printf("Rotational Error: %.4f Traslational Error: %.4f\n", rot_error, error_pos );

		    if (K == 1 && (error_pos < 0.05 && rot_error < 5.0))
		    {
		    	correct_predictions++;
		    }

		    if(correct_predictions == 0)
		    {
		    	std::cout << "no correct predictions could be made\n" << std::endl;
		    }
		    else
		    {
		    	std::cout << "Accuracy (%) : " << (float)correct_predictions / (float)(cont) * 100.0 << std::endl;
		    }

			std::cout << "==================================================\n";

			// pequeña pausa
			cv::Mat m = data_->getRgbImage(frame).clone();
			cv::namedWindow("Display RGB",cv::WINDOW_AUTOSIZE);
		    cv::namedWindow("Display Depth",cv::WINDOW_AUTOSIZE);

		    for(auto point : test_pixels){
		    	cv::circle(m,point,5,cv::Scalar(255,255,255));
		    }

		    cv::imshow("Display RGB",m);
			show_depth_image("Display Depth", data_->getDepthImage(frame));

			cv::waitKey();

		} // Fin de refinar las hipotesis

		//***/

		/****
		// Evaluando las hipotesis

		int cont_ = 0, depth_nan = 0, mode_nan = 0, inlier = 0, outlier= 0;	
		do{
			// Testing different pixels
			int col = random_->Next( 0 + 100, settings_->image_width_ - 100);
			int row = random_->Next( 0 + 100, settings_->image_height_ -100);

			// Evaluando la profundidad
			ushort depth = data_->getDepthImage(frame).at<ushort>(cv::Point2i(col,row));
			std::cout << "depth: " << depth << std::endl;

			if (depth == 0)
				depth_nan++;
			else // Valid pixel depth
			{
				double X,Y,Z; // Valores a calcularse en camera space
				// Calculando la posicion en camera space
				Z = (double)depth / (double)settings_->depth_factor_;

				X = (col - settings_->cx) * Z / settings_->fx;
				Y = (row - settings_->cy) * Z / settings_->fy;

				Eigen::Vector3d p_camera(X,Y,Z); // Declaramos un vector temporal

				// Agregamos el punto del camera space a las propiedad de la hipotesis
				std::cout << "Camera space Coordinate: " << p_camera(0) << "," << p_camera(1) << "," << p_camera(2) << std::endl;

				// Calculating groundtruth pixel coordinates
				Pose pose = data_->getPose(frame);
				printMat44(pose,"groundtruth pose");

				Eigen::Vector3d world_coord = poseRotation(pose) * p_camera + posePosition(pose);
				std::cout << "World Coord Coordinate: " << world_coord(0) << "," << world_coord(1) << "," << world_coord(2) << std::endl << std::endl;

				// La siguiente funcion tambien evalua que el pixel selecionado tenga un valor valido de profundidad
				// la variable modes tiene la siguiente firma std::vector<Eigen::Vector3d>
				auto modes = this->Eval(col, row, data_->getRgbImage(frame),data_->getDepthImage(frame));

				std::vector<cv::Point3d> tmp_vec;
				if( modes.size() == 0 )
					mode_nan++;
				else
				{
					Eigen::Vector3d best_mode;
					double e_min = DBL_MAX;
					for(auto mode:modes)
					{
						Eigen::Vector3d e = mode - world_coord;
						double e_norm = e.norm();

						std::cout << "error " << e_norm << " ";

						if (e_norm < e_min)
						{
							e_min = e_norm;
							best_mode = mode;
						}
						cv::Point3d tmp(mode(0),mode(1),mode(2));
						tmp_vec.push_back(tmp);
					} // Fin de la Evaluacion de las modas
					std::cout << std::endl;

					// Revisando si la prediccion mas cercana es un inlier
					if (e_min < 0.3)
					{
						std::cout << "Inlier, estimated mode: " << best_mode(0) << "," << best_mode(1) << "," << best_mode(2) << std::endl;
						inlier++;
					}
					else
						outlier++;

					//cv::viz::Viz3d window("Coordinate Frame");
					//window.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());
					//window.showWidget("points", cv::viz::WCloud(tmp_vec, cv::viz::Color::green()));

					//window.spin();
				}

			} // Fin de ELSE

			// Imprimiendo resultados finales
			std::cout << "Pixeles Depth NaN: " << depth_nan << std::endl;
			std::cout << "Pixeles que no pudieron ser evaluados en ningun Arbol: " << mode_nan << std::endl;
			std::cout << "Inliers: " << inlier << " Outliers: " << outlier << std::endl;

			std::cout << "====================================================\n";


			cv::Mat m = data_->getRgbImage(frame).clone();
			cv::namedWindow("Display RGB",cv::WINDOW_AUTOSIZE);
		    cv::namedWindow("Display Depth",cv::WINDOW_AUTOSIZE);

			cv::circle(m,cv::Point2i(col,row),5,cv::Scalar(255,255,255));
		    cv::imshow("Display RGB",m);
			show_depth_image("Display Depth", data_->getDepthImage(frame));

			cv::waitKey();

			cont_++;

		} while(cont_ < 500);

		****/

	} // Fin de probar todos los frames


} // Fin de la funcion Test

// FUNCION TEST_FRAME
//-------------------
// Input: un entero que representa en numero de frame en la lista de pares RGBD
// (Este par NO necesariamente cuenta con una correspondencia en el GT)
// Output: Un objeto hipotesis que representa la mejor Matriz de Transformacion de Relocalizacion
template<typename D, typename RGB>
Hypothesis Forest<D,RGB>::Test_Frame(const cv::Mat& i0_ref, const cv::Mat& d0_ref, double &scale)
{
	///**
	std::vector<Hypothesis> hypotheses;

#ifdef PRINT_CREATE_HYPOTHESIS_INFO
	std::cout << "Creando Hipotesis\n";
	std::cout << "==================================================\n";
#endif

	// Extraemos K_init hipotesis iniciales
	int K = 1024;
	int offset = 50;
	for (int i = 0; i < K; ++i)
	{
		Hypothesis h;
		// 3 points
		h.input_.resize(3, 3);
		h.output_.resize(3, 3);

		int no_inliers = 0, no_outliers = 0;
		for (int j = 0; j < 3; ++j)
		{
			int col = random_->Next( 0 + offset, settings_->image_width_ - offset);
			int row = random_->Next( 0 + offset, settings_->image_height_ - offset);
			//int col = random_->Next( 0 , settings_->image_width_);
			//int row = random_->Next( 0 , settings_->image_height_);

			ushort depth = d0_ref.at<ushort>(cv::Point2i(col,row));
#ifdef PRINT_CREATE_HYPOTHESIS_INFO
			std::cout << "depth: " << depth << std::endl;
#endif
			auto modes = this->Eval(col, row, i0_ref, d0_ref);

			if (depth == 0 || modes.empty())
			{
				j--;
			}
			else
			{
				// Calculamos las coordenadas en Camera Space
				// ------------------------------------------
				double X,Y,Z;
				Z = (double)depth / (double)settings_->depth_factor_;

				X = (col - settings_->cx) * Z / settings_->fx;
				Y = (row - settings_->cy) * Z / settings_->fy;

				Eigen::Vector3d p_camera(X,Y,Z); // Declaramos un vector temporal

				//h.camera_space_point = p_camera; // No se si esto sea necesario
				// Agregamos un punto para el input!
				h.input_.col(j) = p_camera;

				// Calculamos las coordenadas en World Space
				// ------------------------------------------

				// Agregamos el punto del camera space a las propiedad de la hipotesis
				Eigen::Vector3d selected_mode;
				if (modes.size() > 1)
					selected_mode = modes.at(random_->Next(0, modes.size()));
				else
					selected_mode = modes.front();

				// Agregamos un punto para el output!
				h.output_.col(j) = selected_mode;

			} // Fin de la Condicional de pixeles invalidos				

		} // Terminando de recolector los tres puntos iniciales

		// Usando el algoritmo de Kabsch y refistrando la Hipotesis
		// --------------------------------------------------------
		Eigen::Affine3d transform = Find3DAffineTransform(h.input_, h.output_);
		h.pose_ = transform;
		h.energy_ = 0;
		hypotheses.push_back(h);

	} // Fin de Calcular K = 1024 hipotesis

	//**/

	///***
	// Hypothesis Optimization
	// ------------------------

	while( K > 1)
	{
		std::vector<cv::Point2i> test_pixels;
		int batch_size = 500;
		for (int i = 0; i < batch_size; ++i)
		{
			int col = random_->Next( 0 + offset, settings_->image_width_ - offset);
			int row = random_->Next( 0 + offset, settings_->image_height_ - offset);
			//int col = random_->Next( 0 , settings_->image_width_);
			//int row = random_->Next( 0 , settings_->image_height_);

			ushort depth = d0_ref.at<ushort>(cv::Point2i(col,row));

			// Filtramos los pixeles con depth no validos
			if (depth == 0)
				--i;
			else
				test_pixels.push_back(cv::Point2i(col, row));
		} // Fin de Samplear el Batch Size

		// Evaluamos el Batch de los pixeles
		// ---------------------------------
		for(auto p : test_pixels)
		{
			auto modes = this->Eval(p.x, p.y, i0_ref, d0_ref);

			if (!modes.empty())
			{
				D test = d0_ref.at<D>(p);
				double Z = (double)test / (double)settings_->depth_factor_;
				double X = (p.x - settings_->cx) * Z / settings_->fx;
				double Y = (p.y - settings_->cy) * Z / settings_->fy;

				Eigen::Vector3d p_camera(X, Y, Z);
				//std::cout << "p_camera " << p_camera(0) << "," << p_camera(1) << "," << p_camera(2) << std::endl;

				for (int i = 0; i < K; ++i)
				{
					// Estimacion de la hipotesis
					Eigen::Vector3d h_pcamera = hypotheses.at(i).pose_ * p_camera;

					double e_min = DBL_MAX; // Valor maximo del double
					Eigen::Vector3d best_mode;
					Eigen::Vector3d best_camera_p;

					for (auto mode : modes) {
						Eigen::Vector3d e = mode - h_pcamera;
						double e_norm = e.norm();

						if (e_norm < e_min) {
							e_min = e_norm;
							best_mode = mode;
							best_camera_p = p_camera;
						}
					} // Fin del bucle de las modas

					// update energy
					hypotheses.at(i).energy_ += tophat_error(e_min);

					///*
					// inlier
					if (tophat_error(e_min) == 0) {
						//std::cout << "*************Inlier!!\n";
						// add to kabsch matrices
						hypotheses.at(i).input_.conservativeResize(3, hypotheses.at(i).input_.cols() + 1);
						hypotheses.at(i).input_.col(hypotheses.at(i).input_.cols() - 1) = best_camera_p;

						hypotheses.at(i).output_.conservativeResize(3, hypotheses.at(i).output_.cols() + 1);
						hypotheses.at(i).output_.col(hypotheses.at(i).output_.cols() - 1) = best_mode;
					} // Fin del Condicional

				} // Revisando todas las hipotesis

			} // Fin de condicional de revisar si hay modas para evaluar

		} // Fin de Testear todo un batch!

		// sort hypotheses (de forma ascendente)
		std::sort(hypotheses.begin(), hypotheses.begin() + K); // Solo ordena los K primeras hipotesis

	    K = K / 2; // discard half

	    // Refinando las hipotesis
    	for (int i = 0; i < K; ++i){
	    	// Printing ordered energies
	    	std::cout << hypotheses.at(i).energy_ << ", ";
			hypotheses.at(i).pose_ = Find3DAffineTransform(hypotheses.at(i).input_, hypotheses.at(i).output_, scale);

			// Resetting hypothesis energies
	    	hypotheses.at(i).energy_ = 0;

	    	// Resetting inliers
	    	hypotheses.at(i).input_.resize(3, 0);
	    	hypotheses.at(i).output_.resize(3, 0);
    	} std::cout << std::endl;

    	Eigen::Affine3d estimated_pose = hypotheses.front().pose_;
    	Eigen::IOFormat stdFormat(4);

    	std::cout << "Estimated Pose:\n" << estimated_pose.matrix().format(stdFormat) << std::endl;

    	std::cout << "scale: " << scale << std::endl;

    	std::cout << "==================================================\n";
    	
	} // Fin de refinar las hipotesis

	return hypotheses.front();

} // Fin de la funcion Test_Frame

template<typename D, typename RGB>
void Forest<D,RGB>::show_tree_estimation(int depth_level)
{
	// map para guardar las coordenadas
	std::map<std::string,std::vector<cv::Point3d>> tree_mapping;

	std::vector<LabeledPixel> labeled_data = LabelData();
	std::cout << "LabelData Size: " << labeled_data.size() << std::endl;

	for (int tree = 0; tree < forest_.size(); ++tree)
	{
		for (int i = 0; i < labeled_data.size(); ++i)
		{
			std::string s = ""; bool is_valid = true;
			forest_[tree]->eval_recursive(forest_[tree]->getRootPointer(), labeled_data[i].pos_.x, labeled_data[i].pos_.y, depth_level, s,data_->getRgbImage(labeled_data[i].frame_), data_->getDepthImage(labeled_data[i].frame_),is_valid);

			if (is_valid)
			{
				//std::cout << s << std::endl;
				tree_mapping[s].push_back(labeled_data[i].label_);
			}

		}

		std::map<std::string,std::vector<cv::Point3d>>::iterator it = tree_mapping.begin();

		/*
		// Mostrando nodos independientes
		while(it != tree_mapping.end())
		{
			cv::viz::Viz3d window("Coordinate Frame");
			window.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());
			std::cout << "node" << it->first << std::endl;
			//cv::Vec3b color(random_->Next(0,255),random_->Next(0,255),random_->Next(0,255));

			window.showWidget("points", cv::viz::WCloud(it->second, cv::viz::Color(random_->Next(0,255),random_->Next(0,255),random_->Next(0,255))));
			//window.showWidget("points", cv::viz::WCloud(tmp_pos, tmp_color));
			/// Construct a cube widget
		    cv::viz::WCube cube_widget(cv::Point3f(1.5,1.5,3.0), cv::Point3f(-1.5,-1.5,0.0), true, cv::viz::Color::blue());
		    cube_widget.setRenderingProperty(cv::viz::LINE_WIDTH, 4.0);
		    window.showWidget("Cube Widget", cube_widget);

			window.spin();

			it++;
		}
		*/

		// Mostrando todas las clasificaciones

		cv::viz::Viz3d window("Coordinate Frame");
		window.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());

		int cont = 0;
		while(it != tree_mapping.end())
		{
			std::cout << "node" << it->first << std::endl;
			//cv::Vec3b color(random_->Next(0,255),random_->Next(0,255),random_->Next(0,255));

			window.showWidget("points" + std::to_string(cont), cv::viz::WCloud(it->second, cv::viz::Color(random_->Next(0,255),random_->Next(0,255),random_->Next(0,255))));
			//window.showWidget("points", cv::viz::WCloud(tmp_pos, tmp_color));
			/// Construct a cube widget

			it++;
			cont++;
		}

	    cv::viz::WCube cube_widget(cv::Point3f(1.5,1.5,3.0), cv::Point3f(-1.5,-1.5,0.0), true, cv::viz::Color::blue());
	    cube_widget.setRenderingProperty(cv::viz::LINE_WIDTH, 4.0);
	    window.showWidget("Cube Widget", cube_widget);

		window.spin();
	}


}




// Declarando el template para su compilación
template class Forest<ushort,cv::Vec3b>;