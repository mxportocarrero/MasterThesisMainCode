#include "../inc/main_system.hpp"

void MainSystem_D::execute()
{
	// Absolute Path Trajetory
    // -----------------------
    // Sobre esta matriz se iran acumulando las demas transformaciones
    Mat44 Identity;
    float id[] = {
        1,  0,  0,  0,
        0,  1,  0,  0,
        0,  0,  1,  0,
        0,  0,  0,  1
    };

    std::copy(id,id+16,Identity.m[0]);

    cv::Mat a(4,4,CV_32FC1,&Identity.m);
    a.convertTo(a, CV_64FC1);
    cv::Affine3d p_abs_rbm(a); // Usaremos la funcion affine3d de opencv para acumular las matrices

    // Declaring Viz Objects for visualtion
    // ------------------------------------
    cv::viz::Viz3d window("Coordinate Frame");
    //window.setBackgroundColor(cv::viz::Color::white(),cv::viz::Color::white()); // Cuando pones los dos colores dan un tono de interpolacion
    window.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());

    // Displaying a cube
    //-------------------
    //cv::viz::WCube cube_widget(cv::Point3f(1.5,1.5,3.0), cv::Point3f(-1.5,-1.5,0.0), true, cv::viz::Color::blue());
    //cube_widget.setRenderingProperty(cv::viz::LINE_WIDTH, 4.0);
    //window.showWidget("Cube Widget", cube_widget);

    // Guardamos los puntos de un modelo de camara
    std::vector<cv::Point3d> cam_model_lines = getFrameCoordPairs(Identity,settings_);

    // Generamos las lineas correspondientes al modelo y registramos en window
    for (int k = 0; k < cam_model_lines.size(); k += 2)
    {
        cv::viz::WLine line( cam_model_lines[k], cam_model_lines[k+1], cv::viz::Color::red() );
        line.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);

        window.showWidget("a" + std::to_string(k),line);

        cv::viz::WLine line_gt( cam_model_lines[k], cam_model_lines[k+1], cv::viz::Color::green() );
        line_gt.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);

        window.showWidget("a_gt" + std::to_string(k),line_gt);
    }


    // Contenedores adicionales para el alineamiento
    std::vector<cv::Vec3d> cam_path, cam_path_gt; // En estos vectores guardaremos el recorrido de la camara
    cam_path.push_back(cv::Vec3d(0,0,0));
    Pose initial_pose = data_->getPose(0);
    cam_path_gt.push_back(cv::Vec3d(initial_pose.m[0][3], initial_pose.m[1][3], initial_pose.m[2][3]));

    // Aqui guardaremos las correspondecias ( estos vectores son miembros de la Clase Base )
    // Agregamos los primeros valores
    matched_coord_data_.push_back(Eigen::Vector3d(0,0,0));
    matched_coord_gt_.push_back(Eigen::Vector3d(initial_pose.m[0][3], initial_pose.m[1][3], initial_pose.m[2][3]));




    // Inicializando con la Relocalizacion
    // -----------------------------------
    // cv::Mat para almacenar los pares rgbd
    cv::Mat i0,i1,d0,d1;
    int init_frame = 0;
    i0 = cv::imread(data_->dataset_path_ + "/" + data_->rgb_filenames_.at(init_frame));
    d0 = cv::imread(data_->dataset_path_ + "/" + data_->depth_filenames_.at(init_frame), cv::IMREAD_ANYDEPTH);
	// Evaluating Random Forest
    double sc;
	Hypothesis h0 = forest_->Test_Frame(i0,d0,sc);

	cv::Affine3d h0_tmp = EigenAffine3d_2_CvAffine3d(h0.pose_);
	p_abs_rbm = h0_tmp; // Actualizamos nuestra primera matriz
    cvAbsPoses.push_back(p_abs_rbm); // Agregando a nuestro vector de CV Poses

    cam_path[0] = h0_tmp.translation();

    matched_coord_data_[0](0) = h0_tmp.translation()[0];
    matched_coord_data_[0](1) = h0_tmp.translation()[1];
    matched_coord_data_[0](2) = h0_tmp.translation()[2];




    // Ventanas para visualizar los pares RGBD
    cv::namedWindow("Display RGB0",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Display RGB1",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Display Depth",cv::WINDOW_AUTOSIZE);

    int matches_count = 0; int rate = 50;
    // Iterating over frames // Estas iteraciones se realizan sobre la sequencia de pares rgbd
    // --------------------- // no importa si no tienen groundtruth asociado
    for (int frame = init_frame + 1; frame < data_->rgb_filenames_.size(); ++frame)
    //for (int frame = init_frame + 1; frame < 500; ++frame)
    {
        myVector6d xi; // instead of Eigen::VectorXd xi(6)
        xi << 0,0,0,0,0,0;

        // Reading rgbd pairs w/o groundtruth synchronization
        i0 = cv::imread(data_->dataset_path_ + "/" + data_->rgb_filenames_.at(frame-1));
        i1 = cv::imread(data_->dataset_path_ + "/" + data_->rgb_filenames_.at(frame));
        d0 = cv::imread(data_->dataset_path_ + "/" + data_->depth_filenames_.at(frame-1), cv::IMREAD_ANYDEPTH);
        d1 = cv::imread(data_->dataset_path_ + "/" + data_->depth_filenames_.at(frame), cv::IMREAD_ANYDEPTH);

        double err;
        direct_odometry_->doAlignment(i0,d0,i1,xi,err); // Estimamos el movimiento

        std::cout << "===============================================\n";
        std::cout << "pair: "<< frame << std::endl;
        std::cout << "ts0 : " << data_->timestamp_rgbd_.at(frame-1) << " ts1 " << data_->timestamp_rgbd_.at(frame) << " diff: " << std::stod(data_->timestamp_rgbd_.at(frame)) - std::stod(data_->timestamp_rgbd_.at(frame-1)) << std::endl;
        std::cout << "xi: " << xi.transpose() << std::endl;
        std::cout << "err = " << err << std::endl; // Ojo este error esta en pixeles.. no son errores metricos
        std::cout << "===============================================\n";

        std::cout << "Frame: " << frame << std::endl;
        std::cout << "Time Difference: " << std::stod(data_->timestamp_rgbd_.at(frame)) - std::stod(data_->timestamp_rgbd_.at(frame-1)) << std::endl;

        // ACCUMULATING ESTIMATED CAMERA POSE
        // ----------------------------------
        // Intercalamos entre ambos algoritmos
        //bool change_conditions = frame % rate != 0;
        //bool change_conditions = err < 0.0025;
        bool change_conditions = err < 0.0025 && frame % rate != 0;
        if (change_conditions)
        {
            std::cout << "Direct Odometry\n";
            cv::Affine3d last_pose = TwistCoord_2_CvAffine3d(xi);

            p_abs_rbm = p_abs_rbm * last_pose;
            //std::cout << "Transformacion\n" << p_abs_rbm.matrix << std::endl;
        }
        else
        {
            // Evaluating Random Forest
            // ------------------------
            double scale; int num_hyps = 10; bool tmp_flag = true;
            std::vector<Hypothesis> hyps;
            for (int i = 0; i < num_hyps; ++i)
            {
                Hypothesis h = forest_->Test_Frame(i1, d1, scale); // <---------------------------------
                if (scale > 0.95)
                    hyps.push_back(h);
                else
                    tmp_flag = false;
            }

            for (int i = 1; i < num_hyps; ++i)
            {
                double dist = (hyps[i].pose_.translation() - hyps[i-1].pose_.translation()).norm();
                if (dist > 0.1)
                    tmp_flag = false;   
            }

            if(tmp_flag)
            {
                std::cout << "Random Forest\n";
                Eigen::Matrix4d acc_mat_sum = Eigen::Matrix4d::Zero();
                for (int i = 0; i < num_hyps; ++i)
                {
                    //std::cout << "Transformacion_pre\n" << h.pose_.matrix() << std::endl;
                    //p_abs_rbm = EigenAffine3d_2_CvAffine3d(h.pose_);
                    acc_mat_sum = acc_mat_sum + hyps[i].pose_.matrix();
                }

                acc_mat_sum /= (double)num_hyps;

                Eigen::Affine3d final_pose;
                final_pose.matrix() = acc_mat_sum;

                p_abs_rbm = EigenAffine3d_2_CvAffine3d(final_pose);
            }
            else
            {
                std::cout << "Direct Odometry\n";
                cv::Affine3d last_pose = TwistCoord_2_CvAffine3d(xi);

                p_abs_rbm = p_abs_rbm * last_pose;
                //std::cout << "Transformacion\n" << p_abs_rbm.matrix << std::endl;
            }

        } // Fin de If de validacion

        // Imprimiendo la transformacion final (cv::Affine3d)
        std::cout << "Transformacion\n" << p_abs_rbm.matrix << std::endl;
        cvAbsPoses.push_back(p_abs_rbm);
        
        // Agregamos el punto de referencia del actual pose estimado
        cam_path.push_back(p_abs_rbm.translation());

        // Alineamos deacuerdo al Groundtruth
        // Debemos buscar correspondencia con el groundtruth
        int matched_frame; cv::Affine3d affine_t;
        if(data_->check_timestamp_rgbd_match(data_->timestamp_rgbd_.at(frame), matched_frame))
        {
            std::cout << "matched_frame " << matched_frame << std::endl;
            Pose pose_gt = data_->getPose(matched_frame);

            printMat44(pose_gt,"Groundtruth Pose");

            // Agregando la data a los vectores para su posterior evaluacion
            matched_coord_data_.push_back(Eigen::Vector3d( cam_path.back()[0],cam_path.back()[1],cam_path.back()[2] ));
            matched_coord_gt_.push_back(Eigen::Vector3d(pose_gt.m[0][3], pose_gt.m[1][3], pose_gt.m[2][3]));

            Eigen::Matrix3d estimated_rot;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    estimated_rot(i,j) = p_abs_rbm.rotation()(i,j);

            matched_rot_data_.push_back( estimated_rot );
            matched_rot_gt_.push_back( poseRotation(pose_gt) );


            matches_count++;

            // DISPLAY GROUNDTRUTH CAMERA POSE
            // -------------------------------
            // Mostramos las correspondencias con el Groundtruth
            cv::Mat b(4,4,CV_32FC1,&pose_gt.m);
            b.convertTo(b, CV_64FC1);
            cv::Affine3d p_gt(b);

            for (int k = 0; k < cam_model_lines.size(); k+=2)
                window.setWidgetPose("a_gt" + std::to_string(k),p_gt);

            cam_path_gt.push_back(p_gt.translation());

            // Dibujamos la linea correspondiente
            int last_idx = cam_path_gt.size() - 1;
            cv::viz::WSphere s(cam_path_gt[last_idx],0.003,10,cv::viz::Color::blue());
            cv::viz::WLine l( cam_path_gt[last_idx],cam_path_gt[last_idx-1],cv::viz::Color::green() );
            l.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);

            window.showWidget("cam_path_pose_gt" + std::to_string(last_idx),s);
            window.showWidget("cam_path_gt" + std::to_string(last_idx),l);

            // Evaluacion del error acumulado
            double acc_err = (matched_coord_data_.back() - matched_coord_gt_.back()).norm();

            std::cout << "Accumulated Error: " << acc_err << " num_frames " << frame % rate <<  std::endl;

            // Evaluacion de la rotacion
            Eigen::Matrix3d mat = matched_rot_gt_.back().transpose() * matched_rot_data_.back();
            Eigen::IOFormat std2Format(8);
            std::cout << "mat: "  << std::endl << mat.format(std2Format) << std::endl;
            double rot_error = ( matched_rot_gt_.back().transpose() * matched_rot_data_.back() ).trace();
                    // Debido a falta de precision, a veces eigen calcula valores para la traza mayores a 3.0, lo que genera inconsistencias en las operaciones posteriores
                    if (rot_error > 3.0)
                        rot_error = 2.9999;
            printf("Rotational Error: %.8f\n", rot_error);
            
            rot_error = ( rot_error - 1.0 ) / 2.0;
            printf("Rotational Error: %.8f\n", rot_error);

            rot_error = acos(rot_error) * 180.0 / M_PI;
            printf("Rotational Error: %.4f\n", rot_error );



        } // FIN DE DISPLAY GROUNDTRUTH CAMERA POSE

        // DISPLAY ESTIMATED CAMERA POSE
        // -----------------------------
        for (int k = 0; k < cam_model_lines.size(); k += 2)
            window.setWidgetPose("a" + std::to_string(k),p_abs_rbm);

        // Dibujamos una linea como el ultimo par de puntos calculados // Recordemo q a este momento el vector ya tenia 1 punto
        int last_idx = cam_path.size() - 1;

        cv::viz::WSphere s( cam_path[last_idx], 0.003, 10, cv::viz::Color::blue() );
        window.showWidget("cam_path_pose"+std::to_string(last_idx),s);

        if (change_conditions)
        {
	        cv::viz::WLine l( cam_path[last_idx], cam_path[last_idx-1], cv::viz::Color::red() );
	        l.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
	        window.showWidget("cam_path"+std::to_string(last_idx),l);
        }
        else
        {
        	cv::viz::WLine l( cam_path[last_idx], cam_path[last_idx-1], cv::viz::Color::yellow() );
	        l.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
	        window.showWidget("cam_path"+std::to_string(last_idx),l);

        }



        // actualizamos el camera path para el recorrido estimado
        //for (int k = 1; k < cam_path.size(); ++k)
        //{
        //    window.setWidgetPose("cam_path_pose" + std::to_string(k), affine_t);
        //    window.setWidgetPose("cam_path" + std::to_string(k), affine_t);
        //}
        // FIN DE DISPLAY ESTIMATED CAMERA POSE

        //window.spinOnce(1,true);
        //window.spin();

        // DISPLAY RGBD PAIRS
        // ------------------
        cv::imshow("Display RGB0",i0);
        cv::imshow("Display RGB1",i1);
        show_depth_image("Display Depth", d0);

        cv::waitKey(1); // Desactivar para analizar frame a frame
        //cv::waitKey();

    } // Fin de iterar sobre los frames

    window.spin();

    displayReconstructionPreview(window);

} // Fin de la funcion execute