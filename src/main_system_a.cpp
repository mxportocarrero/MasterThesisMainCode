#include "../inc/main_system.hpp"

void MainSystem_A::execute()
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
    window.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());

    // Displaying a cube
    cv::viz::WCube cube_widget(cv::Point3f(1.5,1.5,3.0), cv::Point3f(-1.5,-1.5,0.0), true, cv::viz::Color::blue());
    cube_widget.setRenderingProperty(cv::viz::LINE_WIDTH, 4.0);
    window.showWidget("Cube Widget", cube_widget);

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
    int matches_count=0;

    std::vector<cv::Vec3d> cam_path, cam_path_gt; // En estos vectores guardaremos el recorrido de la camara
    cam_path.push_back(cv::Vec3d(0,0,0));
    Pose initial_pose = data_->getPose(0);
    cam_path_gt.push_back(cv::Vec3d(initial_pose.m[0][3], initial_pose.m[1][3], initial_pose.m[2][3]));

    //std::vector<Eigen::Vector3d> tmp_coord_data, tmp_coord_gt; // Aqui guardaremos las correspondecias
    // Agregamos los primeros valores
    matched_coord_data_.push_back(Eigen::Vector3d(0,0,0));
    matched_coord_gt_.push_back(Eigen::Vector3d(initial_pose.m[0][3], initial_pose.m[1][3], initial_pose.m[2][3]));

    // Ventanas para visualizar los pares RGBD
    cv::namedWindow("Display RGB0",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Display RGB1",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("Display Depth",cv::WINDOW_AUTOSIZE);

    // Iterating over frames // Estas iteraciones se realizan sobre la sequencia de pares rgbd
    // --------------------- // no importa si no tienen groundtruth asociado
    cv::Affine3d affine_t; // Este sera la transformacion final que usaremos para calcular el error
    for (int frame = 1; frame < data_->rgb_filenames_.size(); ++frame)
    {
        myVector6d xi; // instead of Eigen::VectorXd xi(6)
        xi << 0,0,0,0,0,0;

        // Reading rgbd pairs w/o groundtruth synchronization
        cv::Mat i0 = cv::imread(data_->dataset_path_ + "/" + data_->rgb_filenames_.at(frame-1));
        cv::Mat i1 = cv::imread(data_->dataset_path_ + "/" + data_->rgb_filenames_.at(frame));
        cv::Mat d0 = cv::imread(data_->dataset_path_ + "/" + data_->depth_filenames_.at(frame-1), CV_LOAD_IMAGE_ANYDEPTH);
        cv::Mat d1 = cv::imread(data_->dataset_path_ + "/" + data_->depth_filenames_.at(frame), CV_LOAD_IMAGE_ANYDEPTH);

        double err;
        direct_odometry_->doAlignment(i0,d0,i1,xi,err); // Estimamos el movimiento

        std::cout << "===============================================\n";
        std::cout << "pair: "<< frame << std::endl;
        std::cout << "ts0 : " << data_->timestamp_rgbd_.at(frame-1) << " ts1 " << data_->timestamp_rgbd_.at(frame) << " diff: " << std::stod(data_->timestamp_rgbd_.at(frame)) - std::stod(data_->timestamp_rgbd_.at(frame-1)) << std::endl;
        std::cout << "xi: " << xi.transpose() << std::endl;
        std::cout << "err = " << err << std::endl; // Ojo este error esta en pixeles.. no son errores metricos
        std::cout << "===============================================\n";

        // Evaluating Random Forest
        //Hypothesis h = forest->Test_Frame(i1,d1);


        // ACCUMULATING ESTIMATED CAMERA POSE
        // ----------------------------------
        // Intercalamos entre ambos algoritmos
        //if (frame % 10 != 0)
        //{
            //std::cout << "Direct Odometry\n";
            cv::Affine3d last_pose = TwistCoord_2_CvAffine3d(xi);

            p_abs_rbm = p_abs_rbm * last_pose;
            //std::cout << "Transformacion\n" << p_abs_rbm.matrix << std::endl;
        //}
        //else
        //{
            /*    
            std::cout << "Random Forest\n";
            std::cout << "Transformacion_pre\n" << h.pose_.matrix() << std::endl;
            p_abs_rbm = EigenAffine3d_2_CvAffine3d(h.pose_);
            std::cout << "Transformacion\n" << p_abs_rbm.matrix << std::endl;

            if (frame == 1)
            {
                Hypothesis hi = forest->Test_Frame(i0,d0);
                cv::Affine3d hi_tmp = EigenAffine3d_2_CvAffine3d(h.pose_);

                cam_path[0] = hi_tmp.translation();

                tmp_coord_data[0](0) = hi_tmp.translation()[0];
                tmp_coord_data[0](1) = hi_tmp.translation()[1];
                tmp_coord_data[0](2) = hi_tmp.translation()[2];
            }
            */
            
        //}
        
        
        // Agregamos el punto de referencia del actual pose estimado
        cam_path.push_back(p_abs_rbm.translation());

        // Alineamos deacuerdo al Groundtruth
        // Debemos buscar correspondencia con el groundtruth
        int matched_frame;
        if(data_->check_timestamp_rgbd_match(data_->timestamp_rgbd_.at(frame), matched_frame))
        {
            Pose pose_gt = data_->getPose(matched_frame);

            matched_coord_data_.push_back(Eigen::Vector3d( cam_path.back()[0],cam_path.back()[1],cam_path.back()[2] ));
            matched_coord_gt_.push_back(Eigen::Vector3d(pose_gt.m[0][3], pose_gt.m[1][3], pose_gt.m[2][3]));

            Eigen::Matrix3d estimated_rot;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    estimated_rot(i,j) = p_abs_rbm.rotation()(i,j);

            matched_rot_data_.push_back( estimated_rot );
            matched_rot_gt_.push_back(poseRotation(pose_gt));

            // Calculamos nuestra matriz de transformacion, de alineamiento
            Eigen::Matrix4d T = QuickTransformation(matched_coord_gt_, matched_coord_data_);
            //Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

            // Copiamos los valores a un Pose
            Pose t;
            for (int x = 0; x < 4; ++x)
                for (int y = 0; y < 4; ++y)
                    t.m[x][y] = T(x,y);

            // Construimos un objeto affine
            cv::Mat a(4,4,CV_32FC1,&t.m);
            a.convertTo(a, CV_64FC1);
            cv::Affine3d affine(a);
            affine_t = affine; // 

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
            // FIN DE DISPLAY GROUNDTRUTH CAMERA POSE

            //if (matches_count > 5 && matches_count % 20 == 0)
            //{
                //
            //}
        }

        // DISPLAY ESTIMATED CAMERA POSE
        // -----------------------------
        for (int k = 0; k < cam_model_lines.size(); k += 2)
            window.setWidgetPose("a" + std::to_string(k), affine_t * p_abs_rbm);

        // Dibujamos una linea como el ultimo par de puntos calculados // Recordemo q a este momento el vector ya tenia 1 punto
        int last_idx = cam_path.size() - 1;

        cv::viz::WSphere s( cam_path[last_idx], 0.003, 10, cv::viz::Color::blue() );
        cv::viz::WLine l( cam_path[last_idx], cam_path[last_idx-1], cv::viz::Color::red() );
        l.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);

        window.showWidget("cam_path_pose"+std::to_string(last_idx),s);
        window.showWidget("cam_path"+std::to_string(last_idx),l);

        // actualizamos el camera path para el recorrido estimado
        for (int k = 1; k < cam_path.size(); ++k)
        {
            window.setWidgetPose("cam_path_pose" + std::to_string(k), affine_t);
            window.setWidgetPose("cam_path" + std::to_string(k), affine_t);
        }
        // FIN DE DISPLAY ESTIMATED CAMERA POSE

        window.spinOnce();
        //window.spin();

        // DISPLAY RGBD PAIRS
        // ------------------
        cv::imshow("Display RGB0",i0);
        cv::imshow("Display RGB1",i1);
        show_depth_image("Display Depth", d0);

        cv::waitKey(1); // Desactivar para analizar frame a frame

    } // Fin de iterar sobre los frames


    // Conversion de AffineCV a AffineEigen
    Eigen::Affine3d affine_eigen;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            affine_eigen(i,j) = affine_t.matrix(i,j);

    // affine_t.matrix es un tipo especial de matrix (cv::Matx)

    std::cout << "CV Affine" << std::endl << affine_t.matrix << std::endl;
    
    std::cout << "Eigen Affine" << std::endl << affine_eigen.matrix() << std::endl;


    // Actualizamos nuestro vector de correspondencias para calcular el error
    for (int k = 0; k < matched_coord_data_.size(); ++k)
    {
        // Actualizar los valores en el vector matched_coord_data
        Eigen::Vector3d v = affine_eigen * matched_coord_data_[k];
        matched_coord_data_[k] = v;

        // Actualizar los valores de las matirces de roatacion
        // usamos la parte linear de la matriz affine3d
        Eigen::Matrix3d r = affine_eigen.linear() * matched_rot_data_[k];
        matched_rot_data_[k] = r;

    }

} // Fin de la funcion execute