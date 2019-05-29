
#include "../inc/direct_odometry.hpp"

void DirectOdometryA::doAlignment(const cv::Mat& i0_ref, const cv::Mat& d0_ref, const cv::Mat& i1_ref, myVector6d &xi, double& err)
{
	std::cout << "calling the Direct Odometry Algorithm\n";

	// Obtaining Initial Pyramidal Images
    std::vector<cv::Mat> vo_imgs_ref,vo_imgs,vo_depths,vo_intrinsics;

    int pyramid_level = 5;

    // Creating pyramidal images
	prepare_pyramidal_rgbs(i0_ref,vo_imgs_ref,pyramid_level);
	prepare_pyramidal_rgbs(i1_ref,vo_imgs,pyramid_level);
	prepare_pyramidal_depths(d0_ref,vo_depths,pyramid_level);
	prepare_pyramidal_intrinsics(settings_->K_ref,vo_intrinsics,pyramid_level);

	for (int i = pyramid_level - 1; i >= 1; --i)
	{
		//std::cout << std::endl << "level = " << i << std::endl << std::endl;
		cv::Mat i0,i1,d0,K;
		
		// Setting the images
		i0 = vo_imgs_ref.at(i);
		i1 = vo_imgs.at(i);
		d0 = vo_depths.at(i);
		K = vo_intrinsics.at(i);

		// Calculating gradient images
		cv::Mat XGradient, YGradient;
		Gradient(i1,XGradient,YGradient);

		double last_err = DBL_MAX;

		FOR(it,20)
		{
	        //std::cout << "\niteracion " << it << std::endl << std::endl;
	        Eigen::VectorXd R;
	        Eigen::MatrixXd J;

	        // Calculamos los residuales y el jacobiano
	        // mostramos la imagen de los residuales
	        auto start = cv::getTickCount();
	        CalcDiffImage(i0,d0,i1,XGradient,YGradient,xi,K,R,J);
	        auto end = cv::getTickCount();
	        double time = (end - start) / cv::getTickFrequency();
	        //std::cout << "Residual and Jacobian Process Time: " << time << " seconds\n";

	        // Calculamos nuestro diferencial de xi
	        start = cv::getTickCount();
	        Eigen::VectorXd d_xi = -(J.transpose() * J).inverse() * J.transpose() * R;
	        end = cv::getTickCount();
	        time = (end - start) / cv::getTickFrequency();
	        std::cout.precision(6);
	        //std::cout << "d_xi Process Time: " << time << " seconds\n";

	        //std::cout << "d_xi:\n" << d_xi.transpose() << std::endl;

	        Eigen::VectorXd last_xi = xi;
	        xi = rbm2twistcoord( twistcoord2rbm(d_xi) * twistcoord2rbm(xi));

	        //std::cout << "xi:\n" << xi.transpose() << std::endl;

	        // Calculamos la media de todos los errores cuadráticos
	        // de esta forma el error no estará ligado al numero de muestras
	        // que varia en el cambio de cada nivel
	        // ademas, solo actualizaremos si el error es menor

	        err = R.dot(R)/R.rows();
	        //std::cout << "err=" << err << " last_err=" << last_err << std::endl;

	        // Visualizacion de los residuales
	        //CalcDiffImage(vo_imgs_ref.at(0), vo_depths.at(0), vo_imgs.at(0), xi, vo_intrinsics.at(0));

	        if( err / last_err > 0.995){
	            //xi = last_xi;
	            break;
	        }

	        last_err = err;

        } // Fin de las 20 iteraciones

	    //cv::waitKey(); // activar aqui para realizar una inspeccion frame a frame

	} // Fin de iterar en la piramide

} // Fin de la funcion doAlignment

// Funcion principal para calcular los residuales y jacobianos
void DirectOdometryA::CalcDiffImage(const cv::Mat & i0, const cv::Mat & d0, const cv::Mat & i1,const cv::Mat& XGradient, const cv::Mat& YGradient, const myVector6d &xi, const cv::Mat& K, Eigen::VectorXd &Res, Eigen::MatrixXd &Jac){
	//writeMat2File(i0,"trash_data/i0.txt");
    //writeMat2File(d0,"trash_data/d0.txt");
    //writeMat2File(i1,"trash_data/i1.txt");
    //writeMat2File(K,"trash_data/K.txt");

    // Obtenemos el tamaño de la imagen
    int rows = i0.rows, cols = i0.cols;

    /* Cambiar este codigo para la matriz*/
    // Pasamos la matriz intrinseca a Eigen+
    Eigen::Matrix3d  eigen_K; // Tiene que coincidir con el tipo de dato de K
    eigen_K << K.at<double>(0,0), K.at<double>(0,1), K.at<double>(0,2),
               K.at<double>(1,0), K.at<double>(1,1), K.at<double>(1,2),
               K.at<double>(2,0), K.at<double>(2,1), K.at<double>(2,2);
    Eigen::Matrix3d eigen_K_inverse = eigen_K.inverse();


    double fx = eigen_K(0,0), fy = eigen_K(1,1);
    //std::cout << "input xi=" << xi.transpose() << std::endl;

    // Calculamos la transformación rigid-body motion
    Eigen::Matrix4d g = twistcoord2rbm(xi);
    // Creamos nuestros mapeos para x e y
    cv::Mat map_warped_x(i1.size(),i1.type(),-100.0);
    cv::Mat map_warped_y(i1.size(),i1.type(),-100.0);

    // Y los mapeos para los Warp Coordinates(no proyectados)
    // restamos 100 para simular NaN values
    cv::Mat xp(i1.size(),i1.type(),-100.0);
    cv::Mat yp(i1.size(),i1.type(),-100.0);
    cv::Mat zp(i1.size(),i1.type(),-100.0);

    auto start = cv::getTickCount();
    //Eigen::Vector2d coord0;
    Eigen::Vector3d world_coord;
    Eigen::Vector4d transformed_coord;
    Eigen::Vector3d projected_coord;
    //Eigen::Vector2d warped_coord;
    // Calculamos las nuevas coordenadas
    double de;
    double* w_c0 = &world_coord(0);
    double* w_c1 = &world_coord(1);
    double* w_c2 = &world_coord(2);
    double* t_c0 = &transformed_coord(0);
    double* t_c1 = &transformed_coord(1);
    double* t_c2 = &transformed_coord(2);
    transformed_coord(3) = 1;
    double* p_c0 = &projected_coord(0);
    double* p_c1 = &projected_coord(1);
    double* p_c2 = &projected_coord(2);
    FOR(j,rows){
        const double* d = d0.ptr<double>(j);
        double* map_w_x = map_warped_x.ptr<double>(j);
        double* map_w_y = map_warped_y.ptr<double>(j);

        double* x = xp.ptr<double>(j);
        double* y = yp.ptr<double>(j);
        double* z = zp.ptr<double>(j);
        FOR(i,cols){
            de = d[i]; // guardamos temporalmente este valor para no estarlo calculando a cada momento
            if( d[i] > 0 ){
                // El problema con usar las asignaciones del Eigen
                // es q no son eficientes. lo que podemos salvarlo al usar punteros
                // el codigo se hace un poco engorroso pero es mas liviano
                // dejamos que eigen se haga cargo sólo de las multiplicaciones

                 //coord0 << i, j;
                 //std::cout << "x,y " << coord0 << " ;";

                 //world_coord << d[i] * i, d[i] * j, d[i];
                 *w_c0 = de * i; *w_c1 = de * j; *w_c2 = de;
                 //world_coord = d[i] * world_coord;
                 world_coord = eigen_K_inverse * world_coord;
                 //std::cout << world_coord << " ;";

                 // Transformed coord by rigid-body motion
                 //transformed_coord << world_coord, 1;
                 *t_c0 = *w_c0; *t_c1 = *w_c1; *t_c2 = *w_c2;
                 transformed_coord = g * transformed_coord;
                 //std::cout << transformed_coord << " ;";

                 //projected_coord << *t_c0,*t_c1,*t_c2;
                 *p_c0=*t_c0; *p_c1=*t_c1; *p_c2=*t_c2;
                 projected_coord = eigen_K * projected_coord;
                 //std::cout << projected_coord << " ;";

                 //warped_coord << projected_coord(0) / projected_coord(2), projected_coord(1) / projected_coord(2);
                 //std::cout << warped_coord << " ;\n";

                 // Probemos usar los mapeos de opencv
                 map_w_x[i] = *p_c0 / *p_c2;
                 map_w_y[i] = *p_c1 / *p_c2;

                 // Verificamos que el número sea positivo
                 // Para que no haya problema al calcular el Jacobiano
                 //if(t_c2 > 0.0f){
                     x[i] = *t_c0;
                     y[i] = *t_c1;
                     z[i] = *t_c2;
                 //}
            }
        } // Fin Bucle FOR cols
    } // Fin Bucle FOR rows
    auto end = cv::getTickCount();
    double time = (end - start) / cv::getTickFrequency();
    //std::cout << "Warping Process Time: " << time << "seconds" << std::endl;

    // Declaramos una matriz para guardar la interpolacion y despues los residuales
    cv::Mat i1_warped = cv::Mat::zeros(cv::Size(cols,rows),i1.type());
    // Realizamos la interpolación
    start = cv::getTickCount();
    interpolate(i1,i1_warped,map_warped_x,map_warped_y,0);
    end = cv::getTickCount();
    time = (end - start) / cv::getTickFrequency();
    //std::cout << "1st Interpolation Process Time: " << time << "seconds" << std::endl;
    //cv::remap(i1,i1_warped,map_warped_x,map_warped_y,CV_INTER_LINEAR,cv::BORDER_CONSTANT, cv::Scalar(0));


    cv::Mat residuals = cv::Mat::zeros(i1.size(),i1.type());
    residuals = i0 - i1_warped; // Revisar estas operaciones para el calculo de los maps!!!!

    // Interpolamos sobre las gradientes
    cv::Mat map_XGradient = cv::Mat::zeros(cv::Size(cols,rows),i1.type());
    cv::Mat map_YGradient = cv::Mat::zeros(cv::Size(cols,rows),i1.type());
    start = cv::getTickCount();
    interpolate(XGradient,map_XGradient,map_warped_x,map_warped_y,1); // El padding es 1 por que no se pueden calcular
    interpolate(YGradient,map_YGradient,map_warped_x,map_warped_y,1); // valores de gradiente para los bordes
    end = cv::getTickCount();
    time = (end - start) / cv::getTickFrequency();
    //std::cout << "2nd BiInterpolation Process Time: " << time << "seconds" << std::endl;


    start = cv::getTickCount();

    /*** CONSTRUCION DE c y r0 ***/
    // Formamos las matrices pero usando la libreria eigen
    Eigen::VectorXd r(rows * cols);
    Eigen::MatrixXd c(rows * cols,6);
    int cont = 0;
    double map_w_x, map_w_y;
    double gradX, gradY;
    double x,y,z;
    FOR(j,rows){
        double* res = residuals.ptr<double>(j);
        double* m_w_x = map_warped_x.ptr<double>(j);
        double* m_w_y = map_warped_y.ptr<double>(j);
        double* x_ = xp.ptr<double>(j);
        double* y_ = yp.ptr<double>(j);
        double* z_ = zp.ptr<double>(j);
        double* xgrad = map_XGradient.ptr<double>(j);
        double* ygrad = map_YGradient.ptr<double>(j);
        FOR(i,cols){
            // Sólo usaremos los datos que sean válidos
            // Es decir aquellos que tengan valores válidos de gradiente
            // 1 < map_warped_x < width -1; 1 < map_warped_y < height -1
            // Valores válidos para el Image warped
            // i1_warped != 0
            // Valores válidos para las coordenadas de pixel en 3D
            // xp,yp,zp != -100
            map_w_x = m_w_x[i];
            map_w_y = m_w_y[i];

            if( (1 < map_w_x && map_w_x < cols-2) &&
                (1 < map_w_y && map_w_y < rows-2) &&
                i1_warped.at<double>(j,i) != 0 &&
                x_[i] != -100){
                    // Residuales
                    r(cont) = res[i];

                    gradX = xgrad[i]; gradY = ygrad[i];
                    gradX *= fx; gradY *= fy;
                    x = x_[i]; y = y_[i]; z = z_[i];

                    // Jacobiano
                    c(cont,0) = gradX / z;
                    c(cont,1) = gradY / z;
                    c(cont,2) = -( gradX * x + gradY * y ) / (z*z);
                    c(cont,3) = -( gradX * x * y / (z*z)) -  (gradY * (1 + (y*y)/(z*z)));
                    c(cont,4) = ( gradX * (1 + (x*x)/(z*z))) + (gradY * x * y / (z*z));
                    c(cont,5) = (- gradX * y + gradY * x) / z;

                    cont++;
            }
        } // Fin bucle for Cols
    }// Fin bucle for Rows
    end = cv::getTickCount();
    time = (end - start) / cv::getTickFrequency();
    //std::cout << "R0 y C Interpolation Process Time: " << time << "seconds" << std::endl;

    /** Retornamos los Residuales y el Jacobiano **/
    // Hacemos un slice con el conteo de "cont" pixeles validos
    Res = r.block(0,0,cont,1);
    Jac = c.block(0,0,cont,6); Jac = -Jac;

    /**
    // displaying ressults

    // FILTRAMOS LOS VALORES CON VALOR 0 DESPUES DE LA INTERPOLACION
    FOR(j,rows)
        FOR(i,cols){
            // la siguiente condición funciona por la mayoria
            // de los numeros que si son tomados en cuenta no son exactamente 0
            if(i1_warped.at<double>(j,i) == 0){
                residuals.at<double>(j,i) = -1;
            }
        }

    // Como las diferencias entre imágenes están en el rango de [-1,1]
    // Sumamos 1 a todos los valores para que el intervalo vaya de [0,2]
    residuals = residuals + 1.0f;

    // Aquí aplicaremos un mapeo proporcional a este intervalo
    // y le aplicamos una mascara de colores para observar las zonas
    // de mayor diferencia

    double min,max;
    cv::minMaxIdx(residuals, &min, &max);
    //std::cout << "max: " << max << "min: " << min << std::endl;

    cv::Mat adjMap;
    cv::convertScaleAbs(residuals, adjMap, 255 / max);

    cv::Mat FalseColorMap;
    cv::applyColorMap(adjMap,FalseColorMap,cv::COLORMAP_BONE);

    // En este mapeo de Colores los pixeles con mas alto valor son los Rojos y los de mínimo azules
    // Observemos que i1 toma los valores positivos e i0 toma los valores negativos

    cv::imshow("Diff", FalseColorMap);

    //cv::waitKey();
    **/ // FIN DE VISUALIZACIÓN BÁSICA

    /** // Showing dual difference color maps
    cv::Mat FalseColorMap2;
    cv::applyColorMap(adjMap,FalseColorMap2,cv::COLORMAP_JET);

    // En este mapeo de Colores los pixeles con mas alto valor son los Rojos y los de mínimo azules
    // Observemos que i1 toma los valores positivos e i0 toma los valores negativos

    cv::imshow("Diff2", FalseColorMap2);
    **/

} // Fin de la funcion

void DirectOdometryA::CalcDiffImage(const cv::Mat & i0, const cv::Mat & d0, const cv::Mat & i1, const myVector6d &xi, const cv::Mat &K){
    // Obtenemos el tamaño de la imagen
    int rows = i0.rows, cols = i0.cols;

    // Pasamos la matriz intrinseca a Eigen+
    Eigen::Matrix3d  eigen_K; // Tiene que coincidir con el tipo de dato de K
    eigen_K << K.at<double>(0,0), K.at<double>(0,1), K.at<double>(0,2),
               K.at<double>(1,0), K.at<double>(1,1), K.at<double>(1,2),
               K.at<double>(2,0), K.at<double>(2,1), K.at<double>(2,2);
    Eigen::Matrix3d eigen_K_inverse = eigen_K.inverse();

    // Declaramos una matriz para guardar los residuales
    cv::Mat i1_warped = cv::Mat::zeros(cv::Size(cols,rows),i1.type());
    // Calculamos la transformación rigid-body motion
    Eigen::Matrix4d g = twistcoord2rbm(xi);
    // Creamos nuestros mapeos para x e y
    cv::Mat map_warped_x, map_warped_y;
    map_warped_x.create(i1.size(), CV_32FC1); // Este debe ser float por una asercion de la funcion remap
    map_warped_y.create(i1.size(), CV_32FC1);

    // Y los mapeos para los Warp Coordinates(no proyectados)
    // restamos 100 para simular NaN values
    cv::Mat xp, yp, zp;
    xp = cv::Mat::zeros(i1.size(),i1.type()) - 100;
    yp = cv::Mat::zeros(i1.size(),i1.type()) - 100;
    zp = cv::Mat::zeros(i1.size(),i1.type()) - 100;

    // Calculamos las nuevas coordenadas
    FOR(j,rows){
        FOR(i,cols){
            if( d0.at<double>(j,i) > 0 ){
                 Eigen::Vector2d coord0(i,j);
                 //std::cout << "x,y " << coord0 << " ;";

                 Eigen::Vector3d world_coord;
                 world_coord << coord0 , 1;
                 world_coord = eigen_K_inverse * d0.at<double>(j,i) * world_coord;
                 //std::cout << world_coord << " ;";

                 // Transformed coord by rigid-body motion
                 Eigen::Vector4d transformed_coord;
                 transformed_coord << world_coord, 1;
                 transformed_coord = g * transformed_coord;
                 //std::cout << transformed_coord << " ;";

                 Eigen::Vector3d projected_coord;
                 projected_coord << transformed_coord(0), transformed_coord(1), transformed_coord(2);
                 projected_coord = eigen_K * projected_coord;
                 //std::cout << projected_coord << " ;";

                 Eigen::Vector2d warped_coord;
                 warped_coord << projected_coord(0) / projected_coord(2), projected_coord(1) / projected_coord(2);
                 //std::cout << warped_coord << " ;\n";

                 // Probemos usar los mapeos de opencv
                 map_warped_x.at<float>(j,i) = warped_coord(0);
                 map_warped_y.at<float>(j,i) = warped_coord(1);

                 } else {
                    map_warped_x.at<float>(j,i) = -100;
                    map_warped_y.at<float>(j,i) = -100;
                 } // Fin de Condicional exterior
        } // Fin Bucle FOR cols
    } // Fin Bucle FOR rows

    // Interpolamos los valores para los warped coordinates
    // dejaremos la interpolacion de opencv porque aqui solo la usamos para visualizar
    cv::remap(i1,i1_warped,map_warped_x,map_warped_y,CV_INTER_LINEAR,cv::BORDER_CONSTANT, cv::Scalar(0));

    cv::Mat residuals = cv::Mat::zeros(i1.size(),i1.type());
    residuals = i0 - i1_warped; // Revisar estas operaciones para el calculo de los maps!!!!

    /*FILTRAMOS LOS VALORES CON VALOR 0 DESPUES DE LA INTERPOLACION*/
    FOR(j,rows)
        FOR(i,cols){
            // la siguiente condición funciona por la mayoria
            // de los numeros que si son tomados en cuenta no son exactamente 0
            if(i1_warped.at<double>(j,i) == 0){
                residuals.at<double>(j,i) = -1;
            }
        }

    // Como las diferencias entre imágenes están en el rango de [-1,1]
    // Sumamos 1 a todos los valores para que el intervalo vaya de [0,2]
    residuals = residuals + 1.0f;

    // Aquí aplicaremos un mapeo proporcional a este intervalo
    // y le aplicamos una mascara de colores para observar las zonas
    // de mayor diferencia

    double min,max;
    cv::minMaxIdx(residuals, &min, &max);
    std::cout << "max: " << max << "min: " << min << std::endl;

    cv::Mat adjMap;
    cv::convertScaleAbs(residuals, adjMap, 255.0 / max);

    cv::Mat FalseColorMap;
    cv::applyColorMap(adjMap,FalseColorMap,cv::COLORMAP_BONE);

    // En este mapeo de Colores los pixeles con mas alto valor son los Rojos y los de mínimo azules
    // Observemos que i1 toma los valores positivos e i0 toma los valores negativos

    cv::imshow("Actual comparison", FalseColorMap);

    //cv::waitKey();

    /** // Showing dual difference color maps
    cv::Mat FalseColorMap2;
    cv::applyColorMap(adjMap,FalseColorMap2,cv::COLORMAP_JET);

    // En este mapeo de Colores los pixeles con mas alto valor son los Rojos y los de mínimo azules
    // Observemos que i1 toma los valores positivos e i0 toma los valores negativos

    cv::imshow("Diff2", FalseColorMap2);
    **/

}

void DirectOdometryA::prepare_pyramidal_rgbs(const cv::Mat& in, std::vector<cv::Mat>& vec, int level)
{
	cv::Mat in_double, out;
	in.convertTo(in_double, CV_64FC3);

	out = cv::Mat::zeros(cv::Size(in.cols,in.rows),CV_64FC1);

	for(int y = 0; y < out.rows; y++)
        for(int x = 0; x < out.cols; x++){
            // no necesariamente cumple con el orden RGB, puede ser tambien BGR
            double b = in_double.at<cv::Vec3d>(y,x)[0];
            double g = in_double.at<cv::Vec3d>(y,x)[1];
            double r = in_double.at<cv::Vec3d>(y,x)[2];

            // Hay varias formas para calcular el grayscale
            //img.at<myNum>(y,x) = (r+g+b)/3; // segun el paper
            out.at<double>(y,x) = 0.299 * r + 0.587 * g + 0.114 * b; // segun opencv (CCIR 601)

            //Normalizamos el color para [0,1]
            out.at<double>(y,x) = out.at<double>(y,x) / 255.0;
        }

    // Downscaling images // empezamos en 1 por q ya añadimos la imagen principal
    vec.push_back(out);
    for (int i = 1; i < level; ++i)
    {
    	cv::Mat temp;
    	downscaler->doDownscale(vec.back(),temp,intensity);
    	vec.push_back(temp);
    }

} // fin de prepare_rgb

void DirectOdometryA::prepare_pyramidal_depths(const cv::Mat& in, std::vector<cv::Mat>& vec , int level)
{
	cv::Mat out;

	in.convertTo(out, CV_64FC1);

	out = out / settings_->depth_factor_;

	// Downscaling images
    vec.push_back(out);
    for (int i = 1; i < level; ++i)
    {
    	cv::Mat temp;
    	downscaler->doDownscale(vec.back(),temp,depth);
    	vec.push_back(temp);
    }

}

void DirectOdometryA::prepare_pyramidal_intrinsics(const cv::Mat& in, std::vector<cv::Mat>& vec, int level)
{
	// Downscaling images
    vec.push_back(in);
    for (int i = 1; i < level; ++i)
    {
    	cv::Mat temp;
    	downscaler->doDownscale(vec.back(),temp,intrinsics);
    	vec.push_back(temp);
    }

    //for(auto item : vec)
    //    std::cout << item << std::endl;
}

// Funcion para calcular la gradiente en direccion X e Y
void DirectOdometryA::Gradient(const cv::Mat & InputImg, cv::Mat & OutputXImg, cv::Mat & OutputYImg)
{
    // Creamos un Mat de Zeros con las mismas propiedades de InputImg
    int rows = InputImg.rows, cols = InputImg.cols;
    OutputXImg = cv::Mat::zeros(rows,cols,InputImg.type());
    OutputYImg = cv::Mat::zeros(rows,cols,InputImg.type());

    // Iteamos para calcular las gradientes
    // Observamos que no es posible calcular esta gradiente en los márgenes de la imagen
    for(int j = 1; j < rows-1; ++j){
        double* grad_x = OutputXImg.ptr<double>(j);
        double* grad_y = OutputYImg.ptr<double>(j);

        const double* input_j = InputImg.ptr<double>(j);
        const double* input_ja = InputImg.ptr<double>(j+1);
        const double* input_jb = InputImg.ptr<double>(j-1);

        for(int i = 1; i < cols-1; ++i){
            // Gradiente en X
            grad_x[i] = 0.5f * (input_j[i+1] - input_j[i-1]);
            // Gradiente en Y
            grad_y[i] = 0.5f * (input_ja[i] - input_jb[i]);
        }
    }

    
    /** // Codigo para visualizar las gradientes
    double min,max;
    OutputXImg = OutputXImg + 0.5f;
    cv::minMaxIdx(OutputXImg,&min,&max);
    //cout << "max:" << max << "min:" << min << endl;
    cv::Mat adjMap;
    OutputXImg.convertTo(adjMap,CV_8UC1,255/(max-min),-min); // Coloramiento de acuerdo a valores maximos y minimos

    cv::Mat FalseColorMap;
    cv::applyColorMap(adjMap,FalseColorMap,cv::COLORMAP_BONE);
    cv::cvtColor(FalseColorMap,FalseColorMap,CV_BGR2RGB);

    cv::imshow("Hola",FalseColorMap);

    OutputYImg = OutputYImg + 0.5f;
    cv::minMaxIdx(OutputYImg,&min,&max);
    cv::Mat adjMap2;
    OutputYImg.convertTo(adjMap2,CV_8UC1,255/(max-min),-min); // Coloramiento de acuerdo a valores maximos y minimos

    cv::Mat FalseColorMap2;
    cv::applyColorMap(adjMap2,FalseColorMap2,cv::COLORMAP_BONE);
    cv::cvtColor(FalseColorMap2,FalseColorMap2,CV_BGR2RGB);

    cv::imshow("Hola2",FalseColorMap2);

    int key = cv::waitKey();
    **/
} // Fin de la funcion Gradient

void DirectOdometryA::interpolate(const cv::Mat& InputImg, cv::Mat& OutputImg, const cv::Mat& map_x, const cv::Mat& map_y, int padding)
{
    double warp_coord_x, warp_coord_y;
    //double result;
    double a, b, t, s, d, r;
    double t_s, d_s, t_r, d_r;
    FOR(j,InputImg.rows){
        const double* pixel_x = map_x.ptr<double>(j);
        const double* pixel_y = map_y.ptr<double>(j);
        double* pixel_out = OutputImg.ptr<double>(j);
        FOR(i,InputImg.cols){
            // Debemos corregir los valores de double
            warp_coord_x = pixel_x[i];
            warp_coord_y = pixel_y[i];
            // Revisamos si las coord se encuentran dentro de los limites de la imagen
            // Considerando el padding
            if( warp_coord_x > padding && warp_coord_x < InputImg.cols - 1 - padding &&
                warp_coord_y > padding && warp_coord_y < InputImg.rows - 1 - padding ){
                a = warp_coord_y - floor(warp_coord_y);
                b = warp_coord_x - floor(warp_coord_x);
                t = floor(warp_coord_y), d = ceil(warp_coord_y);
                s = floor(warp_coord_x), r = ceil(warp_coord_x);

                t_s = InputImg.at<double>(t,s);
                d_s = InputImg.at<double>(d,s);
                t_r = InputImg.at<double>(t,r);
                d_r = InputImg.at<double>(d,r);

                //result = (d_r * a + t_r * (1-a)) * b + (d_s * a + t_s * (1-a)) * (1-b);

                //*(pixel_out+i) = result;
                //pixel_out[i] = (d_r * a + t_r * (1-a)) * b + (d_s * a + t_s * (1-a)) * (1-b);

                // Podemos ahorrarnos unas cuantas instrucciones si efectuamos algo de algebra para simplificar
                // el calculo de la interpolacion, reduciendo el numero de multiplicaciones, adiciones y asignaciones

                pixel_out[i] = ( t_r + a*(d_r-t_r) )*b + ( t_s + a*(d_s-t_s) )*(1-b);
            }
        } // Fin del FOR interior
    } // Fin del FOR exterior

} // Fin de la funcion interpolate