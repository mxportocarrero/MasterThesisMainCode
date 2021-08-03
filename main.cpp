
//#include "inc/direct_odometry.hpp"
//#include "inc/forest.hpp"

#include "inc/main_system.hpp"


// Secuencias aptas para la evaluacion
//#define SEQUENCE_PATH "data/rgbd_dataset_freiburg11_desk" // (test21)
//#define SEQUENCE_PATH "data/rgbd_dataset_freiburg11_room" // (test23) La anterior escena pero completa
//#define SEQUENCE_PATH "data/rgbd_dataset_freiburg12_desk" // (test24) escritorio con osito simple
#define SEQUENCE_PATH "data/rgbd_dataset_freiburg13_long_office_household" // (test26) escritorio con osito y separacion

// Secuencias con errores
//#define SEQUENCE_PATH "data/rgbd_dataset_freiburg11_floor"
//#define SEQUENCE_PATH "data/rgbd_dataset_freiburg12_large_with_loop" // Salon de Pruebas con pocos obstaculos

int main(int argc, char const *argv[])
{
    // SETTINGS
    // --------
    // los argumentos representan los siquientes parámetros
    // num_trees / width / heigth / depth_factor / fx / fy / cx / cy
	// Freiburg 1
	//Settings *settings = new Settings(5, 640, 480, 5000, 517.3, 516.5, 318.6, 255.3);
    // Freiburg 2
    //Settings *settings = new Settings(5, 640, 480, 5000, 520.9f, 521.0f, 325.1f, 249.7f);
    // Freiburg 3
    Settings *settings = new Settings(5, 640, 480, 5000, 535.4f, 539.2f, 320.1f, 247.6f);

    // Leyendo el dataset
    Dataset *data = new Dataset(SEQUENCE_PATH);

    // Direct Odometry Algorithm
    DirectOdometryBase *odometry_algorithm = new DirectOdometryA(settings);

    // Inicializando el Bosque desde el archivo Binario
    std::string forest_file_name("test_data/test26.txt"); // Nombre del arbol a leerse

    // Declarando el Sistema a emplearse

    // Algoritmo A
    //MainSystemBase *main_system = new MainSystem_A(data,settings,odometry_algorithm,forest_file_name);

    // Algoritmo B
    // Este algoritmo usa solamente las relocalizaciones hechas por el Regression Random Forest
    //MainSystemBase *main_system = new MainSystem_B(data,settings,odometry_algorithm,forest_file_name);

    // Algoritmo C
    // Realiza la relocalizacion cada n=50 cantidad de frames
    //MainSystemBase *main_system = new MainSystem_C(data,settings,odometry_algorithm,forest_file_name);

    // Algoritmo D
    // es similar al algoritmo C, pero realiza la relocacion
    // tomando como referencia el error calculado por el algoritmo de visual odometry
    MainSystemBase *main_system = new MainSystem_D(data,settings,odometry_algorithm,forest_file_name);

    main_system->execute();
    main_system->EvalSystem("error.txt"); 

	std::cout << "Fin del programa\n";

}