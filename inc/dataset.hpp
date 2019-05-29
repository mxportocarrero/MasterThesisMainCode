#ifndef DATASET_HPP
#define DATASET_HPP

// Esta clase se encarga de leer el archivo que contiene
// los archivos de los pares RGB-D sincronizados

#include "general_includes.hpp"
#include "linear_algebra_functions.hpp"
#include "utilities.hpp"
#include <map>

// CLASE DATASET
// ----------
//      ->General Includes
//      ->Linear Algebra Functions
//      ->Utilities



/**
Dataset
Esta clase se encarga de manejar todos los datos de entrada.
Toma como base el formato propuesto por el Dataset del TUM Sturm 2012

La estructura de los datos es como sigue 
	-> data
		-> DatasetName_Folder
			->rgb 				// Carpeta con las imagenes RGB (8-bit values)
			->depth 			// Carpeta con las imagenes de Profundidad (16-bit values)

			->rgb.txt			// Nombre de todas las imagenes Depth con sus respectivos Timestamps (30Hz)
			->depth.txt			// Nombre de todas las imagenes RGB con sus respectivos Timestamps (30Hz)
			->groundtruth.txt   // Lista de todas las poses y sus respectivos timestamps captados por el Motion Capture System (100Hz)

			->rgb_t.txt			// Lista Pre-procesada con las correspondencias para los Timestamps
			->depth_t.txt		// Lista Pre-procesada con las correspondencias para los Timestamps

Debido a que los datos tomados por el sensor RGB, Depth y Groundtruth no estan sincronizados, en muchas
ocasiones es necesario preprocesar la data de modo que sea posible encontrar correspondencias.
Ello se logra con el archivo 'associate_corrected.py'
*/

class Dataset
{
private:
    // Loaded images 
    // Vamos a cargar toda la secuencia de imagenes a la RAM
    // Para acelerar su procesamiento
    // Lo bueno de usar diccionarios es que podemos agregar frames teniendo como clave su verdadero indice de secuencia *
    std::map<uint32_t, cv::Mat> rgb_images_; // Usamos map por un tema de implementacion para minimizar la posibilidad de error
    std::map<uint32_t, cv::Mat> depth_images_;
    std::vector<std::string> timestamp_; // Son los timestamps finales (sincronizados) del gt
    //std::vector<Pose> poses_;
    std::map<uint32_t, Pose> poses_;

    // Hace referencia al numero de frames validos que pueden usarse para el entrenamiento
    int num_frames_ = 0;

public:
    // Nombre del Dataset sequence
    std::string dataset_path_;
    // En estos vectores iran los datos RGBD que usaremos
    // Tienen la misma cantidad de elementos
    // No necesariamente son iguales a los datos groundtruth
    std::vector<std::string> rgb_filenames_;
    std::vector<std::string> depth_filenames_;
    std::vector<std::string> timestamp_rgbd_; // Llevan los valores de RGB y no los del Depth
    std::vector<std::string> timestamp_rgbd_sync_; // Son los timestamps finales (sincronizados) del rgbd
    // Estos

    // En estos vectores almacenaremos los timestamps y Poses del groundtruth
    // Estos vectores tienen la misma cantidad de elementos
    std::vector<std::string> timestamp_groundtruth_;
    std::vector<Pose> poses_gt_;

    // Vectores especiales para realizar la Lectura del 7-Scenes Dataset
    // Como este Dataset esta partido en miles, solo guardamos los indices como
    // indicadores de que secuencias son para Training y Testing
    std::vector<int> train_sequences,test_sequences;

public:
    Dataset(std::string dataset_path); // Para leer el Dataset del TUM
    Dataset(std::string dataset_path, int dataset_type); // Para leer los datasets del 7 secenes

    // Devuelven referencias a cv::Mat y Poses correspondientes
    cv::Mat getRgbImage(int frame);
    cv::Mat getDepthImage(int frame);
    std::string getTimestamp(int frame);
    Pose getPose(int frame);

    int getNumFrames();

    // Se encarga de  agregar datos a los diccionarios y vectores que almacenan todo
    void addFrame(int num_frame, cv::Mat rgb_frame, cv::Mat depth_frame, std::string timestamp_rgbd, std::string timestamp_gt, Pose pose);

    // Busca un timestamp en base a una cadena de entrada con referencia a los datos de rgbd sincronizados
    bool check_timestamp_rgbd_match(const std::string &s, int &idx);

}; // Fin de Clase de Dataset

Pose read_pose(const std::string &pose_file);



#endif // DATASET_HPP