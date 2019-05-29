#ifndef FOREST_HPP
#define FOREST_HPP

#include "tree.hpp"
#include "kabsch.hpp"

// Funcion de error Top hat, que sirve para encontrar las modas
uint tophat_error(double val);

// Funcion que devuelve pares de puntos para graficar el camera Pose
std::vector<cv::Point3d> getFrameCoordPairs(Pose pose, Settings *settings);

// CLASE HIPOTESIS
// ---------------
class Hypothesis {
public:
	//Eigen::Affine3d pose_;
	Eigen::Transform<double,3,Eigen::Affine,Eigen::DontAlign> pose_; // Tuvimos que redeclarar esta clase por q tenia error en ejecucion
	//Eigen::Vector3d camera_space_point_;
	Eigen::Matrix3Xd input_;
	Eigen::Matrix3Xd output_;
	uint energy_;

	bool operator < (const Hypothesis& h) const
	{
		return (energy_ < h.energy_);
	}
}; // Fin de la Declaracion de Hipotesis

// CLASE FOREST
// ------------

template<typename D, typename RGB>
class Forest
{
private:
	// Datos preprocesados (sincronizados)
	Dataset *data_;
	// Configuracion del algoritmo
	Settings *settings_;
	// Bosque, vector de arboles
	std::vector< Tree<D,RGB>* > forest_;
	// Candidate Features
	std::vector< DepthAdaptiveRGB<D, RGB> > candidate_features_;

	// Generador de numeros aleatorios
	Random *random_;

	// Encabezado para el archivo binario del Regresion Forest
	const char* binaryFileHeader_ = "ISUE.RelocForest.Forest";

public:
	// Inicializar bosque para entrenamiento
	Forest(Dataset *data, Settings *settings);

	// Inicializar bosque para entrenamiento con candidate features preentrenados
	Forest(Dataset *data, Settings *settings, const std::string &path,int i);

	// Inicializar bosque desde archivo binario
	Forest(Dataset *data, Settings *settings, const std::string &path);

	// Destructor
	~Forest();

	// TRAINING FUNCTIONS
	// ------------------

	// Reading Candidate Features
	std::vector< DepthAdaptiveRGB<D, RGB> > read_candidate_features(const std::string &file);

	// Generacion del LabelData
	std::vector<LabeledPixel> VerifyLabelData();
	std::vector<LabeledPixel> LabelData();

	void Train(const std::string &path);

	// EVALUATION FUNCTIONS
	// --------------------
	std::vector<Hypothesis> CreateHypothesis(int K_init, cv::Mat rgb_frame, cv::Mat depth_frame);

	std::vector<Eigen::Vector3d> Eval(int col, int row, cv::Mat rgb_image, cv::Mat depth_image);

	void Test();

	Hypothesis Test_Frame(const cv::Mat& i_ref, const cv::Mat& d_ref, double &scale);

	// Accesory Functions
	// ------------------

	bool IsValid();

	void show_tree_estimation(int depth_level);
	
};



#endif