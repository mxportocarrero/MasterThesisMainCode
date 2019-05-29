#include "node.hpp"
#include "dataset.hpp"
#include "mean_shift.hpp"

#include <unordered_map>



// CLASE TREE
// ----------
// 		->Dataset
// 			->general_includes
// 			->linear_algebra_functions
// 			->utilities
// 		->Mean Shift
// 		->Nodo
//	 		->Feature // Solo es necesario agregar esta libreria para tener las otras
// 				->random
// 				->settings
// 				->general_includes // No hay conflicto por que tambien solo son cabeceras

/**
Point3D
-------
*/
class Point3D
{
public:
	double x,y,z;
	Point3D(double x, double y, double z) : x(x), y(y), z(z){};	
};

struct hashFunc
{
	size_t operator()(const Point3D &k) const{
		size_t h1 = std::hash<double>()(k.x);
		size_t h2 = std::hash<double>()(k.y);
		size_t h3 = std::hash<double>()(k.z);
		return (h1 ^ (h2 << 1)) ^ h3;
	}
};

struct equalsFunc
{
	bool operator()(const Point3D &l, const Point3D &r) const{
		return (l.x == r.x) && (l.y == r.y) && (l.z == r.z);
	}
};

typedef std::unordered_map<Point3D, uint32_t, hashFunc, equalsFunc> Point3DMap;

/**
LabeledPixel
Esta clase para representar a un solo pixel y su correspondiente valor en world coordinates
*/
class LabeledPixel{
public:
	// Constructor del Labeled Pixel
    LabeledPixel(uint32_t frame, cv::Point2i pos, cv::Point3d label):
    	frame_(frame), pos_(pos), label_(label){}
    uint32_t frame_;
    cv::Point2i pos_;
    cv::Point3d label_;	
};


// Class Tree
// ----------

template<typename D, typename RGB>
class Tree
{
private:
	Node<D, RGB> *root_;
	Dataset *data_;
	Random *random_;
	Settings *settings_;
	const char* binaryFileHeader_ = "ISUE.RelocForests.Tree";

public:
	// Constructores
	// -------------
	Tree();
	Tree(Dataset *data, Random *random, Settings *settings);
	
	Eigen::Vector3d GetLeafMode(std::vector<LabeledPixel> S);
	// Q(S_n, \theta)
	double objective_function(std::vector<LabeledPixel> data, std::vector<LabeledPixel> left, std::vector<LabeledPixel> right);
	// V(S)
	double variance(std::vector<LabeledPixel> labeled_data);

	// Training Functions
	void train_recurse(Node<D, RGB> *node, std::vector<LabeledPixel> S,const std::vector<DepthAdaptiveRGB<D, RGB>> &features );
	void Train(Dataset *data, std::vector<LabeledPixel> labeled_data, Random *random, Settings *settings, const std::vector<DepthAdaptiveRGB<D, RGB>> &features);



	// Writting and Reading Tree!!!
	// ----------------------------
	void WriteTree(std::ostream &o, Node<D,RGB> *node) const;
	void Serialize(std::ostream &stream) const;
	Node<D,RGB>* ReadTree(std::istream &i);
	Tree* Deserialize(std::istream &stream, Settings *settings);

	// Validating Tree Structure
	// -------------------------
	bool IsValidRecurse(Node<D,RGB> *node, bool prevSplit);
	bool IsValid();

	// Testing Functions Tree
	// ----------------------

	enum DECISION { LEFT, RIGHT, TRASH }; // Este Enum sirve para validar la respuesta de los pixeles

	DECISION eval_learner(DepthAdaptiveRGB<D, RGB> feature, cv::Mat depth_image, cv::Mat rgb_image, cv::Point2i pos);
	Eigen::Vector3d eval_recursive(Node<D, RGB> **node, int col, int row, cv::Mat rgb_image, cv::Mat depth_image, bool &valid);
	// Esta funcion servira para evaluar hasta llegar a un depth
	void eval_recursive(Node<D, RGB> **node, int col, int row, int depth, std::string &s, cv::Mat rgb_image, cv::Mat depth_image, bool &valid);

	// Evaluate tree at a pixel
	Eigen::Vector3d Eval(int col, int row, cv::Mat rgb_image, cv::Mat depth_image, bool &valid);


	// Very useful functions
	// ---------------------
	Node<D, RGB> *getRoot();
	Node<D, RGB> **getRootPointer();
	void printBTree(const std::string &prefix, Node<D, RGB> *node, bool isLeft);

	// Showing evaluation

}; // Fin de la clase Tree