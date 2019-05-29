#include "../inc/tree.hpp"

//#define PRINT_TREE_INFO

template<typename D, typename RGB>
Tree<D,RGB>::Tree()
{
	root_ = new Node<D, RGB>();
	root_->depth_ = 0;
};

template<typename D, typename RGB>
Tree<D,RGB>::Tree(Dataset *data, Random *random, Settings *settings) 
{
	data_ = data;
	random_ = random;
	settings_ = settings;	
}

template<typename D, typename RGB>
void Tree<D,RGB>::Train(Dataset *data, std::vector<LabeledPixel> labeled_data, Random *random, Settings *settings, const std::vector<DepthAdaptiveRGB<D, RGB>> &features) 
{
	data_ = data;
	random_ = random;
	settings_ = settings;
	train_recurse(root_, labeled_data, features);
} // Fin de la funcion Train

template<typename D, typename RGB>
void Tree<D,RGB>::train_recurse(Node<D, RGB> *node, std::vector<LabeledPixel> S, const std::vector<DepthAdaptiveRGB<D, RGB>> &features) 
{
	uint16_t height = node->depth_;

	//if (S.size() == 1 || ((height == settings_->max_tree_depth_ - 1) && S.size() >= 1)) {
	if (S.size() == 1 || ((height == settings_->max_tree_depth_ ) && S.size() >= 1)) {
		node->mode_ = GetLeafMode(S);
		node->is_leaf_ = true;
		node->left_ = nullptr;
		node->right_ = nullptr;
		return;
	} // fin de if

	//uint32_t num_candidates = 5, feature = 0; // Valores originales propuesto en la implementacion
	uint32_t num_candidates = 500, thresholds = 1, feature = 0;
	//double minimum_objective = DBL_MAX;
	//double maximum_objective = DBL_MIN;
	double maximum_objective = -DBL_MAX; // Usamos este numero para representar al numero minimo
	float threshold_final;

	std::vector< DepthAdaptiveRGB<D, RGB> > candidate_params;
	std::vector< LabeledPixel > left_final, right_final;

	for (uint32_t i = 0; i < num_candidates; ++i) {
		// add candidate
		//candidate_params.push_back(DepthAdaptiveRGB<D, RGB>::CreateRandom(random_));

		// add a randomly chosen candidate
		candidate_params.push_back(features[ random_->Next(0,features.size()) ]);

#ifdef PRINT_TREE_INFO
		std::cout << "========================================================\n";
		std::cout << "Candidate " << i << std::endl;
		std::cout << "========================================================\n";
#endif

		// Probando diferentes thresholds para cada candidato
		//for (int t = 0; t < thresholds; ++t)
		//{
			// Muestrar uniformemente los thresholds entre el MIN y MAX valor
			//candidate_params.at(i).refreshThreshold(random_);

			// partition data with candidate
			std::vector<LabeledPixel> left_data, right_data;
#ifdef PRINT_TREE_INFO
			std::cout << "Threshold " << t << ": " << candidate_params.at(i).GetThreshold() << std::endl;
#endif

			// Separando los pixeles
			for (uint32_t j = 0; j < S.size(); ++j) {
				LabeledPixel p = S.at(j);
				//std::cout << "Evaluting pixel " << j << " - Frame " << p.frame_ <<std::endl;

				DECISION val = eval_learner(candidate_params.at(i), data_->getDepthImage(p.frame_), data_->getRgbImage(p.frame_), p.pos_);

				switch (val) {
				case LEFT:
					//std::cout << "left" << std::endl;
					left_data.push_back(S.at(j));
					break;
				case RIGHT:
					//std::cout << "right" << std::endl;
					right_data.push_back(S.at(j));
					break;
				case TRASH:
					//std::cout << "invalid pixel" << std::endl;
					// do nothing
					break;
				} // Fin de switch

				//cv::waitKey(); // Activar si quieres ver como  se comportan las evaluaciones de los features

			} // Fin de For interno

			// eval tree training objective function and take best
			// todo: ensure objective function is correct
			double objective = objective_function(S, left_data, right_data);

#ifdef PRINT_TREE_INFO
			cv::Mat c = data_->getDepthImage(0);
			cv::namedWindow("Display Color",cv::WINDOW_AUTOSIZE);
			show_depth_image("Display Color",c);
			cv::waitKey();
#endif

			if (objective > maximum_objective) {
				feature = i;

				left_final = left_data;
				right_final = right_data;

				threshold_final = candidate_params.at(i).GetThreshold();

				maximum_objective = objective;
			}
		//} // Fin de FOR (testeo de varios thresholds)
	} // Fin de bucle FOR para los Feature Candidatos

#ifdef PRINT_TREE_INFO
	// printing Selected Candidate Feature Parameters
	std::cout << "========================================================\n";
	std::cout << "Selected Candidate Feature " << feature << " with Threshold "  << threshold_final << std::endl;
	std::cout << "Maximun Objetive: " << maximum_objective << std::endl;
	std::cout << "left final size " <<  left_final.size() << std::endl;
	std::cout << "Right final size " <<  right_final.size() << std::endl;

	candidate_params.at(feature).printOffsets(); std::cout << std::endl;

	std::cout << "========================================================\n";
#endif

	// split went only one way
	if (left_final.empty()) {
		node->mode_ = GetLeafMode(right_final);
		node->is_leaf_ = true;
		node->left_ = nullptr;
		node->right_ = nullptr;
		return;
	}

	if (right_final.empty()) {
		node->mode_ = GetLeafMode(left_final);
		node->is_leaf_ = true;
		node->left_ = nullptr;
		node->right_ = nullptr;
		return;
	}

	// set feature
	node->is_split_ = true;
	node->is_leaf_ = false;
	node->feature_ = candidate_params.at(feature);
	node->feature_.SetThreshold(threshold_final);
	node->left_ = new Node<D, RGB>();
	node->right_ = new Node<D, RGB>();
	node->left_->depth_ = node->right_->depth_ = node->depth_ + 1;

#ifdef PRINT_TREE_INFO
	// Imprimimos el arbol
	this->printBTree("",this->getRoot(),false);
#endif

	// Parte recursiva
	train_recurse(node->left_, left_final, features);
	train_recurse(node->right_, right_final, features);

} // Fin de la funcion train_recurse

	// V(S)
template<typename D, typename RGB>
double Tree<D,RGB>::variance(std::vector<LabeledPixel> labeled_data)
{
	if (labeled_data.size() == 0)
		return 0.0;

	double V = (1.0f / (double)labeled_data.size());
	double sum = 0.0;

	// calculate mean of S
	cv::Point3d tmp;
	for (auto p : labeled_data)
		tmp += p.label_;

	uint32_t size = labeled_data.size();
	cv::Point3d mean(tmp.x / size, tmp.y / size, tmp.z / size);

	for (auto p : labeled_data) {
		cv::Point3d val = (p.label_ - mean);
		sum += val.x * val.x  + val.y * val.y + val.z * val.z;
	}

	return V * sum;
} // Fin de variance


// Q(S_n, \theta)
template<typename D, typename RGB>
double Tree<D,RGB>::objective_function(std::vector<LabeledPixel> data, std::vector<LabeledPixel> left, std::vector<LabeledPixel> right)
{
	double var = variance(data);
	double left_var = variance(left), right_var = variance(right);
	double left_val = ((double)left.size() / (double)data.size()) * left_var;
	double right_val = ((double)right.size() / (double)data.size()) * right_var;

	double obj_function = var - (left_val + right_val);

#ifdef PRINT_TREE_INFO
	// Printing some useful info!
	std::cout.precision(7);
	std::cout << "Data Size: " << (double)data.size() <<" Variance: " << var << std::endl;
	std::cout << "Left Data Size: " << (double)left.size() <<" Variance: " << left_var << std::endl;
	std::cout << "Right Data Size: " << (double)right.size() <<" Variance: " << right_var << std::endl;

	std::cout << "Objective Function " << obj_function << std::endl;
#endif

	return obj_function;
} // Fin de la Objetive Function

template<typename D, typename RGB>
Eigen::Vector3d Tree<D,RGB>::GetLeafMode(std::vector<LabeledPixel> S)
{
	std::vector<Eigen::Vector3d> data;

	// calc mode for leaf, sub-sample N_SS = 500
	for (uint16_t i = 0; i < (S.size() < 500 ? S.size() : 500); i++) {
		auto p = S.at(i);
		Eigen::Vector3d point{ p.label_.x, p.label_.y, p.label_.z };
		data.push_back(point);
	}

	// cluster
	MeanShift ms = MeanShift(nullptr);
	double kernel_bandwidth = 0.01f; // gaussian
	std::vector<Eigen::Vector3d> cluster = ms.cluster(data, kernel_bandwidth);

	// find mode
	std::vector<Point3D> clustered_points;
	for (auto c : cluster)
	clustered_points.push_back(Point3D(floor(c[0] * 10000) / 10000,
										floor(c[1] * 10000) / 10000,
										floor(c[2] * 10000) / 10000));

	Point3DMap cluster_map;

	for (auto p : clustered_points)
		cluster_map[p]++;

	std::pair<Point3D, uint32_t> mode(Point3D(0.0, 0.0, 0.0), 0);

	for (auto p : cluster_map)
		if (p.second > mode.second)
			mode = p;

	return Eigen::Vector3d(mode.first.x, mode.first.y, mode.first.z);
} // Fin de la funcion GetLeafMode

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// 							TESTING FUNCTIONS
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

template<typename D, typename RGB>
typename Tree<D,RGB>::DECISION Tree<D,RGB>::eval_learner(DepthAdaptiveRGB<D, RGB> feature, cv::Mat depth_image, cv::Mat rgb_image, cv::Point2i pos)
{
	bool valid = true;
	float response = feature.GetResponse(depth_image, rgb_image, pos, *settings_, valid);
	//std::cout << "response: " << response << std::endl;

	if (!valid) // no depth or out of bounds
		return DECISION::TRASH;

	return (DECISION)(response >= feature.GetThreshold());
}

// Recordemos que las coordenadas en pixeles para esta funcion sera (x,y) <-> (width,height) // Cambie esta funcion a la original
template<typename D, typename RGB>
Eigen::Vector3d Tree<D,RGB>::eval_recursive(Node<D, RGB> **node, int col, int row, cv::Mat rgb_image, cv::Mat depth_image, bool &valid)
{
	if ((*node)->is_leaf_) {
		return (*node)->mode_;
	}

	DECISION val = eval_learner((*node)->feature_, depth_image, rgb_image, cv::Point2i(col, row));

	switch (val) {
		case LEFT:
			return eval_recursive(&(*node)->left_, col, row, rgb_image, depth_image, valid);
			break;
		case RIGHT:
			return eval_recursive(&(*node)->right_, col, row, rgb_image, depth_image, valid);
			break;
		case TRASH:
			valid = false;
			break;
	}
} // Fin de la funcion eval_recursive

template<typename D, typename RGB>
void Tree<D,RGB>::eval_recursive(Node<D, RGB> **node, int col, int row, int depth, std::string &s, cv::Mat rgb_image, cv::Mat depth_image, bool &valid)
{
	if ((*node)->is_leaf_ || (*node)->depth_ == depth) {
		return;
	}

	DECISION val = eval_learner((*node)->feature_, depth_image, rgb_image, cv::Point2i(col, row));

	switch (val) {
		case LEFT:
			s += "0";
			eval_recursive(&(*node)->left_, col, row, depth, s, rgb_image, depth_image, valid);
			break;
		case RIGHT:
			s += "1";
			eval_recursive(&(*node)->right_, col, row, depth, s, rgb_image, depth_image, valid);
			break;
		case TRASH:
			valid = false;
			break;
	}
} // Fin de la funcion eval_recursive

template<typename D, typename RGB>
Eigen::Vector3d Tree<D,RGB>::Eval(int col, int row, cv::Mat rgb_image, cv::Mat depth_image, bool &valid)
{
	auto m = eval_recursive(&root_, col, row, rgb_image, depth_image, valid);
	return m;
} // Fin de la función eval

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// 							USEFUL FUNCTIONS
// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


template<typename D, typename RGB>
Node<D,RGB> * Tree<D,RGB>::getRoot()	
{
	return root_;
}

template<typename D, typename RGB>
Node<D,RGB> ** Tree<D,RGB>::getRootPointer()	
{
	return &root_;
}

template<typename D, typename RGB>
void Tree<D,RGB>::printBTree(const std::string &prefix, Node<D, RGB> *node, bool isLeft)
{
	if ( node != nullptr)
	{
		std::cout << prefix;

		std::cout << (isLeft ? "├──" : "└──" );

		// print the value of the node
		std::cout.precision(4);
		if(node->is_leaf_)
			std::cout << node->depth_ << " (" << node->mode_(0)<<","<< node->mode_(1)<<","<< node->mode_(2)<<") "
					<< node->feature_.GetThreshold() << " "; 
		else
			std::cout << node->depth_ << " " << node->feature_.GetThreshold() << " ";

		node->feature_.printOffsets(); std::cout << std::endl;

		printBTree( prefix + (isLeft ? "│   " : "    "), node->left_,true);
		printBTree( prefix + (isLeft ? "│   " : "    "), node->right_,false);
	}
} // Fin de printBTree

// READ AND WRITE TREE DATA
// ------------------------
template<typename D, typename RGB>
void Tree<D, RGB>::WriteTree(std::ostream &o, Node<D,RGB> *node) const
{
	if (node == nullptr) {
		o.write("#", sizeof('#'));
		return;
	}

	node->Serialize(o);
	WriteTree(o, node->left_);
	WriteTree(o, node->right_);
} // Fin de la Funcion WriteTree

template<typename D, typename RGB>
void Tree<D,RGB>::Serialize(std::ostream &stream) const
{
	const int majorVersion = 0, minorVersion = 0;

	stream.write(binaryFileHeader_, strlen(binaryFileHeader_));
	stream.write((const char*)(&majorVersion), sizeof(majorVersion));
	stream.write((const char*)(&minorVersion), sizeof(minorVersion));

	//stream.write((const char*)(&settings_->max_tree_depth_), sizeof(settings_->max_tree_depth_));

	WriteTree(stream, root_);
}

template<typename D, typename RGB>
Node<D,RGB>* Tree<D,RGB>::ReadTree(std::istream& i)
{
	int flag = i.peek();
	char val = (char)flag;
	if (val == '#') {
		i.get();
		return nullptr;
	}

	Node<D, RGB> *tmp = new Node<D, RGB>();
	tmp->Deserialize(i);
	tmp->left_ = ReadTree(i);
	tmp->right_ = ReadTree(i);

	return tmp;
}

template<typename D, typename RGB>
Tree<D,RGB>* Tree<D,RGB>::Deserialize(std::istream& stream, Settings *settings)
{
	//settings_ = new Settings(); // aqui esta el error
	settings_ = settings;

	std::vector<char> buffer(strlen(binaryFileHeader_) + 1);
	stream.read(&buffer[0], strlen(binaryFileHeader_));
	buffer[buffer.size() - 1] = '\0';

	if (strcmp(&buffer[0], binaryFileHeader_) != 0)
		throw std::runtime_error("Unsupported forest format.");

	const int majorVersion = 0, minorVersion = 0;
	stream.read((char*)(&majorVersion), sizeof(majorVersion));
	stream.read((char*)(&minorVersion), sizeof(minorVersion));

	root_ = ReadTree(stream);
}

// Validating the tree
// Estas funciones validan si las particiones y las hojas fueran hechas correctamente
template<typename D, typename RGB>
bool Tree<D,RGB>::IsValidRecurse(Node<D, RGB> *node, bool prevSplit)
{
	if (!node && prevSplit)
		return false;

	if (node->is_leaf_)
		return true;

	return IsValidRecurse(node->left_, node->is_split_) && IsValidRecurse(node->right_, node->is_split_);
}

template<typename D, typename RGB>
bool Tree<D,RGB>::IsValid()
{
	return IsValidRecurse(root_, true);
}

template class Tree<ushort,cv::Vec3b>;