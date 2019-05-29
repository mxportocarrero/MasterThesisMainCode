#include "../inc/node.hpp"

template<typename D, typename RGB>
Node<D,RGB>::Node()
{
	left_ = nullptr;
	right_ = nullptr;
}

template<typename D, typename RGB>
Node<D,RGB>::~Node()
{
	delete left_;
	delete right_;
}

// Estos metodos son la escritura de la informacion de los nodos
template<typename D, typename RGB>
void Node<D,RGB>::Serialize(std::ostream& o) const
{
	Serialize_(o, is_leaf_);
	Serialize_(o, is_split_);
	Serialize_(o, depth_);
	Serialize_(o, feature_);
	Serialize_(o, mode_);
}

template<typename D, typename RGB>
void Node<D,RGB>::Deserialize(std::istream& i)
{
	Deserialize_(i, is_leaf_);
	Deserialize_(i, is_split_);
	Deserialize_(i, depth_);
	Deserialize_(i, feature_);
	Deserialize_(i, mode_);
}

template class Node<ushort,cv::Vec3b>;
