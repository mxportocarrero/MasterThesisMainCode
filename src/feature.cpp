#include "../inc/feature.hpp"

Feature::Feature()
{
	offset_1_ = cv::Point2i(0,0);
	offset_2_ = cv::Point2i(0,0);
}

// Depth GetResponse Function
// --------------------------
template <typename D, typename RGB>
float Depth<D,RGB>::GetResponse(cv::Mat depth_image, cv::Mat rgb_image, cv::Point2i pos, bool &valid)
{
	D depth_at_pos = depth_image.at<D>(pos);
	cv::Point2i depth_inv_1(offset_1_.x / depth_at_pos, offset_2_.y / depth_at_pos);
	cv::Point2i depth_inv_2(offset_2_.x / depth_at_pos, offset_2_.y / depth_at_pos);

	if (depth_at_pos == 0)
		valid = false;

	D D_1 = depth_image.at<D>(pos + depth_inv_1);
	D D_2 = depth_image.at<D>(pos + depth_inv_2);

	return D_1 - D_2;
}

// Depth Adaptive RGB
// ------------------
template <typename D, typename RGB>
DepthAdaptiveRGB<D,RGB>::DepthAdaptiveRGB()
{
	color_channel_1_ = 0;
	color_channel_2_ = 0;
	tau_ = 0;
} // Fin de Contructor Default

template <typename D, typename RGB>
DepthAdaptiveRGB<D,RGB> DepthAdaptiveRGB<D,RGB>::CreateRandom(Random *random)
{
	cv::Point2i offset_1(random->Next(-130, 130), random->Next(-130, 130)); // Value from the paper -- +/- 130 pixel meters
	cv::Point2i offset_2(random->Next(-130, 130), random->Next(-130, 130));
	int color_channel_1 = random->Next(0, 3); // es del 0 al 3 por que excluye el ultimo el upper boundary(cambiamos esto)
	int color_channel_2 = random->Next(0, 3);
	int tau = random->Next(-250, 250); // Revisar esto en el paper !!!!
	// Segun lo que dice el paper este valor debe estar entre los valores maximos y minimos
	return DepthAdaptiveRGB(offset_1, offset_2, color_channel_1, color_channel_2, tau);
}

template <typename D, typename RGB>
float DepthAdaptiveRGB<D,RGB>::GetResponse(cv::Mat depth_img, cv::Mat rgb_img, cv::Point2i pos, Settings &settings, bool &valid)
{
	D depth_at_pos = depth_img.at<D>(pos);
	float depth = (float)depth_at_pos;
	//std::cout << "pixel depth: " << depth_at_pos << std::endl;
	//std::cout << "depth factor: " << settings.depth_factor_ << ":" << (float)settings.depth_factor_ << std::endl;

	if (depth <= 0) {
		valid = false;
		return 0.0;
	} else {
		depth /= (float)settings.depth_factor_; // scale value
	}

	cv::Point2i depth_inv_1(offset_1_.x / depth, offset_1_.y / depth);
	cv::Point2i depth_inv_2(offset_2_.x / depth, offset_2_.y / depth);

	cv::Point2i pos1 = pos + depth_inv_1;
	cv::Point2i pos2 = pos + depth_inv_2;

	int width = settings.image_width_;
	int height = settings.image_height_;

	/*// depth invariance
	// Este codigo sirve para observar como se comportan los offsets
	std::cout << "===============================\n";
	std::cout << "pixel depth: " << depth << std::endl;

	std::cout << "Central pixel: " << pos.x << "," << pos.y << "\n";

	std::cout << "Offset 1: " << offset_1_.x << "," << offset_1_.y << " Offset 2: " << offset_2_.x << "," << offset_2_.y << std::endl;
	std::cout << "Depth_inv_1: " << depth_inv_1.x << "," << depth_inv_1.y << " Depth_inv_2: " << depth_inv_2.x << "," << depth_inv_2.y << std::endl;
	std::cout << "pos1: " << pos1.x << "," << pos1.y << " pos2: " << pos2.x << "," << pos2.y << std::endl;
	std::cout << "===============================\n";

	cv::namedWindow("Display Color",cv::WINDOW_AUTOSIZE);
	cv::namedWindow("Display Depth",cv::WINDOW_AUTOSIZE);

	cv::Mat c = rgb_img.clone();
	cv::Mat m = depth_img.clone();

	// Centro
	cv::circle(m,pos,5,cv::Scalar(255,255,255));
	// Offsets
	cv::circle(m,pos1,10,cv::Scalar(255,0,0));
	cv::circle(m,pos2, 10,cv::Scalar(255,0,0));
	cv::circle(m,pos,130,cv::Scalar(255,0,255));

	// Centro
	cv::circle(c,pos,5,cv::Scalar(255,255,255));
	// Offsets
	cv::circle(c,pos1,10,cv::Scalar(255,0,0));
	cv::circle(c,pos2, 10,cv::Scalar(255,0,0));
	cv::circle(c,pos,130,cv::Scalar(255,0,255));

	show_depth_image("Display Color",c);
	show_depth_image("Display Depth",m);
	//cv::waitKey();
	
	*/

	// check bounds
	if (pos1.x >= width || pos1.y >= height ||
		pos1.x < 0.0    || pos1.y < 0.0 ) {
		valid = false;
		return 0.0f;
	}
	if (pos2.x >= width || pos2.y >= height ||
		pos2.x < 0.0    || pos2.y < 0.0) {
		valid = false;
		return 0.0f;
	}

	float I_1 = rgb_img.at<RGB>(pos1)[this->color_channel_1_];
	float I_2 = rgb_img.at<RGB>(pos2)[this->color_channel_2_];

	//std::cout << "I1(" << this->color_channel_1_ << "): " << I_1 << ", I2(" << this->color_channel_2_ << "): " << I_2 << std::endl;

	return I_1 - I_2;
} // Fin de la Funcion GetResponse

template <typename D, typename RGB>
float DepthAdaptiveRGB<D,RGB>::GetThreshold()
{
	return tau_;
} // Fin de la funcion Get Threshold

template <typename D, typename RGB>
void DepthAdaptiveRGB<D,RGB>::SetThreshold(const float &t)
{
	tau_ = t;
}

// Variamos el valor de nuestra variable Tau
template <typename D, typename RGB>
void DepthAdaptiveRGB<D,RGB>::refreshThreshold(Random *random)
{
	int tau = random->Next(-250, 250);
	tau_ = (float)tau;
}

//no deja espacio cuando imprime
template <typename D, typename RGB>
void DepthAdaptiveRGB<D,RGB>::printOffsets()
{
	std::cout << "offset1: " << offset_1_.x << ","<< offset_1_.y
			  << " offset2: " << offset_2_.x << ","<< offset_2_.y;
}

template class DepthAdaptiveRGB<ushort,cv::Vec3b>;



