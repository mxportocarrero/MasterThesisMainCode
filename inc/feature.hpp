#pragma once

#include "general_includes.hpp"
#include "settings.hpp"
#include "random.hpp"
#include "utilities.hpp"


// Clase Feature
// -------------
// 		->random
// 		->settings
// 		->general_includes // No hay conflicto por que tambien solo son cabeceras

// Esta forma parte de los nodos


class Feature
{
//protected:
public:
	cv::Point2i offset_1_;
	cv::Point2i offset_2_;
public:
	Feature();
	Feature(cv::Point2i offset_1, cv::Point2i offset_2) : offset_1_(offset_1), offset_2_(offset_2){};

	virtual float GetResponse(cv::Mat depth_image, cv::Mat rgb_image, cv::Point2i pos, Settings &settings, bool &valid) = 0;	
};

// Especializacion de funcionces

// DEPTH FEATURE RESPONSE FUNCTIONS
// --------------------------------

// La razon por la que lo declaramos en forma de template
// Es por q se hace mas facil si quisieras cambiar y usar otro tipo de features
template <typename D, typename RGB>
class Depth : public Feature
{
public:
	Depth(cv::Point2i offset_1, cv::Point2i offset_2) : Feature(offset_1, offset_2){};

	/** Funcion encargada de devolver el valor response
	Input:
		el par RGBD, punto en pixel coordinates y una referencia a un bool para validar la operacion
	Output:
		double con el valor calculado, segun formula. revisar el paper
	*/
	virtual float GetResponse(cv::Mat depth_image, cv::Mat rgb_image, cv::Point2i pos, bool &valid);

}; // Fin de la definicion de Clase Depth

template <typename D, typename RGB>
class DepthAdaptiveRGB : public Feature
{
//protected:
public:
	int color_channel_1_, color_channel_2_;
	float tau_;	
public:	
	DepthAdaptiveRGB();

	DepthAdaptiveRGB(cv::Point2i offset_1, cv::Point2i offset_2, int color_channel_1, int color_channel_2, float tau)
		: Feature(offset_1, offset_2), color_channel_1_(color_channel_1), color_channel_2_(color_channel_2), tau_(tau){};

	static DepthAdaptiveRGB CreateRandom(Random *random);

	virtual float GetResponse(cv::Mat depth_img, cv::Mat rgb_img, cv::Point2i pos, Settings &settings, bool &valid) override;

	float GetThreshold();

	void SetThreshold(const float &t);

	// Variamos el valor de nuestra variable Tau
	void refreshThreshold(Random *random);

	//no deja espacio cuando imprime
	void printOffsets();


}; // Fin de la Declaracion de la Clase Depth AdaptiveRGB







