#ifndef RANDOM_H
#define RANDOM_H

#include <random>


// Redefinicion de la clase random
class Random
{
public:
    std::random_device *rd_;
    std::mt19937 *mt_;
    // Crea un Generador de numeros aleatorios usando una semilla provista por el sistema
    // Da resultados no-deterministicos
    Random()
    {
        rd_ = new std::random_device; // Semilla aleatoria
        mt_ = new std::mt19937((*rd_)()); // Engine standar para la generacion de numeros aleatorios
    }

    // Crea un numero aleatorio dentro de un rango especifico
    /// <param name="minValue">Inclusive lower bound.</param>
    /// <param name="maxValue">Inclusive upper bound.</param>
    int Next(int minValue, int maxValue)
    {
        std::uniform_int_distribution<int> dis(minValue,maxValue-1);

        return dis(*mt_);
    }


    
};

#endif // RANDOM_H