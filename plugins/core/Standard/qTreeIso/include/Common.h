#pragma once

#include <string>
#include <sstream>
#include <ctime>
#include <functional>
#include <stdio.h>
#include <vector>

namespace patch
{
	template < typename T > std::string to_string(const T& n)
	{
		std::ostringstream stm;
		stm << n;
		return stm.str();
	}
}

enum fidelityType { L2, linear, KL, SPG };

struct GenericParameter
{
	GenericParameter(std::string inName = "in_name", double reg_strength = 0, double fidelity = 0)
	{
		this->in_name = inName;
		char* buffer = new char[inName.size() + 10];
		//char buffer [inName.size() + 10];
		std::string extension = inName.substr(inName.find_last_of(".") + 1);
		this->extension = extension;
		std::string baseName = inName.substr(0, inName.size() - extension.size() - 1);
		this->base_name = baseName;
		sprintf(buffer, "%s_out_%1.0f_%.0f.%s", baseName.c_str(), fidelity, reg_strength * 1000, extension.c_str());
		this->out_name = std::string(buffer);
		this->natureOfData = 0;
		this->fidelity = L2;
	}

	std::string in_name, out_name, base_name, extension;
	int natureOfData;
	fidelityType fidelity;
};

class TimeStack
{
public:
	TimeStack()
		: lastTime{}
	{}

	void tic()
	{
		lastTime = clock();
	}

	std::string toc() const
	{
		std::ostringstream stm;
		stm << static_cast<double>(clock() - lastTime) / CLOCKS_PER_SEC;
		return stm.str();
	}

	double tocDouble() const
	{
		double x = static_cast<double>(clock() - lastTime) / CLOCKS_PER_SEC;
		return x;
	}

protected:
	clock_t lastTime;

};

template<typename T>
struct ComponentsFusion
{//this class encode a potential fusion between two cadjacent component
 //and is ordered wrt the merge_gain
	ComponentsFusion(std::size_t c1, std::size_t c2, std::size_t ind = 0, T gain = 0.)
	{
		this->comp1 = c1;
		this->comp2 = c2;
		this->border_index = ind;
		this->merge_gain = gain;
	}

	std::size_t comp1, comp2; //index of the components
	std::size_t border_index; //index of the border-edge
	T merge_gain; //gain obtained by mergeing the components
	std::vector<T> merged_value; //value of the new components when they are merged
};

template<typename T>
struct lessComponentsFusion
{
    bool operator()(const ComponentsFusion<T> lhs, const ComponentsFusion<T> rhs) const
    {
        return lhs.merge_gain < rhs.merge_gain;
    }
};


template<typename T>
class VectorOfCentroids
{
    //VectorOfCentroids is a vector of size k x 2 x d where k is the number of components and
    // d the dimension of the observation
public:
    std::vector< std::vector< std::vector<T> > > centroids;
    VectorOfCentroids(std::size_t nb_comp, std::size_t dim)
    {
        this->centroids = std::vector< std::vector< std::vector<T> > >(nb_comp,
            std::vector< std::vector<T> >(2, std::vector<T>(dim, 0.0)));
    }
};
template<typename T>
class Point3D
{
public:
    T x,y,z;
    Point3D(T x = 0., T y = 0., T z = 0.)
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }
};

template<typename T>
struct lessPoint3D
{
    bool operator()(const Point3D<T> lhs, const Point3D<T> rhs) const
    {
        if (lhs.x != rhs.x)
        {
            return lhs.x < rhs.x;
        }
        if (lhs.y != rhs.y)
        {
            return lhs.y < rhs.y;
        }
        if (lhs.z > rhs.z)
        {
            return lhs.z < rhs.z;
        }
        return true;
    }
};
