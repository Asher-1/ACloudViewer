//##########################################################################
//#                                                                        #
//#                              CLOUDVIEWER                               #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 or later of the License.      #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

#ifndef ECV_COLOR_TYPES_HEADER
#define ECV_COLOR_TYPES_HEADER

//Local
#include "eCV_db.h"

#include <Eigen/Core>

//Qt
#include <QColor>

//system
#include <cmath>
#include <random>
#include <algorithm>
#include <type_traits>

#include <Console.h>
#include <iostream>

//! Default color components type (R,G and B)
typedef unsigned char ColorCompType;

//! Colors namespace
namespace ecvColor
{
	//! Max value of a single color component (default type)
	const ColorCompType MAX = 255;
	const ColorCompType OPACITY = 255;

	template<typename T1, typename T2>
	struct is_same_type
	{
		operator bool()
		{
			return false;
		}
	};

	template<typename T1>
	struct is_same_type<T1, T1>
	{
		operator bool()
		{
			return true;
		}
	};

	//! RGB color structure
	template <typename Type> class RgbTpl
	{
	public:
	
		//! 3-tuple as a union
		union
		{
			struct
			{
				Type r,g,b;
			};
			Type rgb[3];
		};

		//! Default constructor
		/** Inits color to (0,0,0).
		**/
		inline RgbTpl() : r(0), g(0), b(0) {}

		//! Constructor from a triplet of r,g,b values
		explicit inline RgbTpl(Type red, Type green, Type blue) : r(red), g(green), b(blue) {}

		//! Constructor from an array of 3 values
		explicit inline RgbTpl(const Type col[3]) : r(col[0]), g(col[1]), b(col[2]) {}

		inline static Eigen::Vector3d ToEigen(const Type col[3]) { 
			if (is_same_type<Type, float>())
			{
				return Eigen::Vector3d(col[0], col[1], col[2]);
			}
			else
			{
				bool sameType = is_same_type<Type, ColorCompType>();
				assert(sameType);
				return Eigen::Vector3d(col[0] / 255.0, col[1] / 255.0, col[2] / 255.0);
			}
		}
		inline static Eigen::Vector3d ToEigen(const RgbTpl<Type>& t) { return ToEigen(t.rgb); }
		inline static RgbTpl FromEigen(const Eigen::Vector3d& t) {
			Type newCol[3];
			if (is_same_type<Type, float>())
			{
				newCol[0] = static_cast<Type>(std::min(1.0, std::max(0.0, t(0))));
				newCol[1] = static_cast<Type>(std::min(1.0, std::max(0.0, t(1))));
				newCol[2] = static_cast<Type>(std::min(1.0, std::max(0.0, t(2))));
			}
			else
			{
				if (t(0) > 1 || t(1) > 1 || t(2) > 1)
				{
					CVLib::utility::LogWarning("[ecvColor] Find invalid color: ");
					std::cout << t << std::endl;
				}
				
				newCol[0] = static_cast<Type>(std::min(255.0, std::max(0.0, t(0) * MAX)));
				newCol[1] = static_cast<Type>(std::min(255.0, std::max(0.0, t(1) * MAX)));
				newCol[2] = static_cast<Type>(std::min(255.0, std::max(0.0, t(2) * MAX)));
			}

			return RgbTpl(newCol[0], newCol[1], newCol[2]); 
		}

		//! Direct coordinate access
		inline Type& operator () (unsigned i) { return rgb[i]; }
		//! Direct coordinate access (const)
		inline const Type& operator () (unsigned i) const { return rgb[i]; }

		//! In-place addition operator
		inline RgbTpl& operator += (const RgbTpl<Type>& c) { r += c.r; g += c.g; b += c.b; return *this; }
		//! In-place subtraction operator
		inline RgbTpl& operator -= (const RgbTpl<Type>& c) { r -= c.r; g -= c.g; b -= c.b; return *this; }
		//! Comparison operator
		inline bool operator != (const RgbTpl<Type>& t) const { return (r != t.r || g != t.g || b != t.b); }
	};

	//! 3 components, float type
	typedef RgbTpl<float> Rgbf;
	//! 3 components, unsigned byte type
	typedef RgbTpl<unsigned char> Rgbub;
	//! 3 components, default type
	typedef RgbTpl<ColorCompType> Rgb;

	//! RGBA color structure
	template <class Type> class RgbaTpl
	{
	public:
	
		// 4-tuple values as a union
		union
		{
			struct
			{
				Type r,g,b,a;
			};
			Type rgba[4];
		};

		//! Default constructor
		/** Inits color to (0,0,0,0).
		**/
		inline RgbaTpl() : r(0), g(0), b(0), a(0) {}

		//! Constructor from a triplet of r,g,b values and a transparency value
		explicit inline RgbaTpl(Type red, Type green, Type blue, Type alpha) : r(red), g(green), b(blue), a(alpha) {}

		//! RgbaTpl from an array of 4 values
		explicit inline RgbaTpl(const Type col[4]) : r(col[0]), g(col[1]), b(col[2]), a(col[3]) {}
		//! RgbaTpl from an array of 3 values and a transparency value
		explicit inline RgbaTpl(const Type col[3], Type alpha) : r(col[0]), g(col[1]), b(col[2]), a(alpha) {}
	
		//! Copy constructor
		inline RgbaTpl(const RgbTpl<Type>& c, Type alpha) : r(c.r), g(c.g), b(c.b), a(alpha) {}

		//! Cast operator
		inline operator RgbTpl<Type>() const { return RgbTpl<Type>(rgba); }
		//! Cast operator (const version)
		//inline operator const Type*() const { return rgba; }

		//! Comparison operator
		inline bool operator != (const RgbaTpl<Type>& t) const { return (r != t.r || g != t.g || b != t.b || a != t.a); }
	};

	//! 4 components, float type
	typedef RgbaTpl<float> Rgbaf;
	//! 4 components, unsigned byte type
	typedef RgbaTpl<unsigned char> Rgbaub;
	//! 4 components, default type
	typedef RgbaTpl<ColorCompType> Rgba;

	// Predefined colors (default type)
	ECV_DB_LIB_API extern const Rgb white;
	ECV_DB_LIB_API extern const Rgb lightGrey;
	ECV_DB_LIB_API extern const Rgb darkGrey;
	ECV_DB_LIB_API extern const Rgb red;
	ECV_DB_LIB_API extern const Rgb green;
	ECV_DB_LIB_API extern const Rgb blue;
	ECV_DB_LIB_API extern const Rgb darkBlue;
	ECV_DB_LIB_API extern const Rgb magenta;
	ECV_DB_LIB_API extern const Rgb cyan;
	ECV_DB_LIB_API extern const Rgb orange;
	ECV_DB_LIB_API extern const Rgb black;
	ECV_DB_LIB_API extern const Rgb yellow;

	ECV_DB_LIB_API extern const Rgba owhite;
	ECV_DB_LIB_API extern const Rgba olightGrey;
	ECV_DB_LIB_API extern const Rgba odarkGrey;
	ECV_DB_LIB_API extern const Rgba ored;
	ECV_DB_LIB_API extern const Rgba ogreen;
	ECV_DB_LIB_API extern const Rgba oblue;
	ECV_DB_LIB_API extern const Rgba odarkBlue;
	ECV_DB_LIB_API extern const Rgba omagenta;
	ECV_DB_LIB_API extern const Rgba ocyan;
	ECV_DB_LIB_API extern const Rgba oorange;
	ECV_DB_LIB_API extern const Rgba oblack;
	ECV_DB_LIB_API extern const Rgba oyellow;

	// Predefined materials (float)
	ECV_DB_LIB_API extern const Rgbaf bright;
	ECV_DB_LIB_API extern const Rgbaf lighter;
	ECV_DB_LIB_API extern const Rgbaf light;
	ECV_DB_LIB_API extern const Rgbaf middle;
	ECV_DB_LIB_API extern const Rgbaf dark;
	ECV_DB_LIB_API extern const Rgbaf darker;
	ECV_DB_LIB_API extern const Rgbaf darkest;
	ECV_DB_LIB_API extern const Rgbaf night;
	ECV_DB_LIB_API extern const Rgbaf defaultMeshFrontDiff;
	ECV_DB_LIB_API extern const Rgbaf defaultMeshBackDiff;
	ECV_DB_LIB_API extern const Rgbf defaultViewBkgColor;

	// Default foreground color (unsigned byte)
	ECV_DB_LIB_API extern const Rgbub defaultColor;				//white
	ECV_DB_LIB_API extern const Rgbub defaultBkgColor;			//dark blue
	ECV_DB_LIB_API extern const Rgbub defaultLabelBkgColor;		//white
	ECV_DB_LIB_API extern const Rgbub defaultLabelMarkerColor;	//magenta



	//! Colors generator
	class Generator
	{
	public:
		
		//! Generates a random color
		static Rgb Random(bool lightOnly = true)
		{
			std::random_device rd;   // non-deterministic generator
			std::mt19937 gen(rd());  // to seed mersenne twister.
			std::uniform_int_distribution<unsigned> dist(0, MAX);

			Rgb col;
			col.r = dist(gen);
			col.g = dist(gen);
			if (lightOnly)
			{
				col.b = MAX - static_cast<ColorCompType>((static_cast<double>(col.r) + static_cast<double>(col.g)) / 2); //cast to double to avoid overflow (whatever the type of ColorCompType!!!)
			}
			else
			{
				col.b = dist(gen);
			}

			return col;
		}
	};

	//! Color space conversion
	class Convert
	{
	public:

		//! Converts a HSL color to RGB color space
		/** \param H [out] hue [0;360[
			\param S [out] saturation [0;1]
			\param L [out] light [0;1]
			\return RGB color (unsigned byte)
		**/
		static Rgb hsl2rgb(float H, float S, float L)
		{
			H /= 360;
			float q = L < 0.5f ? L * (1.0f + S) : L + S - L * S;
			float p = 2 * L - q;

			float r = hue2rgb(p, q, H + 1.0f / 3.0f);
			float g = hue2rgb(p, q, H);
			float b = hue2rgb(p, q, H - 1.0f / 3.0f);

			return Rgb(	static_cast<ColorCompType>(r * ecvColor::MAX),
						static_cast<ColorCompType>(g * ecvColor::MAX),
						static_cast<ColorCompType>(b * ecvColor::MAX));

		}

		//! Converts a HSV color to RGB color space
		/** \param H [out] hue [0;360[
			\param S [out] saturation [0;1]
			\param V [out] value [0;1]
			\return RGB color (unsigned byte)
		**/
		static Rgb hsv2rgb(float H, float S, float V)
		{
			float hi = 0;
			float f = std::modf(H / 60.0f, &hi);

			float l = V*(1.0f - S);
			float m = V*(1.0f - f*S);
			float n = V*(1.0f - (1.0f - f)*S);

			Rgbf rgb(0, 0, 0);

			switch (static_cast<int>(hi) % 6)
			{
			case 0:
				rgb.r = V; rgb.g = n; rgb.b = l;
				break;
			case 1:
				rgb.r = m; rgb.g = V; rgb.b = l;
				break;
			case 2:
				rgb.r = l; rgb.g = V; rgb.b = n;
				break;
			case 3:
				rgb.r = l; rgb.g = m; rgb.b = V;
				break;
			case 4:
				rgb.r = n; rgb.g = l; rgb.b = V;
				break;
			case 5:
				rgb.r = V; rgb.g = l; rgb.b = m;
				break;
			}

			return Rgb (static_cast<ColorCompType>(rgb.r * ecvColor::MAX),
						static_cast<ColorCompType>(rgb.g * ecvColor::MAX),
						static_cast<ColorCompType>(rgb.b * ecvColor::MAX));
		}

	protected:

		//! Method used by hsl2rgb
		static float hue2rgb(float m1, float m2, float hue)
		{
			if (hue < 0)
				hue += 1.0f;
			else if (hue > 1.0f)
				hue -= 1.0f;

			if (6 * hue < 1.0f)
				return m1 + (m2 - m1) * hue * 6;
			else if (2 * hue < 1.0f)
				return m2;
			else if (3 * hue < 2.0f)
				return m1 + (m2 - m1) * (4.0f - hue * 6);
			else
				return m1;
		}
	};

	namespace LookUpTable
	{
		ECV_DB_LIB_API Rgb at(size_t color_id);
	}

	//! Conversion from Rgbf
	inline Rgb FromRgbf(const Rgbf& color) { return Rgb(static_cast<ColorCompType>(color.r * MAX),
														static_cast<ColorCompType>(color.g * MAX),
														static_cast<ColorCompType>(color.b * MAX)); }

	//! Conversion from Rgbaf
	inline Rgb FromRgbf(const Rgbaf& color) { return Rgb(static_cast<ColorCompType>(color.r * MAX),
														 static_cast<ColorCompType>(color.g * MAX),
														 static_cast<ColorCompType>(color.b * MAX)); }

	inline Rgbf FromRgb(const Rgb& color) { return Rgbf(static_cast<float>(1.0 * color.r / MAX),
														static_cast<float>(1.0 * color.g / MAX),
														static_cast<float>(1.0 * color.b / MAX)); }

	inline Rgbf FromRgb(const Rgba& color) { return Rgbf(static_cast<float>(1.0 * color.r / MAX),
														static_cast<float>(1.0 * color.g / MAX),
														static_cast<float>(1.0 * color.b / MAX)); }

	inline Rgbaf FromRgba(const Rgba& color) { return Rgbaf(static_cast<float>(1.0 * color.r / MAX),
															static_cast<float>(1.0 * color.g / MAX),
															static_cast<float>(1.0 * color.b / MAX),
															static_cast<float>(1.0 * color.a / MAX)); }
	inline Rgbaf FromRgbub(const Rgbub& color) {
		return Rgbaf(static_cast<float>(1.0 * color.r / MAX),
			static_cast<float>(1.0 * color.g / MAX),
			static_cast<float>(1.0 * color.b / MAX),
			1.0f);
	}

	//! Conversion from QRgb
	inline Rgb FromQRgb(QRgb qColor) { return Rgb(	static_cast<unsigned char>(qRed(qColor)),
													static_cast<unsigned char>(qGreen(qColor)),
													static_cast<unsigned char>(qBlue(qColor))); }

	//! Conversion from QRgb'a'
	inline Rgba FromQRgba(QRgb qColor) { return Rgba(	static_cast<unsigned char>(qRed(qColor)),
														static_cast<unsigned char>(qGreen(qColor)),
														static_cast<unsigned char>(qBlue(qColor)),
														static_cast<unsigned char>(qAlpha(qColor))); }

	//! Conversion from QColor
	inline Rgb FromQColor(QColor qColor) { return Rgb(	static_cast<unsigned char>(qColor.red()),
														static_cast<unsigned char>(qColor.green()),
														static_cast<unsigned char>(qColor.blue())); }

	//! Conversion from QColor'a'
	inline Rgba FromQColora(QColor qColor) { return Rgba(	static_cast<unsigned char>(qColor.red()),
															static_cast<unsigned char>(qColor.green()),
															static_cast<unsigned char>(qColor.blue()),
															static_cast<unsigned char>(qColor.alpha())); }

	//! Conversion from QColor (floating point)
	inline Rgbf FromQColorf(QColor qColor) { return Rgbf(	static_cast<float>(qColor.redF()),
															static_cast<float>(qColor.greenF()),
															static_cast<float>(qColor.blueF())); }
	//! Conversion from QColor'a' (floating point)
	inline Rgbaf FromQColoraf(QColor qColor) { return Rgbaf(	static_cast<float>(qColor.redF()),
																static_cast<float>(qColor.greenF()),
																static_cast<float>(qColor.blueF()),
																static_cast<float>(qColor.alphaF()));
	}

};

#endif //CC_COLOR_TYPES_HEADER
