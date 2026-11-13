/// @ref gtc_integer
/// @file glm/gtc/integer.hpp
///
/// @see core (dependence)
/// @see gtc_integer (dependence)
///
/// @defgroup gtc_integer GLM_GTC_integer
/// @ingroup gtc
///
/// Include <glm/gtc/integer.hpp> to use the features of this extension.
///
/// @brief Allow to perform bit operations on integer values

#pragma once

// Dependencies
#include "tasks/sam3d/pbr/third_party/mip-splatting-diff-gaussian-rasterization/third_party/glm/glm/detail/setup.hpp"

#include "tasks/sam3d/pbr/third_party/mip-splatting-diff-gaussian-rasterization/third_party/glm/glm/detail/qualifier.hpp"

#include "tasks/sam3d/pbr/third_party/mip-splatting-diff-gaussian-rasterization/third_party/glm/glm/common.hpp"

#include "tasks/sam3d/pbr/third_party/mip-splatting-diff-gaussian-rasterization/third_party/glm/glm/integer.hpp"

#include "tasks/sam3d/pbr/third_party/mip-splatting-diff-gaussian-rasterization/third_party/glm/glm/exponential.hpp"

#include "tasks/sam3d/pbr/third_party/mip-splatting-diff-gaussian-rasterization/third_party/glm/glm/ext/scalar_common.hpp"

#include "tasks/sam3d/pbr/third_party/mip-splatting-diff-gaussian-rasterization/third_party/glm/glm/ext/vector_common.hpp"

#include <limits>

#if GLM_MESSAGES == GLM_ENABLE && !defined(GLM_EXT_INCLUDED)
#	pragma message("GLM: GLM_GTC_integer extension included")
#endif

namespace glm
{
	/// @addtogroup gtc_integer
	/// @{

	/// Returns the log2 of x for integer values. Useful to compute mipmap count from the texture size.
	/// @see gtc_integer
	template<typename genIUType>
	GLM_FUNC_DECL genIUType log2(genIUType x);

	/// @}
} //namespace glm

#include "integer.inl"
