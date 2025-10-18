// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "LasVlr.h"

#include "LasMetadata.h"

// Qt
#include <QtGlobal>
// qCC_db
#include <ecvPointCloud.h>
// System
#include <algorithm>
#include <cstring>

LasVlr::LasVlr(const laszip_header& header)
{
	const auto vlrShouldBeCopied = [](const laszip_vlr_struct& vlr)
	{
		return !LasDetails::IsLaszipVlr(vlr) && !LasDetails::IsExtraBytesVlr(vlr);
	};

	ptrdiff_t numVlrs = std::count_if(header.vlrs, header.vlrs + header.number_of_variable_length_records, vlrShouldBeCopied);
	if (numVlrs > 0)
	{
		vlrs.resize(numVlrs);
		laszip_U32 j = 0;
		for (laszip_U32 i = 0; i < header.number_of_variable_length_records; ++i)
		{
			if (vlrShouldBeCopied(header.vlrs[i]))
			{
				LasDetails::CloneVlrInto(header.vlrs[i], vlrs[j]);
				j++;
			}
		}
	}
}

LasVlr& LasVlr::operator=(LasVlr rhs)
{
	LasVlr::Swap(*this, rhs);
	return *this;
}

LasVlr::LasVlr(const LasVlr& rhs)
    : extraScalarFields(rhs.extraScalarFields)
{
	if (rhs.numVlrs() != 0)
	{
		vlrs.resize(rhs.numVlrs());
		for (laszip_U32 i = 0; i < rhs.numVlrs(); ++i)
		{
			LasDetails::CloneVlrInto(rhs.vlrs[i], vlrs[i]);
		}
	}
}

void LasVlr::Swap(LasVlr& lhs, LasVlr& rhs) noexcept
{
	std::swap(lhs.vlrs, rhs.vlrs);
	std::swap(lhs.extraScalarFields, rhs.extraScalarFields);
}
