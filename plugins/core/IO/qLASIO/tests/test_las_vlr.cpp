// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "LasVlr.h"

#include <QBuffer>
#include <gtest/gtest.h>
#include <limits>

TEST(LasVlrSerialization, EmptyPayloadRoundTrips)
{
	LasVlr      source;
	QByteArray  bytes;
	QDataStream out(&bytes, QIODevice::WriteOnly);
	out.setVersion(QDataStream::Qt_5_12);
	out << source;
	QDataStream in(bytes);
	in.setVersion(QDataStream::Qt_5_12);
	LasVlr restored;
	in >> restored;
	EXPECT_EQ(QDataStream::Ok, in.status());
	EXPECT_TRUE(restored.vlrs.empty());
	EXPECT_TRUE(restored.extraScalarFields.empty());
}

TEST(LasVlrSerialization, UnboundedCountsAreRejectedBeforeAllocation)
{
	const quint64 huge = (std::numeric_limits<quint64>::max)();
	for (bool extraFields : {false, true})
	{
		QByteArray  bytes;
		QDataStream out(&bytes, QIODevice::WriteOnly);
		if (extraFields)
			out << quint64(0);
		out << huge;
		QDataStream in(bytes);
		LasVlr      restored;
		in >> restored;
		EXPECT_NE(QDataStream::Ok, in.status());
		EXPECT_TRUE(restored.vlrs.empty());
		EXPECT_TRUE(restored.extraScalarFields.empty());
	}
}

TEST(LasVlrSerialization, TruncatedRecordIsRejectedBeforePayloadAllocation)
{
	QByteArray  bytes;
	QDataStream out(&bytes, QIODevice::WriteOnly);
	out << quint64(1) << quint16(0);
	out.writeRawData(QByteArray(16, '\0').constData(), 16);
	out << quint16(7) << quint16(100);
	out.writeRawData(QByteArray(32, '\0').constData(), 32);
	QDataStream in(bytes);
	LasVlr      restored;
	in >> restored;
	EXPECT_NE(QDataStream::Ok, in.status());
	EXPECT_TRUE(restored.vlrs.empty());
}

TEST(LasVlrSerialization, VlrBytesAndExtraRecordsRoundTrip)
{
	LasVlr            source;
	QByteArray        payload("A\0B", 3);
	laszip_vlr_struct record{};
	record.record_id                  = 7;
	record.record_length_after_header = payload.size();
	record.data                       = reinterpret_cast<laszip_U8*>(payload.data());
	source.vlrs.push_back(record);
	source.extraScalarFields.emplace_back();
	source.extraScalarFields[0].scalarFields[0] = reinterpret_cast<ccScalarField*>(quintptr(0x1234));
	QByteArray  bytes;
	QDataStream out(&bytes, QIODevice::WriteOnly);
	out.setVersion(QDataStream::Qt_5_12);
	out << source;
	QDataStream in(bytes);
	in.setVersion(QDataStream::Qt_5_12);
	LasVlr restored;
	in >> restored;
	ASSERT_EQ(QDataStream::Ok, in.status());
	ASSERT_EQ(1, restored.vlrs.size());
	EXPECT_EQ(7, restored.vlrs[0].record_id);
	EXPECT_EQ(payload, QByteArray(reinterpret_cast<char*>(restored.vlrs[0].data), restored.vlrs[0].record_length_after_header));
	ASSERT_EQ(1, restored.extraScalarFields.size());
	EXPECT_EQ(nullptr, restored.extraScalarFields[0].scalarFields[0]);
	delete[] restored.vlrs[0].data;
}
