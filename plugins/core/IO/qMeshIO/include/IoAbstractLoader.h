#pragma once
// MeshIO Copyright © 2019 Andy Maloney <asmaloney@gmail.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "FileIOFilter.h"


class IoAbstractLoader : public FileIOFilter
{
 public:
   bool canSave( CV_CLASS_ENUM inType, bool &outMultiple, bool &outExclusive ) const override;

   CC_FILE_ERROR loadFile( const QString &inFileName, ccHObject &ioContainer, LoadParameters &inParameters ) override;   
   
 protected:
   explicit IoAbstractLoader(const FileIOFilter::FilterInfo &info);

   virtual void _postProcess( ccHObject &ioContainer );
};
