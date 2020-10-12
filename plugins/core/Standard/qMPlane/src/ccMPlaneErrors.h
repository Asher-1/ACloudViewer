//##########################################################################
//#                                                                        #
//#                    CLOUDVIEWER PLUGIN: qMPlane                         #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU General Public License as published by  #
//#  the Free Software Foundation; version 2 of the License.               #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#         COPYRIGHT: AIT Austrian Institute of Technology GmbH           #
//#                                                                        #
//##########################################################################


#ifndef CC_MPLANE_ERRORS
#define CC_MPLANE_ERRORS

// std
#include <stdexcept>


class MplaneInvalidArgument : public std::invalid_argument {
	using std::invalid_argument::invalid_argument;
};

class MplaneFittingError : public std::logic_error {
	using std::logic_error::logic_error;
};

#endif
