//##########################################################################
//#                                                                        #
//#                   CLOUDCOMPARE PLUGIN: qAnimation                      #
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
//#             COPYRIGHT: Ryan Wicks, 2G Robotics Inc., 2015              #
//#                                                                        #
//##########################################################################


#ifndef VIEWINTERPOLATE_H
#define VIEWINTERPOLATE_H

// CV_DB_LIB
#include <ecvViewportParameters.h>

class ecvViewportParameters;
class ccPolyline;

//! The ViewInterpolate class
/** This class takes pointers to two viewport objects, and returns intermediate viewports between over a set number of steps.
**/
class ViewInterpolate
{
public:

	//! Constructor from two viewports and a number of steps
    ViewInterpolate( const ecvViewportParameters& view1,  const ecvViewportParameters& view2, unsigned int stepCount = 0 );

	//! Sets the smooth trajectory (optional)
	void setSmoothTrajectory(	ccPolyline* smoothTrajectory,
								ccPolyline* smoothTrajectoryReversed,
								unsigned i1,
								unsigned i2,
								PointCoordinateType length);

    //! Returns the first viewport object
    inline const ecvViewportParameters& view1 () const { return m_view1; }
    // Returns the second viewport object
    inline const ecvViewportParameters& view2 () const { return m_view2; }

	//! Interpolates the 2 viewports at a given (relative) position
    bool interpolate(ecvViewportParameters& a_returned_viewport, double ratio ) const;
	
	//! Returns the next viewport
    bool nextView (ecvViewportParameters& a_returned_viewport );

    //! Returns the current step
	inline unsigned int currentStep () { return m_currentStep; }
    //! Sets the current step
	inline void setCurrentStep ( unsigned int step ) { m_currentStep = step; }

    //! Returns the max number of steps
	inline unsigned int maxStep() { return m_totalSteps; }
    //! Sets the max number of steps
	inline void setMaxStep ( unsigned int stepCount ) { m_totalSteps = stepCount; }

	//! Resets the interpolator
	inline void reset() { m_currentStep = 0; }

private:

    const ecvViewportParameters& m_view1;
    const ecvViewportParameters& m_view2;

    unsigned int m_totalSteps;
    unsigned int m_currentStep;

	ccPolyline *smoothTrajectory, *smoothTrajectoryReversed;
	unsigned smoothTrajStartIndex, smoothTrajStopIndex, smoothTrajCurrentIndex;
	PointCoordinateType smoothSegmentLength, smoothCurrentLength;

};

#endif // VIEWINTERPOLATE_H
