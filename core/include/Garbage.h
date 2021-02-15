//##########################################################################
//#                                                                        #
//#                               CVCoreLib                                #
//#                                                                        #
//#  This program is free software; you can redistribute it and/or modify  #
//#  it under the terms of the GNU Library General Public License as       #
//#  published by the Free Software Foundation; version 2 or later of the  #
//#  License.                                                              #
//#                                                                        #
//#  This program is distributed in the hope that it will be useful,       #
//#  but WITHOUT ANY WARRANTY; without even the implied warranty of        #
//#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the          #
//#  GNU General Public License for more details.                          #
//#                                                                        #
//#          COPYRIGHT: EDF R&D / DAHAI LU                                 #
//#                                                                        #
//##########################################################################

#ifndef CV_GARBAGE_HEADER
#define CV_GARBAGE_HEADER

//STL
#include <unordered_set>

//! Garbage container (automatically deletes pointers when destroyed)
template<typename C> class Garbage
{
public:
	//! Puts an item in the trash
	inline void add(C* item)
	{
		try
		{
			m_items.insert(item);
		}
		catch (const std::bad_alloc&)
		{
			//what can we do?!
		}
	}

	//! Removes an item from the trash
	/** \warning The item won't be destroyed!
	**/
	inline void remove(C* item)
	{
		m_items.erase(item);
	}

	//! To manually delete an item already in the trash
	inline void destroy(C* item)
	{
		m_items.erase(item);
		delete item;
	}

	//! Destructor
	/** Automatically deletes all items
	**/
	~Garbage()
	{
		//dispose of left over
		for (auto it = m_items.begin(); it != m_items.end(); ++it)
			delete *it;
		m_items.clear();
	}

	//! Items to delete
	std::unordered_set<C*> m_items;
};


#endif //CV_GARBAGE_HEADER
