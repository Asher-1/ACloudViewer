// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#ifndef CV_GARBAGE_HEADER
#define CV_GARBAGE_HEADER

// STL
#include <unordered_set>

//! Garbage container (automatically deletes pointers when destroyed)
template <typename C>
class Garbage {
public:
    //! Puts an item in the trash
    inline void add(C* item) {
        try {
            m_items.insert(item);
        } catch (const std::bad_alloc&) {
            // what can we do?!
        }
    }

    //! Removes an item from the trash
    /** \warning The item won't be destroyed!
     **/
    inline void remove(C* item) { m_items.erase(item); }

    //! To manually delete an item already in the trash
    inline void destroy(C* item) {
        m_items.erase(item);
        delete item;
    }

    //! Destructor
    /** Automatically deletes all items
     **/
    ~Garbage() {
        // dispose of left over
        for (auto it = m_items.begin(); it != m_items.end(); ++it) delete *it;
        m_items.clear();
    }

    //! Items to delete
    std::unordered_set<C*> m_items;
};

#endif  // CV_GARBAGE_HEADER
