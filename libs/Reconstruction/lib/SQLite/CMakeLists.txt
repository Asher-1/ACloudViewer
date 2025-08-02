add_compile_definitions(SQLITE_THREADSAFE=1)

COLMAP_ADD_STATIC_LIBRARY(sqlite3
    sqlite3.h
    sqlite3ext.h
    sqlite3.c
)
