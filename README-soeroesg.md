changes:
adaptations for Visual Studio

cleaner CMakeLists, not only GCC-specific, target-specific linking libraries, installable cmake package
guarded Unix-only headers
changed usleep to std::sleep
changed OpenSSL MD5 to hashlibpp MD5, added hashlibpp library in thirdparty folder
MapPoint.cpp: changed float Distances[N][N] array to std::vector<std::vector<int>>
LoopClosing.h: changed bool mnFullBAIdx; to size_t
added build and run scripts


PnPsolver.h -- is this needed? seems to have been deleted on upstream