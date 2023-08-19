///-------------------------------------------------------------------------------------------------
// file:	kmeansraw.cu
//
// summary:	kmeans implementation over extents of floats (no underlying point/vector struct)
///-------------------------------------------------------------------------------------------------

#include <CL/sycl.hpp>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <set>
#include <map>
#include <string>

#include "ResultDatabase.h"

typedef double (*LPFNKMEANS)(ResultDatabase &DB,
                             const int nSteps,
                             void * h_Points,
                             void * h_Centers,
	                         const int nPoints,
                             const int nCenters,
	                         bool bVerify,
	                         bool bVerbose);

typedef void (*LPFNBNC)(ResultDatabase &DB,
                        char * szFile, 
                        LPFNKMEANS lpfn, 
                        int nSteps,
                        int nSeed,
                        bool bVerify,
                        bool bVerbose);

#include "kmeansraw.h"
#include "testsuitedecl.h"

declare_testsuite(G_RANK, G_CENTER_COUNT);
