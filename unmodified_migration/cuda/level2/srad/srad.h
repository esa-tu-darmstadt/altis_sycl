////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\level2\srad\srad.h
//
// summary:	Declares the srad class
// 
// origin: Rodinia Benchmark (http://rodinia.cs.virginia.edu/doku.php)
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines size. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

#define STR_SIZE 256

#ifdef RD_WG_SIZE_0_0

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block size. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

        #define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block size. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

        #define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block size. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

        #define BLOCK_SIZE RD_WG_SIZE
#else

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A macro that defines block size. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

        #define BLOCK_SIZE 16
#endif
