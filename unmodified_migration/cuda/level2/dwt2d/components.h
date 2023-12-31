/* 
 * Copyright (c) 2009, Jiri Matela
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

 ////////////////////////////////////////////////////////////////////////////////////////////////////
 // file:	altis\src\cuda\level2\dwt2d\dwt_cuda\components.h
 //
 // summary:	Sort class
 // 
 // origin: Rodinia Benchmark (http://rodinia.cs.virginia.edu/doku.php)
 ////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _COMPONENTS_H
#define _COMPONENTS_H

#include "OptionParser.h"

/* Separate compoents of source 8bit RGB image */

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	RGB to components. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="d_r">		   	[in,out] If non-null, the r. </param>
/// <param name="d_g">		   	[in,out] If non-null, the g. </param>
/// <param name="d_b">		   	[in,out] If non-null, the b. </param>
/// <param name="src">		   	[in,out] If non-null, source for the. </param>
/// <param name="width">	   	The width. </param>
/// <param name="height">	   	The height. </param>
/// <param name="transferTime">	[in,out] The transfer time. </param>
/// <param name="kernelTime">  	[in,out] The kernel time. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void rgbToComponents(T *d_r, T *d_g, T *d_b, unsigned char * src, int width, int height, float &transferTime, float &kernelTime, OptionParser &op);

/* Copy a 8bit source image data into a color compoment of type T */

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Bw to component. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="d_c">		   	[in,out] If non-null, the c. </param>
/// <param name="src">		   	[in,out] If non-null, source for the. </param>
/// <param name="width">	   	The width. </param>
/// <param name="height">	   	The height. </param>
/// <param name="transferTime">	[in,out] The transfer time. </param>
/// <param name="kernelTime">  	[in,out] The kernel time. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
void bwToComponent(T *d_c, unsigned char * src, int width, int height, float &transferTime, float &kernelTime);

#endif
