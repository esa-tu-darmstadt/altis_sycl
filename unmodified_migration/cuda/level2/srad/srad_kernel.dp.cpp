////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	C:\Users\ed\source\repos\altis\src\cuda\level2\srad\srad_kernel.cu
//
// summary:	Srad kernel class
// 
//  origin: Rodinia Benchmark (http://rodinia.cs.virginia.edu/doku.php)
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A srad parameters. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

struct srad_params {
    /// <summary>	The c. </summary>
    float *E_C;
    /// <summary>	The c. </summary>
    float *W_C;
    /// <summary>	The c. </summary>
    float *N_C;
    /// <summary>	The c. </summary>
    float *S_C;
    /// <summary>	The cuda. </summary>
    float *J_cuda;
    /// <summary>	The cuda. </summary>
    float *C_cuda;
    /// <summary>	The cols. </summary>
    int cols;
    /// <summary>	The rows. </summary>
    int rows;
    /// <summary>	The lambda. </summary>
    float lambda;
    /// <summary>	The 0sqr. </summary>
    float q0sqr;
    /*
    srad_params(float *E, float *W, float *N, float *S, float *J_c, float *C_c, int c, int r, float l, float q) {
        E_C = E;
        W_C = W;
        N_C = N;
        S_C = S;
        J_cuda = J_c;
        C_cuda = C_c;
        cols = c;
        rows = r;
        lambda = l;
        q0sqr = q;
    }
    */
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Srad cuda 1. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="E_C">   	[in,out] If non-null, the c. </param>
/// <param name="W_C">   	[in,out] If non-null, the c. </param>
/// <param name="N_C">   	[in,out] If non-null, the c. </param>
/// <param name="S_C">   	[in,out] If non-null, the c. </param>
/// <param name="J_cuda">	[in,out] If non-null, the cuda. </param>
/// <param name="C_cuda">	[in,out] If non-null, the cuda. </param>
/// <param name="cols">  	The cols. </param>
/// <param name="rows">  	The rows. </param>
/// <param name="q0sqr"> 	The 0sqr. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void
srad_cuda_1(
		  float *E_C, 
		  float *W_C, 
		  float *N_C, 
		  float *S_C,
		  float * J_cuda, 
		  float * C_cuda, 
		  int cols, 
		  int rows, 
		  float q0sqr
,
		  const sycl::nd_item<3> &item_ct1,
		  sycl::local_accessor<float, 2> temp,
		  sycl::local_accessor<float, 2> temp_result,
		  sycl::local_accessor<float, 2> north,
		  sycl::local_accessor<float, 2> south,
		  sycl::local_accessor<float, 2> east,
		  sycl::local_accessor<float, 2> west) 
{

  //block id
  int bx = item_ct1.get_group(2);
  int by = item_ct1.get_group(1);

  //thread id
  int tx = item_ct1.get_local_id(2);
  int ty = item_ct1.get_local_id(1);

  //indices
  int index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
  int index_n = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + tx - cols;
  int index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
  int index_w = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty - 1;
  int index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;

  if(index_n >= rows * cols ||
     index_s >= rows * cols ||
     index_e >= rows * cols ||
     index_w >= rows * cols ||
     index_n < 0 ||
     index_s < 0||
     index_e < 0 ||
     index_w < 0) {
      return;
  }

  float n, w, e, s, jc, g2, l, num, den, qsqr, c;

  //shared memory allocation

  //load data to shared memory
  north[ty][tx] = J_cuda[index_n]; 
  south[ty][tx] = J_cuda[index_s];
  if ( by == 0 ){
  north[ty][tx] = J_cuda[BLOCK_SIZE * bx + tx];
  } else if (by == item_ct1.get_group_range(1) - 1) {
  south[ty][tx] = J_cuda[cols * BLOCK_SIZE * (item_ct1.get_group_range(1) - 1) +
                         BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx];
  }
   /*
   DPCT1065:95: Consider replacing sycl::nd_item::barrier() with
   sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
   performance if there is no access to global memory.
   */
   item_ct1.barrier();

  west[ty][tx] = J_cuda[index_w];
  east[ty][tx] = J_cuda[index_e];

  if ( bx == 0 ){
  west[ty][tx] = J_cuda[cols * BLOCK_SIZE * by + cols * ty];
  } else if (bx == item_ct1.get_group_range(2) - 1) {
  east[ty][tx] = J_cuda[cols * BLOCK_SIZE * by +
                        BLOCK_SIZE * (item_ct1.get_group_range(2) - 1) +
                        cols * ty + BLOCK_SIZE - 1];
  }

  /*
  DPCT1065:96: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  temp[ty][tx]      = J_cuda[index];

  item_ct1.barrier(sycl::access::fence_space::local_space);

   jc = temp[ty][tx];

   if ( ty == 0 && tx == 0 ){ //nw
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = west[ty][tx]  - jc; 
    e  = temp[ty][tx+1] - jc;
   }	    
   else if ( ty == 0 && tx == BLOCK_SIZE-1 ){ //ne
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = east[ty][tx] - jc;
   }
   else if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = east[ty][tx]  - jc;
   }
   else if ( ty == BLOCK_SIZE -1 && tx == 0 ){//sw
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = west[ty][tx]  - jc; 
    e  = temp[ty][tx+1] - jc;
   }

   else if ( ty == 0 ){ //n
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = temp[ty][tx+1] - jc;
   }
   else if ( tx == BLOCK_SIZE -1 ){ //e
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = east[ty][tx] - jc;
   }
   else if ( ty == BLOCK_SIZE -1){ //s
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = temp[ty][tx+1] - jc;
   }
   else if ( tx == 0 ){ //w
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = west[ty][tx] - jc; 
    e  = temp[ty][tx+1] - jc;
   }
   else{  //the data elements which are not on the borders 
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = temp[ty][tx+1] - jc;
   }


    g2 = ( n * n + s * s + w * w + e * e ) / (jc * jc);

    l = ( n + s + w + e ) / jc;

	num  = (0.5*g2) - ((1.0/16.0)*(l*l)) ;
	den  = 1 + (.25*l);
	qsqr = num/(den*den);

	// diffusion coefficent (equ 33)
	den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
	c = 1.0 / (1.0+den) ;

    // saturate diffusion coefficent
	if (c < 0){temp_result[ty][tx] = 0;}
	else if (c > 1) {temp_result[ty][tx] = 1;}
	else {temp_result[ty][tx] = c;}

    item_ct1.barrier(sycl::access::fence_space::local_space);

    C_cuda[index] = temp_result[ty][tx];
	E_C[index] = e;
	W_C[index] = w;
	S_C[index] = s;
	N_C[index] = n;

}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Srad cuda 2. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="E_C">   	[in,out] If non-null, the c. </param>
/// <param name="W_C">   	[in,out] If non-null, the c. </param>
/// <param name="N_C">   	[in,out] If non-null, the c. </param>
/// <param name="S_C">   	[in,out] If non-null, the c. </param>
/// <param name="J_cuda">	[in,out] If non-null, the cuda. </param>
/// <param name="C_cuda">	[in,out] If non-null, the cuda. </param>
/// <param name="cols">  	The cols. </param>
/// <param name="rows">  	The rows. </param>
/// <param name="lambda">	The lambda. </param>
/// <param name="q0sqr"> 	The 0sqr. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void
srad_cuda_2(
		  float *E_C, 
		  float *W_C, 
		  float *N_C, 
		  float *S_C,	
		  float * J_cuda, 
		  float * C_cuda, 
		  int cols, 
		  int rows, 
		  float lambda,
		  float q0sqr
,
		  const sycl::nd_item<3> &item_ct1,
		  sycl::local_accessor<float, 2> south_c,
		  sycl::local_accessor<float, 2> east_c,
		  sycl::local_accessor<float, 2> c_cuda_temp,
		  sycl::local_accessor<float, 2> c_cuda_result,
		  sycl::local_accessor<float, 2> temp) 
{
	//block id
        int bx = item_ct1.get_group(2);
    int by = item_ct1.get_group(1);

        //thread id
    int tx = item_ct1.get_local_id(2);
    int ty = item_ct1.get_local_id(1);

        //indices
    int index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
	int index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
    int index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;
	float cc, cn, cs, ce, cw, d_sum;

	//shared memory allocation

    //load data to shared memory
	temp[ty][tx]      = J_cuda[index];

    /*
    DPCT1065:97: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

        south_c[ty][tx] = C_cuda[index_s];

        if (by == item_ct1.get_group_range(1) - 1) {
        south_c[ty][tx] =
            C_cuda[cols * BLOCK_SIZE * (item_ct1.get_group_range(1) - 1) +
                   BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx];
        }
        /*
        DPCT1065:98: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        east_c[ty][tx] = C_cuda[index_e];

        if (bx == item_ct1.get_group_range(2) - 1) {
        east_c[ty][tx] = C_cuda[cols * BLOCK_SIZE * by +
                                BLOCK_SIZE * (item_ct1.get_group_range(2) - 1) +
                                cols * ty + BLOCK_SIZE - 1];
        }

    /*
    DPCT1065:99: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    c_cuda_temp[ty][tx]      = C_cuda[index];

    /*
    DPCT1065:100: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

        cc = c_cuda_temp[ty][tx];

   if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
	cn  = cc;
    cs  = south_c[ty][tx];
    cw  = cc; 
    ce  = east_c[ty][tx];
   } 
   else if ( tx == BLOCK_SIZE -1 ){ //e
	cn  = cc;
    cs  = c_cuda_temp[ty+1][tx];
    cw  = cc; 
    ce  = east_c[ty][tx];
   }
   else if ( ty == BLOCK_SIZE -1){ //s
	cn  = cc;
    cs  = south_c[ty][tx];
    cw  = cc; 
    ce  = c_cuda_temp[ty][tx+1];
   }
   else{ //the data elements which are not on the borders 
	cn  = cc;
    cs  = c_cuda_temp[ty+1][tx];
    cw  = cc; 
    ce  = c_cuda_temp[ty][tx+1];
   }

   // divergence (equ 58)
   d_sum = cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E_C[index];

   // image update (equ 61)
   c_cuda_result[ty][tx] = temp[ty][tx] + 0.25 * lambda * d_sum;

   /*
   DPCT1065:101: Consider replacing sycl::nd_item::barrier() with
   sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
   performance if there is no access to global memory.
   */
   item_ct1.barrier();

   J_cuda[index] = c_cuda_result[ty][tx];
    
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Srad cuda 3. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="params">	A variable-length parameters list containing parameters. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

/*
DPCT1110:102: The total declared local variable size in device function
srad_cuda_3 exceeds 128 bytes and may cause high register pressure. Consult with
your hardware vendor to find the total register size available and adjust the
code, or use smaller sub-group size to avoid high register pressure.
*/
void srad_cuda_3(srad_params params, const sycl::nd_item<3> &item_ct1,
                 sycl::local_accessor<float, 2> temp,
                 sycl::local_accessor<float, 2> temp_result,
                 sycl::local_accessor<float, 2> north,
                 sycl::local_accessor<float, 2> south,
                 sycl::local_accessor<float, 2> east,
                 sycl::local_accessor<float, 2> west,
                 sycl::local_accessor<float, 2> south_c,
                 sycl::local_accessor<float, 2> east_c,
                 sycl::local_accessor<float, 2> c_cuda_temp,
                 sycl::local_accessor<float, 2> c_cuda_result) {
    float *E_C = params.E_C;
    float *W_C = params.W_C;
    float *N_C = params.N_C;
    float *S_C = params.S_C;
    float * J_cuda = params.J_cuda;
    float * C_cuda = params.C_cuda;
    int cols = params.cols;
    int rows = params.rows;
    float lambda = params.lambda;
    float q0sqr = params.q0sqr;

  /*
  DPCT1087:109: SYCL currently does not support cross group synchronization. You
  can specify "--use-experimental-features=nd_range_barrier" to use the dpct
  helper function nd_range_barrier to migrate this_grid().
  */
  grid_group g = this_grid();
  //block id
  int bx = item_ct1.get_group(2);
  int by = item_ct1.get_group(1);

  //thread id
  int tx = item_ct1.get_local_id(2);
  int ty = item_ct1.get_local_id(1);

  //indices
  int index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
  int index_n = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + tx - cols;
  int index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
  int index_w = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty - 1;
  int index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;

  float n, w, e, s, jc, g2, l, num, den, qsqr, c;

  //shared memory allocation

  if(index_n >= rows * cols ||
     index_s >= rows * cols ||
     index_e >= rows * cols ||
     index_w >= rows * cols ||
     index_n < 0 ||
     index_s < 0||
     index_e < 0 ||
     index_w < 0) {
  } else {
  //load data to shared memory
  north[ty][tx] = J_cuda[index_n]; 
  south[ty][tx] = J_cuda[index_s];
  if ( by == 0 ){
  north[ty][tx] = J_cuda[BLOCK_SIZE * bx + tx];
  } else if (by == item_ct1.get_group_range(1) - 1) {
  south[ty][tx] = J_cuda[cols * BLOCK_SIZE * (item_ct1.get_group_range(1) - 1) +
                         BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx];
  }
   /*
   DPCT1065:110: Consider replacing sycl::nd_item::barrier() with
   sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
   performance if there is no access to global memory.
   */
   item_ct1.barrier();

  west[ty][tx] = J_cuda[index_w];
  east[ty][tx] = J_cuda[index_e];

  if ( bx == 0 ){
  west[ty][tx] = J_cuda[cols * BLOCK_SIZE * by + cols * ty];
  } else if (bx == item_ct1.get_group_range(2) - 1) {
  east[ty][tx] = J_cuda[cols * BLOCK_SIZE * by +
                        BLOCK_SIZE * (item_ct1.get_group_range(2) - 1) +
                        cols * ty + BLOCK_SIZE - 1];
  }

  /*
  DPCT1065:111: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

  temp[ty][tx]      = J_cuda[index];

  /*
  DPCT1065:112: Consider replacing sycl::nd_item::barrier() with
  sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
  performance if there is no access to global memory.
  */
  item_ct1.barrier();

   jc = temp[ty][tx];

   if ( ty == 0 && tx == 0 ){ //nw
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = west[ty][tx]  - jc; 
    e  = temp[ty][tx+1] - jc;
   }	    
   else if ( ty == 0 && tx == BLOCK_SIZE-1 ){ //ne
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = east[ty][tx] - jc;
   }
   else if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = east[ty][tx]  - jc;
   }
   else if ( ty == BLOCK_SIZE -1 && tx == 0 ){//sw
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = west[ty][tx]  - jc; 
    e  = temp[ty][tx+1] - jc;
   }

   else if ( ty == 0 ){ //n
	n  = north[ty][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = temp[ty][tx+1] - jc;
   }
   else if ( tx == BLOCK_SIZE -1 ){ //e
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = east[ty][tx] - jc;
   }
   else if ( ty == BLOCK_SIZE -1){ //s
	n  = temp[ty-1][tx] - jc;
    s  = south[ty][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = temp[ty][tx+1] - jc;
   }
   else if ( tx == 0 ){ //w
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = west[ty][tx] - jc; 
    e  = temp[ty][tx+1] - jc;
   }
   else{  //the data elements which are not on the borders 
	n  = temp[ty-1][tx] - jc;
    s  = temp[ty+1][tx] - jc;
    w  = temp[ty][tx-1] - jc; 
    e  = temp[ty][tx+1] - jc;
   }

    g2 = ( n * n + s * s + w * w + e * e ) / (jc * jc);

    l = ( n + s + w + e ) / jc;

	num  = (0.5*g2) - ((1.0/16.0)*(l*l)) ;
	den  = 1 + (.25*l);
	qsqr = num/(den*den);

	// diffusion coefficent (equ 33)
	den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
	c = 1.0 / (1.0+den) ;

    // saturate diffusion coefficent
	if (c < 0){temp_result[ty][tx] = 0;}
	else if (c > 1) {temp_result[ty][tx] = 1;}
	else {temp_result[ty][tx] = c;}

    /*
    DPCT1065:113: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    C_cuda[index] = temp_result[ty][tx];
	E_C[index] = e;
	W_C[index] = w;
	S_C[index] = s;
	N_C[index] = n;

    /*
    DPCT1065:114: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }

    /* GRID SYNC */
    /*
    DPCT1087:103: SYCL currently does not support cross group synchronization.
    You can specify "--use-experimental-features=nd_range_barrier" to use the
    dpct helper function nd_range_barrier to migrate g.sync().
    */
    g.sync();

        //block id
        bx = item_ct1.get_group(2);
    by = item_ct1.get_group(1);

        //thread id
    tx = item_ct1.get_local_id(2);
    ty = item_ct1.get_local_id(1);

        //indices
    index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
	index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
    index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;
	float cc, cn, cs, ce, cw, d_sum;

	//shared memory allocation

    //__shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];

    //load data to shared memory
	temp[ty][tx]      = J_cuda[index];

    /*
    DPCT1065:104: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

        south_c[ty][tx] = C_cuda[index_s];

        if (by == item_ct1.get_group_range(1) - 1) {
        south_c[ty][tx] =
            C_cuda[cols * BLOCK_SIZE * (item_ct1.get_group_range(1) - 1) +
                   BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx];
        }
        /*
        DPCT1065:105: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        east_c[ty][tx] = C_cuda[index_e];

        if (bx == item_ct1.get_group_range(2) - 1) {
        east_c[ty][tx] = C_cuda[cols * BLOCK_SIZE * by +
                                BLOCK_SIZE * (item_ct1.get_group_range(2) - 1) +
                                cols * ty + BLOCK_SIZE - 1];
        }

    /*
    DPCT1065:106: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    c_cuda_temp[ty][tx]      = C_cuda[index];

    /*
    DPCT1065:107: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

        cc = c_cuda_temp[ty][tx];

   if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
	cn  = cc;
    cs  = south_c[ty][tx];
    cw  = cc; 
    ce  = east_c[ty][tx];
   } 
   else if ( tx == BLOCK_SIZE -1 ){ //e
	cn  = cc;
    cs  = c_cuda_temp[ty+1][tx];
    cw  = cc; 
    ce  = east_c[ty][tx];
   }
   else if ( ty == BLOCK_SIZE -1){ //s
	cn  = cc;
    cs  = south_c[ty][tx];
    cw  = cc; 
    ce  = c_cuda_temp[ty][tx+1];
   }
   else{ //the data elements which are not on the borders 
	cn  = cc;
    cs  = c_cuda_temp[ty+1][tx];
    cw  = cc; 
    ce  = c_cuda_temp[ty][tx+1];
   }

   // divergence (equ 58)
   d_sum = cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E_C[index];

   // image update (equ 61)
   c_cuda_result[ty][tx] = temp[ty][tx] + 0.25 * lambda * d_sum;

   /*
   DPCT1065:108: Consider replacing sycl::nd_item::barrier() with
   sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
   performance if there is no access to global memory.
   */
   item_ct1.barrier();

   J_cuda[index] = c_cuda_result[ty][tx];
    
}