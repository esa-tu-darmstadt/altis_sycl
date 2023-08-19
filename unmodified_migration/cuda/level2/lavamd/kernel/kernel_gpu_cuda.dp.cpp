#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------200
//	plasmaKernel_gpu_2
//
//	origin: Rodinia (http://rodinia.cs.virginia.edu/doku.php)
//----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------200

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Kernel GPU cuda. </summary>
///
/// <remarks>	Ed, 5/20/2020. </remarks>
///
/// <param name="d_par_gpu">	The par GPU. </param>
/// <param name="d_dim_gpu">	The dim GPU. </param>
/// <param name="d_box_gpu">	[in,out] If non-null, the box GPU. </param>
/// <param name="d_rv_gpu"> 	[in,out] If non-null, the rv GPU. </param>
/// <param name="d_qv_gpu"> 	[in,out] If non-null, the qv GPU. </param>
/// <param name="d_fv_gpu"> 	[in,out] If non-null, the fv GPU. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

/*
DPCT1110:12: The total declared local variable size in device function
kernel_gpu_cuda exceeds 128 bytes and may cause high register pressure. Consult
with your hardware vendor to find the total register size available and adjust
the code, or use smaller sub-group size to avoid high register pressure.
*/
void kernel_gpu_cuda(par_str d_par_gpu, dim_str d_dim_gpu, box_str *d_box_gpu,
                     FOUR_VECTOR *d_rv_gpu, fp *d_qv_gpu, FOUR_VECTOR *d_fv_gpu,
                     const sycl::nd_item<3> &item_ct1, FOUR_VECTOR *rA_shared,
                     FOUR_VECTOR *rB_shared, double *qB_shared)
{

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
	//	THREAD PARAMETERS
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

        int bx = item_ct1.get_group(2); // get current horizontal block index (0-n)
        int tx = item_ct1.get_local_id(
            2); // get current horizontal thread index (0-n)
        // int ax = bx*NUMBER_THREADS+tx;
	// int wbx = bx;
	int wtx = tx;

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
	//	DO FOR THE NUMBER OF BOXES
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

	if(bx<d_dim_gpu.number_boxes){
	// while(wbx<box_indexes_counter){

		//------------------------------------------------------------------------------------------------------------------------------------------------------160
		//	Extract input parameters
		//------------------------------------------------------------------------------------------------------------------------------------------------------160

		// parameters
		fp a2 = 2.0*d_par_gpu.alpha*d_par_gpu.alpha;

		// home box
		int first_i;
		FOUR_VECTOR* rA;
		FOUR_VECTOR* fA;

                // nei box
		int pointer;
		int k = 0;
		int first_j;
		FOUR_VECTOR* rB;
		fp* qB;
		int j = 0;

                // common
		fp r2;
		fp u2;
		fp vij;
		fp fs;
		fp fxij;
		fp fyij;
		fp fzij;
		THREE_VECTOR d;

		//------------------------------------------------------------------------------------------------------------------------------------------------------160
		//	Home box
		//------------------------------------------------------------------------------------------------------------------------------------------------------160

		//----------------------------------------------------------------------------------------------------------------------------------140
		//	Setup parameters
		//----------------------------------------------------------------------------------------------------------------------------------140

		// home box - box parameters
		first_i = d_box_gpu[bx].offset;

		// home box - distance, force, charge and type parameters
		rA = &d_rv_gpu[first_i];
		fA = &d_fv_gpu[first_i];

		//----------------------------------------------------------------------------------------------------------------------------------140
		//	Copy to shared memory
		//----------------------------------------------------------------------------------------------------------------------------------140

		// home box - shared memory
		while(wtx<NUMBER_PAR_PER_BOX){
			rA_shared[wtx] = rA[wtx];
			wtx = wtx + NUMBER_THREADS;
		}
		wtx = tx;

		// synchronize threads  - not needed, but just to be safe
                /*
                DPCT1065:13: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();

                //------------------------------------------------------------------------------------------------------------------------------------------------------160
		//	nei box loop
		//------------------------------------------------------------------------------------------------------------------------------------------------------160

		// loop over neiing boxes of home box
		for (k=0; k<(1+d_box_gpu[bx].nn); k++){

			//----------------------------------------50
			//	nei box - get pointer to the right box
			//----------------------------------------50

			if(k==0){
				pointer = bx;													// set first box to be processed to home box
			}
			else{
				pointer = d_box_gpu[bx].nei[k-1].number;							// remaining boxes are nei boxes
			}

			//----------------------------------------------------------------------------------------------------------------------------------140
			//	Setup parameters
			//----------------------------------------------------------------------------------------------------------------------------------140

			// nei box - box parameters
			first_j = d_box_gpu[pointer].offset;

			// nei box - distance, (force), charge and (type) parameters
			rB = &d_rv_gpu[first_j];
			qB = &d_qv_gpu[first_j];

			//----------------------------------------------------------------------------------------------------------------------------------140
			//	Setup parameters
			//----------------------------------------------------------------------------------------------------------------------------------140

			// nei box - shared memory
			while(wtx<NUMBER_PAR_PER_BOX){
				rB_shared[wtx] = rB[wtx];
				qB_shared[wtx] = qB[wtx];
				wtx = wtx + NUMBER_THREADS;
			}
			wtx = tx;

			// synchronize threads because in next section each thread accesses data brought in by different threads here
                        /*
                        DPCT1065:14: Consider replacing sycl::nd_item::barrier()
                        with
                        sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                        for better performance if there is no access to global
                        memory.
                        */
                        item_ct1.barrier();

                        //----------------------------------------------------------------------------------------------------------------------------------140
			//	Calculation
			//----------------------------------------------------------------------------------------------------------------------------------140

			// loop for the number of particles in the home box
			// for (int i=0; i<nTotal_i; i++){
			while(wtx<NUMBER_PAR_PER_BOX){

				// loop for the number of particles in the current nei box
				for (j=0; j<NUMBER_PAR_PER_BOX; j++){

					// r2 = rA[wtx].v + rB[j].v - DOT(rA[wtx],rB[j]); 
					// u2 = a2*r2;
					// vij= exp(-u2);
					// fs = 2.*vij;

					// d.x = rA[wtx].x  - rB[j].x;
					// fxij=fs*d.x;
					// d.y = rA[wtx].y  - rB[j].y;
					// fyij=fs*d.y;
					// d.z = rA[wtx].z  - rB[j].z;
					// fzij=fs*d.z;

					// fA[wtx].v +=  qB[j]*vij;
					// fA[wtx].x +=  qB[j]*fxij;
					// fA[wtx].y +=  qB[j]*fyij;
					// fA[wtx].z +=  qB[j]*fzij;



					r2 = (fp)rA_shared[wtx].v + (fp)rB_shared[j].v - DOT((fp)rA_shared[wtx],(fp)rB_shared[j]); 
					u2 = a2*r2;
                                        vij = sycl::exp(-u2);
                                        fs = 2*vij;

					d.x = (fp)rA_shared[wtx].x  - (fp)rB_shared[j].x;
					fxij=fs*d.x;
					d.y = (fp)rA_shared[wtx].y  - (fp)rB_shared[j].y;
					fyij=fs*d.y;
					d.z = (fp)rA_shared[wtx].z  - (fp)rB_shared[j].z;
					fzij=fs*d.z;

					fA[wtx].v +=  (double)((fp)qB_shared[j]*vij);
					fA[wtx].x +=  (double)((fp)qB_shared[j]*fxij);
					fA[wtx].y +=  (double)((fp)qB_shared[j]*fyij);
					fA[wtx].z +=  (double)((fp)qB_shared[j]*fzij);

				}

				// increment work thread index
				wtx = wtx + NUMBER_THREADS;

			}

			// reset work index
			wtx = tx;

			// synchronize after finishing force contributions from current nei box not to cause conflicts when starting next box
                        /*
                        DPCT1065:15: Consider replacing sycl::nd_item::barrier()
                        with
                        sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                        for better performance if there is no access to global
                        memory.
                        */
                        item_ct1.barrier();

                        //----------------------------------------------------------------------------------------------------------------------------------140
			//	Calculation END
			//----------------------------------------------------------------------------------------------------------------------------------140

		}

		// // increment work block index
		// wbx = wbx + NUMBER_BLOCKS;

		// // synchronize - because next iteration will overwrite current shared memory
		// __syncthreads();

		//------------------------------------------------------------------------------------------------------------------------------------------------------160
		//	nei box loop END
		//------------------------------------------------------------------------------------------------------------------------------------------------------160

	}

}
