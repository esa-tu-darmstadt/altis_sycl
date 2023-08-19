#include <CL/sycl.hpp>
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

void kernel_gpu_cuda(par_str d_par_gpu,
								dim_str d_dim_gpu,
								box_str* d_box_gpu,
								FOUR_VECTOR* d_rv_gpu,
								fp* d_qv_gpu,
								FOUR_VECTOR* d_fv_gpu,
								sycl::nd_item<3> item_ct1,
								FOUR_VECTOR *rA_shared,
								FOUR_VECTOR *rB_shared,
								double *qB_shared)
{

	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180
	//	THREAD PARAMETERS
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------180

        int bx =
            item_ct1.get_group(2); // get current horizontal block index (0-n)
        int tx = item_ct1.get_local_id(
            2); // get current horizontal thread index (0-n)
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
		while(wtx<NUMBER_PAR_PER_BOX)
        {
			rA_shared[wtx] = rA[wtx];
			wtx = wtx + NUMBER_THREADS;
		}
		wtx = tx;

		// synchronize threads  - not needed, but just to be safe
        item_ct1.barrier(sycl::access::fence_space::local_space);

		// loop over neiing boxes of home box
		for (k=0; k<(1+d_box_gpu[bx].nn); k++)
        {
			if(k==0)
				pointer = bx;	
			else
				pointer = d_box_gpu[bx].nei[k-1].number;	
			
			// nei box - box parameters
			first_j = d_box_gpu[pointer].offset;

			// nei box - distance, (force), charge and (type) parameters
			rB = &d_rv_gpu[first_j];
			qB = &d_qv_gpu[first_j];

			// nei box - shared memory
			while(wtx<NUMBER_PAR_PER_BOX)
            {
				rB_shared[wtx] = rB[wtx];
				qB_shared[wtx] = qB[wtx];
				wtx = wtx + NUMBER_THREADS;
			}
			wtx = tx;

			// synchronize threads because in next section each thread accesses data brought in by different threads here
            item_ct1.barrier(sycl::access::fence_space::local_space);

			// loop for the number of particles in the home box
			// for (int i=0; i<nTotal_i; i++){
			while(wtx<NUMBER_PAR_PER_BOX)
            {
				// loop for the number of particles in the current nei box
				for (j=0; j<NUMBER_PAR_PER_BOX; j++)
                {
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
            item_ct1.barrier();
		}
	}
}
