////////////////////////////////////////////////////////////////////////////////////////////////////
// file:
// C:\Users\ed\source\repos\altis\src\cuda\level2\srad\srad_kernel.cu
//
// summary:	Srad kernel class
//
//  origin: Rodinia Benchmark (http://rodinia.cs.virginia.edu/doku.php)
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

void
srad_cuda_1(float                 *E_C,
            float                 *W_C,
            float                 *N_C,
            float                 *S_C,
            float                 *J_cuda,
            float                 *C_cuda,
            int                    cols,
            int                    rows,
            float                  q0sqr,
            sycl::nd_item<2>       item_ct1,
            sycl::local_ptr<float> temp,
            sycl::local_ptr<float> temp_result,
            sycl::local_ptr<float> north,
            sycl::local_ptr<float> south,
            sycl::local_ptr<float> east,
            sycl::local_ptr<float> west)
{

    // block id
    int bx = item_ct1.get_group(1);
    int by = item_ct1.get_group(0);

    // thread id
    int tx = item_ct1.get_local_id(1);
    int ty = item_ct1.get_local_id(0);

    // indices
    int index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
    int index_n = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + tx - cols;
    int index_s
        = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
    int index_w = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty - 1;
    int index_e
        = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;

    if (index_n >= rows * cols || index_s >= rows * cols
        || index_e >= rows * cols || index_w >= rows * cols || index_n < 0
        || index_s < 0 || index_e < 0 || index_w < 0)
    {
        return;
    }

    float n, w, e, s, jc, g2, l, num, den, qsqr, c;

    // shared memory allocation

    // load data to shared memory
    north[ty + tx * BLOCK_SIZE] = J_cuda[index_n];
    south[ty + tx * BLOCK_SIZE] = J_cuda[index_s];
    if (by == 0)
    {
        north[ty + tx * BLOCK_SIZE] = J_cuda[BLOCK_SIZE * bx + tx];
    }
    else if (by == item_ct1.get_group_range(0) - 1)
    {
        south[ty + tx * BLOCK_SIZE]
            = J_cuda[cols * BLOCK_SIZE * (item_ct1.get_group_range(0) - 1)
                     + BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx];
    }
    item_ct1.barrier(sycl::access::fence_space::local_space);

    west[ty + tx * BLOCK_SIZE] = J_cuda[index_w];
    east[ty + tx * BLOCK_SIZE] = J_cuda[index_e];

    if (bx == 0)
        west[ty + tx * BLOCK_SIZE] = J_cuda[cols * BLOCK_SIZE * by + cols * ty];
    else if (bx == item_ct1.get_group_range(1) - 1)
        east[ty + tx * BLOCK_SIZE]
            = J_cuda[cols * BLOCK_SIZE * by
                     + BLOCK_SIZE * (item_ct1.get_group_range(1) - 1)
                     + cols * ty + BLOCK_SIZE - 1];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    temp[ty + tx * BLOCK_SIZE] = J_cuda[index];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    jc = temp[ty + tx * BLOCK_SIZE];

    if (ty == 0 && tx == 0)
    { // nw
        n = north[ty + tx * BLOCK_SIZE] - jc;
        s = temp[(ty + 1) + tx * BLOCK_SIZE] - jc;
        w = west[ty + tx * BLOCK_SIZE] - jc;
        e = temp[ty + (tx + 1) * BLOCK_SIZE] - jc;
    }
    else if (ty == 0 && tx == BLOCK_SIZE - 1)
    { // ne
        n = north[ty + tx * BLOCK_SIZE] - jc;
        s = temp[(ty + 1) + tx * BLOCK_SIZE] - jc;
        w = temp[ty + (tx - 1) * BLOCK_SIZE] - jc;
        e = east[ty + tx * BLOCK_SIZE] - jc;
    }
    else if (ty == BLOCK_SIZE - 1 && tx == BLOCK_SIZE - 1)
    { // se
        n = temp[ty - 1 + tx * BLOCK_SIZE] - jc;
        s = south[ty + tx * BLOCK_SIZE] - jc;
        w = temp[ty + (tx - 1) * BLOCK_SIZE] - jc;
        e = east[ty + tx * BLOCK_SIZE] - jc;
    }
    else if (ty == BLOCK_SIZE - 1 && tx == 0)
    { // sw
        n = temp[ty - 1 + tx * BLOCK_SIZE] - jc;
        s = south[ty + tx * BLOCK_SIZE] - jc;
        w = west[ty + tx * BLOCK_SIZE] - jc;
        e = temp[ty + (tx + 1) * BLOCK_SIZE] - jc;
    }
    else if (ty == 0)
    { // n
        n = north[ty + tx * BLOCK_SIZE] - jc;
        s = temp[ty + 1 + tx * BLOCK_SIZE] - jc;
        w = temp[ty + (tx - 1) * BLOCK_SIZE] - jc;
        e = temp[ty + (tx + 1) * BLOCK_SIZE] - jc;
    }
    else if (tx == BLOCK_SIZE - 1)
    { // e
        n = temp[ty - 1 + tx * BLOCK_SIZE] - jc;
        s = temp[ty + 1 + tx * BLOCK_SIZE] - jc;
        w = temp[ty + (tx - 1) * BLOCK_SIZE] - jc;
        e = east[ty + tx * BLOCK_SIZE] - jc;
    }
    else if (ty == BLOCK_SIZE - 1)
    { // s
        n = temp[ty - 1 + tx * BLOCK_SIZE] - jc;
        s = south[ty + tx * BLOCK_SIZE] - jc;
        w = temp[ty + (tx - 1) * BLOCK_SIZE] - jc;
        e = temp[ty + (tx + 1) * BLOCK_SIZE] - jc;
    }
    else if (tx == 0)
    { // w
        n = temp[ty - 1 + tx * BLOCK_SIZE] - jc;
        s = temp[ty + 1 + tx * BLOCK_SIZE] - jc;
        w = west[ty + tx * BLOCK_SIZE] - jc;
        e = temp[ty + (tx + 1) * BLOCK_SIZE] - jc;
    }
    else
    { // the data elements which are not on the borders
        n = temp[ty - 1 + tx * BLOCK_SIZE] - jc;
        s = temp[ty + 1 + tx * BLOCK_SIZE] - jc;
        w = temp[ty + (tx - 1) * BLOCK_SIZE] - jc;
        e = temp[ty + (tx + 1) * BLOCK_SIZE] - jc;
    }

    g2 = (n * n + s * s + w * w + e * e) / (jc * jc);

    l = (n + s + w + e) / jc;

    num  = (0.5 * g2) - ((1.0 / 16.0) * (l * l));
    den  = 1 + (.25 * l);
    qsqr = num / (den * den);

    // diffusion coefficent (equ 33)
    den = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr));
    c   = 1.0 / (1.0 + den);

    // saturate diffusion coefficent
    if (c < 0)
        temp_result[ty + tx * BLOCK_SIZE] = 0;
    else if (c > 1)
        temp_result[ty + tx * BLOCK_SIZE] = 1;
    else
        temp_result[ty + tx * BLOCK_SIZE] = c;
    item_ct1.barrier(sycl::access::fence_space::local_space);

    C_cuda[index] = temp_result[ty + tx * BLOCK_SIZE];
    E_C[index]    = e;
    W_C[index]    = w;
    S_C[index]    = s;
    N_C[index]    = n;
}

void
srad_cuda_2(float                 *E_C,
            float                 *W_C,
            float                 *N_C,
            float                 *S_C,
            float                 *J_cuda,
            float                 *C_cuda,
            int                    cols,
            int                    rows,
            float                  lambda,
            float                  q0sqr,
            sycl::nd_item<2>       item_ct1,
            sycl::local_ptr<float> south_c,
            sycl::local_ptr<float> east_c,
            sycl::local_ptr<float> c_cuda_temp,
            sycl::local_ptr<float> c_cuda_result,
            sycl::local_ptr<float> temp)
{
    // block id
    int bx = item_ct1.get_group(1);
    int by = item_ct1.get_group(0);

    // thread id
    int tx = item_ct1.get_local_id(1);
    int ty = item_ct1.get_local_id(0);

    // indices
    int index = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
    int index_s
        = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
    int index_e
        = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;
    float cc, cn, cs, ce, cw, d_sum;

    // load data to shared memory
    temp[ty + tx * BLOCK_SIZE] = J_cuda[index];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    south_c[ty + tx * BLOCK_SIZE] = C_cuda[index_s];
    if (by == item_ct1.get_group_range(0) - 1)
        south_c[ty + tx * BLOCK_SIZE]
            = C_cuda[cols * BLOCK_SIZE * (item_ct1.get_group_range(0) - 1)
                     + BLOCK_SIZE * bx + cols * (BLOCK_SIZE - 1) + tx];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    east_c[ty + tx * BLOCK_SIZE] = C_cuda[index_e];
    if (bx == item_ct1.get_group_range(1) - 1)
        east_c[ty + tx * BLOCK_SIZE]
            = C_cuda[cols * BLOCK_SIZE * by
                     + BLOCK_SIZE * (item_ct1.get_group_range(1) - 1)
                     + cols * ty + BLOCK_SIZE - 1];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    c_cuda_temp[ty + tx * BLOCK_SIZE] = C_cuda[index];
    item_ct1.barrier(sycl::access::fence_space::local_space);

    cc = c_cuda_temp[ty + tx * BLOCK_SIZE];
    if (ty == BLOCK_SIZE - 1 && tx == BLOCK_SIZE - 1)
    { // se
        cn = cc;
        cs = south_c[ty + tx * BLOCK_SIZE];
        cw = cc;
        ce = east_c[ty + tx * BLOCK_SIZE];
    }
    else if (tx == BLOCK_SIZE - 1)
    { // e
        cn = cc;
        cs = c_cuda_temp[ty + 1 + tx * BLOCK_SIZE];
        cw = cc;
        ce = east_c[ty + tx * BLOCK_SIZE];
    }
    else if (ty == BLOCK_SIZE - 1)
    { // s
        cn = cc;
        cs = south_c[ty + tx * BLOCK_SIZE];
        cw = cc;
        ce = c_cuda_temp[ty + (tx + 1) * BLOCK_SIZE];
    }
    else
    { // the data elements which are not on the borders
        cn = cc;
        cs = c_cuda_temp[ty + 1 + tx * BLOCK_SIZE];
        cw = cc;
        ce = c_cuda_temp[ty + (tx + 1) * BLOCK_SIZE];
    }

    // divergence (equ 58)
    d_sum
        = cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E_C[index];

    // image update (equ 61)
    c_cuda_result[ty + tx * BLOCK_SIZE]
        = temp[ty + tx * BLOCK_SIZE] + 0.25 * lambda * d_sum;

    item_ct1.barrier(sycl::access::fence_space::local_space);

    J_cuda[index] = c_cuda_result[ty + tx * BLOCK_SIZE];
}
