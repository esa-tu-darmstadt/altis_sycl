/***********************************************************************************
  Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

  Copyright (c) 2008 International Institute of Information Technology -
 Hyderabad. All rights reserved.

  Permission to use, copy, modify and distribute this software and its
 documentation for educational purpose is hereby granted without fee, provided
 that the above copyright notice and this permission notice appear in all copies
 of this software and that you do not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS,
 IMPLIED OR OTHERWISE.

  Created by Pawan Harish.

  origin: Rodinia (http://rodinia.cs.virginia.edu/doku.php)
 ************************************************************************************/

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

#include <cassert>
#include <chrono>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include "cudacommon.h"
#include "ResultDatabase.h"
#include "OptionParser.h"

#define MIN_NODES      20
#define MAX_NODES      ULONG_MAX
#define MIN_EDGES      2
#define MAX_INIT_EDGES 4 // Nodes will have, on average, 2*MAX_INIT_EDGES edges
#define MIN_WEIGHT     1
#define MAX_WEIGHT     10
#define SEED           7

using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	A node structure in the graph. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////

struct Node
{
    int starting;
    int no_of_edges;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Initializes the graph. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="op">			 	[in,out] The operation. </param>
/// <param name="no_of_nodes">   	[in,out] The no of nodes. </param>
/// <param name="edge_list_size">	[in,out] Size of the edge list. </param>
/// <param name="source">		 	[in,out] Source for the
/// initialization.
/// </param>
/// <param name="h_graph_nodes"> 	[in,out] [in,out] If non-null, the graph
/// nodes. </param> <param name="h_graph_edges"> 	[in,out] [in,out] If
/// non-null, the graph edges. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void initGraph(OptionParser &op,
               int          &no_of_nodes,
               int          &edge_list_size,
               int          &source,
               Node        *&h_graph_nodes,
               int         *&h_graph_edges);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	BFS graph runner. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="resultDB">		 	[in,out] The result database.
/// </param> <param name="op">			 	[in,out] The operation.
/// </param> <param name="no_of_nodes">   	The no of nodes. </param> <param
/// name="edge_list_size">	Size of the edge list. </param> <param
/// name="source">		 	Source for the. </param> <param
/// name="h_graph_nodes"> 	[in,out] [in,out] If non-null, the graph nodes.
/// </param>
/// <param name="h_graph_edges"> 	[in,out] [in,out] If non-null, the graph
/// edges. </param>
///
/// <returns>	A float. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

float BFSGraph(ResultDatabase &resultDB,
               OptionParser   &op,
               int             no_of_nodes,
               int             edge_list_size,
               int             source,
               Node          *&h_graph_nodes,
               int           *&h_graph_edges,
               sycl::queue    &queue);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	BFS graph using unified memory. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="resultDB">		 	[in,out] The result database.
/// </param> <param name="op">			 	[in,out] The operation.
/// </param> <param name="no_of_nodes">   	The no of nodes. </param> <param
/// name="edge_list_size">	Size of the edge list. </param> <param
/// name="source">		 	Source for the. </param> <param
/// name="h_graph_nodes"> 	[in,out] [in,out] If non-null, the graph nodes.
/// </param>
/// <param name="h_graph_edges"> 	[in,out] [in,out] If non-null, the graph
/// edges. </param>
///
/// <returns>	A float. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

float BFSGraphUnifiedMemory(ResultDatabase &resultDB,
                            OptionParser   &op,
                            int             no_of_nodes,
                            int             edge_list_size,
                            int             source,
                            Node          *&h_graph_nodes,
                            int           *&h_graph_edges,
                            sycl::queue    &queue);

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	BFS Kernel. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="g_graph_nodes">			[in,out] If non-null, the
/// graph nodes. </param> <param name="g_graph_edges">			[in,out]
/// If non-null, the graph edges. </param> <param name="g_graph_mask">
/// [in,out] If non-null, true to graph mask. </param> <param
/// name="g_updating_graph_mask">	[in,out] If non-null, true to updating
/// graph mask. </param> <param name="g_graph_visited">
/// [in,out] If non-null, true if graph visited. </param> <param name="g_cost">
/// [in,out] If non-null, the cost.
/// </param> <param name="no_of_nodes">				The no of nodes.
/// </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void
Kernel(Node            *g_graph_nodes,
       int             *g_graph_edges,
       bool            *g_graph_mask,
       bool            *g_updating_graph_mask,
       bool            *g_graph_visited,
       int             *g_cost,
       int              no_of_nodes,
       sycl::nd_item<3> item_ct1)
{
    int tid = (item_ct1.get_group(2) * MAX_THREADS_PER_BLOCK)
              + item_ct1.get_local_id(2);
    if (tid < no_of_nodes && g_graph_mask[tid])
    {
        g_graph_mask[tid] = false;
        for (int i = g_graph_nodes[tid].starting;
             i < (g_graph_nodes[tid].no_of_edges + g_graph_nodes[tid].starting);
             i++)
        {
            int id = g_graph_edges[i];
            if (!g_graph_visited[id])
            {
                g_cost[id]                = g_cost[tid] + 1;
                g_updating_graph_mask[id] = true;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	BFS Kernel 2. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="g_graph_mask">				[in,out] If non-null, true
/// to graph mask. </param> <param name="g_updating_graph_mask">	[in,out]
/// If non-null, true to updating graph mask. </param> <param
/// name="g_graph_visited">			[in,out] If non-null, true if
/// graph visited. </param>
/// <param name="g_over">					[in,out] If non-null, true to
/// over.
/// </param> <param name="no_of_nodes">				The no of nodes.
/// </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void
Kernel2(bool            *g_graph_mask,
        bool            *g_updating_graph_mask,
        bool            *g_graph_visited,
        bool            *g_over,
        int              no_of_nodes,
        sycl::nd_item<3> item_ct1)
{
    int tid = (item_ct1.get_group(2) * MAX_THREADS_PER_BLOCK)
              + item_ct1.get_local_id(2);
    if (tid < no_of_nodes && g_updating_graph_mask[tid])
    {
        g_graph_mask[tid]          = true;
        g_graph_visited[tid]       = true;
        *g_over                    = true;
        g_updating_graph_mask[tid] = false;
    }
}
////////////////////////////////////////////////////////////////////////////////

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************

void
addBenchmarkSpecOptions(OptionParser &op)
{
}

// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Executes the radix sort benchmark
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser / parameter database
//
// Returns:  nothing, results are stored in resultDB
//
// Programmer: Kyle Spafford
// Creation: August 13, 2009
//
// Modifications:
//
// ****************************************************************************

void
RunBenchmark(ResultDatabase &resultDB, OptionParser &op, size_t device_idx)
{
    printf("Running BFS\n");

    std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
    sycl::queue                   queue(devices[device_idx],
                      sycl::property::queue::enable_profiling {});

    // seed random number generator
    srand(SEED);

    int   no_of_nodes    = 0;
    int   edge_list_size = 0;
    int   source         = 0;
    Node *h_graph_nodes;
    int  *h_graph_edges;
    initGraph(
        op, no_of_nodes, edge_list_size, source, h_graph_nodes, h_graph_edges);

    // atts string for result database
    char tmp[64];
    sprintf(tmp, "%dV,%dE", no_of_nodes, edge_list_size);
    string atts = string(tmp);

    bool quiet  = op.getOptionBool("quiet");
    int  passes = op.getOptionInt("passes");

    const bool uvm                 = op.getOptionBool("uvm");
    const bool uvm_advise          = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch        = op.getOptionBool("uvm-prefetch");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");

    for (int i = 0; i < passes; i++)
    {
        if (!quiet)
            printf("Pass %d:\n", i);

        if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise)
        {
            float timeUM = BFSGraphUnifiedMemory(resultDB,
                                                 op,
                                                 no_of_nodes,
                                                 edge_list_size,
                                                 source,
                                                 h_graph_nodes,
                                                 h_graph_edges,
                                                 queue);
            if (!quiet)
            {
                if (timeUM == FLT_MAX)
                    printf("Executing BFS using unified memory...Error.\n");
                else
                    printf("Executing BFS using unified memory...Done.\n");
            }
            // if(time != FLT_MAX && timeUM != FLT_MAX) {
            if (timeUM != FLT_MAX)
            {
                // resultDB.AddResult("bfs_unifiedmem_speedup", atts, "N",
                // time/timeUM);
            }
        }
        else
        {
            float time = BFSGraph(resultDB,
                                  op,
                                  no_of_nodes,
                                  edge_list_size,
                                  source,
                                  h_graph_nodes,
                                  h_graph_edges,
                                  queue);
            if (!quiet)
            {
                if (time == FLT_MAX)
                    printf("Executing BFS...Error.\n");
                else
                    printf("Executing BFS...Done.\n");
            }
        }
    }

    free(h_graph_nodes);
    free(h_graph_edges);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Generate Uniform distribution. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="rangeLow"> 	The range low. </param>
/// <param name="rangeHigh">	The range high. </param>
///
/// <returns>	A scaled random int. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

int
uniform_distribution(int rangeLow, int rangeHigh)
{
    double myRand        = rand() / (1.0 + RAND_MAX);
    int    range         = rangeHigh - rangeLow + 1;
    int    myRand_scaled = (myRand * range) + rangeLow;
    return myRand_scaled;
}

////////////////////////////////////////////////////////////////////////////////
// Initialize Graph
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Initializes the graph. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="op">			 	[in,out] The operation. </param>
/// <param name="no_of_nodes">   	[in,out] The no of nodes. </param>
/// <param name="edge_list_size">	[in,out] Size of the edge list. </param>
/// <param name="source">		 	[in,out] Source for the.
/// </param>
/// <param name="h_graph_nodes"> 	[in,out] [in,out] If non-null, the graph
/// nodes. </param> <param name="h_graph_edges"> 	[in,out] [in,out] If
/// non-null, the graph edges. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

void
initGraph(OptionParser &op,
          int          &no_of_nodes,
          int          &edge_list_size,
          int          &source,
          Node        *&h_graph_nodes,
          int         *&h_graph_edges)
{
    bool quiet = op.getOptionBool("quiet");
    // open input file for reading
    FILE  *fp     = NULL;
    string infile = op.getOptionString("inputFile");
    if (infile != "")
    {
        fp = fopen(infile.c_str(), "r");
        if (!fp && !quiet)
            printf("Error: Unable to read graph file %s.\n", infile.c_str());
    }

    if (!quiet)
    {
        if (fp)
            printf("Reading graph file\n");
        else
            printf("Generating graph with problem size %d\n",
                   (int)op.getOptionInt("size"));
    }

    // initialize number of nodes
    if (fp)
    {
        int n = fscanf(fp, "%d", &no_of_nodes);
        assert(n == 1);
    }
    else
    {
        int problemSizes[5] = { 10, 50, 200, 400, 600 };
        no_of_nodes = problemSizes[op.getOptionInt("size") - 1] * 1024 * 1024;
    }
    // initalize the nodes & number of edges
    h_graph_nodes = (Node *)malloc(sizeof(Node) * no_of_nodes);
    assert(h_graph_nodes);
    int start;
    int edgeno;
    for (int i = 0; i < no_of_nodes; i++)
    {
        if (fp)
        {
            int n = fscanf(fp, "%d %d", &start, &edgeno);
            assert(n == 2);
        }
        else
        {
            start  = edge_list_size;
            edgeno = rand() % (MAX_INIT_EDGES - MIN_EDGES + 1) + MIN_EDGES;
        }
        h_graph_nodes[i].starting    = start;
        h_graph_nodes[i].no_of_edges = edgeno;
        edge_list_size += edgeno;
    }

    // initialize the source node
    if (fp)
    {
        int n = fscanf(fp, "%d", &source);
        assert(n == 1);
    }
    else
    {
        source = uniform_distribution(0, no_of_nodes - 1);
    }
    source = 0;

    if (fp)
    {
        int edges;
        int n = fscanf(fp, "%d", &edges);
        assert(n == 1);
        assert(edges == edge_list_size);
    }

    // initialize the edges
    int id;
    int cost;
    h_graph_edges = (int *)malloc(sizeof(int) * edge_list_size);
    assert(h_graph_edges);
    for (int i = 0; i < edge_list_size; i++)
    {
        if (fp)
        {
            int n = fscanf(fp, "%d %d", &id, &cost);
            assert(n == 2);
        }
        else
        {
            id = uniform_distribution(0, no_of_nodes - 1);
            // cost = rand() % (MAX_WEIGHT - MIN_WEIGHT + 1) + MIN_WEIGHT;
        }
        h_graph_edges[i] = id;
    }

    if (!quiet)
    {
        if (fp)
        {
            fclose(fp);
            printf("Done reading graph file\n");
        }
        else
        {
            printf("Done generating graph\n");
        }
        printf("Graph size: %d nodes, %d edges\n", no_of_nodes, edge_list_size);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Bfs graph using CUDA. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="resultDB">		 	[in,out] The result database.
/// </param> <param name="op">			 	[in,out] The operation.
/// </param> <param name="no_of_nodes">   	The no of nodes. </param> <param
/// name="edge_list_size">	Size of the edge list. </param> <param
/// name="source">		 	Source for the. </param> <param
/// name="h_graph_nodes"> 	[in,out] [in,out] If non-null, the graph nodes.
/// </param>
/// <param name="h_graph_edges"> 	[in,out] [in,out] If non-null, the graph
/// edges. </param>
///
/// <returns>	Transfer time and kernel time. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

float
BFSGraph(ResultDatabase &resultDB,
         OptionParser   &op,
         int             no_of_nodes,
         int             edge_list_size,
         int             source,
         Node          *&h_graph_nodes,
         int           *&h_graph_edges,
         sycl::queue    &queue)
try
{
    bool verbose = op.getOptionBool("verbose");

    int num_of_blocks            = 1;
    int num_of_threads_per_block = no_of_nodes;

    // Make execution Parameters according to the number of nodes
    // Distribute threads across multiple Blocks if necessary
    if (no_of_nodes > MAX_THREADS_PER_BLOCK)
    {
        num_of_blocks = (int)ceil(no_of_nodes / (double)MAX_THREADS_PER_BLOCK);
        num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
    }

    // allocate host memory
    bool *h_graph_mask = (bool *)malloc(sizeof(bool) * no_of_nodes);
    assert(h_graph_mask);
    bool *h_updating_graph_mask = (bool *)malloc(sizeof(bool) * no_of_nodes);
    assert(h_updating_graph_mask);
    bool *h_graph_visited = (bool *)malloc(sizeof(bool) * no_of_nodes);
    assert(h_graph_visited);

    // initalize the memory
    for (int i = 0; i < no_of_nodes; i++)
    {
        h_graph_mask[i]          = false;
        h_updating_graph_mask[i] = false;
        h_graph_visited[i]       = false;
    }

    // set the source node as true in the mask
    h_graph_mask[source]    = true;
    h_graph_visited[source] = true;

    // allocate mem for the result on host side
    int *h_cost = (int *)malloc(sizeof(int) * no_of_nodes);
    assert(h_cost);
    for (int i = 0; i < no_of_nodes; i++)
        h_cost[i] = -1;
    h_cost[source] = 0;

    Node *d_graph_nodes = sycl::malloc_device<Node>(no_of_nodes, queue);
    int  *d_graph_edges = sycl::malloc_device<int>(edge_list_size, queue);
    bool *d_graph_mask  = sycl::malloc_device<bool>(no_of_nodes, queue);
    bool *d_updating_graph_mask = sycl::malloc_device<bool>(no_of_nodes, queue);
    bool *d_graph_visited       = sycl::malloc_device<bool>(no_of_nodes, queue);
    int  *d_cost                = sycl::malloc_device<int>(no_of_nodes, queue);
    bool *d_over                = sycl::malloc_device<bool>(1, queue);
    if (d_graph_nodes == nullptr || d_graph_edges == nullptr
        || d_graph_mask == nullptr || d_updating_graph_mask == nullptr
        || d_graph_visited == nullptr || d_cost == nullptr || d_over == nullptr)
    {
        free(h_graph_mask);
        free(h_updating_graph_mask);
        free(h_graph_visited);
        free(h_cost);
        sycl::free(d_graph_nodes, queue);
        sycl::free(d_graph_edges, queue);
        sycl::free(d_graph_mask, queue);
        sycl::free(d_updating_graph_mask, queue);
        sycl::free(d_graph_visited, queue);
        sycl::free(d_cost, queue);
        sycl::free(d_over, queue);
        return FLT_MAX;
    }

    sycl::event                                        tstart, tstop;
    std::chrono::time_point<std::chrono::steady_clock> tstart_ct1;
    std::chrono::time_point<std::chrono::steady_clock> tstop_ct1;
    float  elapsedTime;
    double transferTime = 0.;
    tstart_ct1 = std::chrono::steady_clock::now();
    queue.memcpy(d_graph_nodes, h_graph_nodes, sizeof(Node) * no_of_nodes);
    queue.memcpy(d_graph_edges, h_graph_edges, sizeof(int) * edge_list_size);
    queue.memcpy(d_graph_mask, h_graph_mask, sizeof(bool) * no_of_nodes);
    queue.memcpy(d_updating_graph_mask,
                 h_updating_graph_mask,
                 sizeof(bool) * no_of_nodes);
    queue.memcpy(d_graph_visited, h_graph_visited, sizeof(bool) * no_of_nodes);
    queue.memcpy(d_cost, h_cost, sizeof(int) * no_of_nodes);
    queue.wait_and_throw();
    tstop_ct1 = std::chrono::steady_clock::now();
    elapsedTime
        = std::chrono::duration<float, std::milli>(tstop_ct1 - tstart_ct1)
              .count();
    transferTime += elapsedTime * 1.e-3; // convert to seconds

    // setup execution parameters
    sycl::range<3> grid(1, 1, num_of_blocks);
    sycl::range<3> threads(1, 1, num_of_threads_per_block);

    double kernelTime = 0;
    int    k          = 0;
    bool   stop;
    // Call the Kernel untill all the elements of Frontier are not false
    do
    {
        // if no thread changes this value then the loop stops
        stop = false;
        queue.memcpy(d_over, &stop, sizeof(bool)).wait();

        sycl::event kernel_event = queue.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                           [=](sycl::nd_item<3> item_ct1) {
                               Kernel(d_graph_nodes,
                                      d_graph_edges,
                                      d_graph_mask,
                                      d_updating_graph_mask,
                                      d_graph_visited,
                                      d_cost,
                                      no_of_nodes,
                                      item_ct1);
                           });
        kernel_event.wait();
        elapsedTime
            = kernel_event
                .get_profiling_info<sycl::info::event_profiling::command_end>()
            - kernel_event.get_profiling_info<
                sycl::info::event_profiling::command_start>();
        kernelTime += elapsedTime * 1.e-9;

        sycl::event kernel2_event = queue.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                           [=](sycl::nd_item<3> item_ct1) {
                               Kernel2(d_graph_mask,
                                       d_updating_graph_mask,
                                       d_graph_visited,
                                       d_over,
                                       no_of_nodes,
                                       item_ct1);
                           });
        kernel2_event.wait();
        elapsedTime
            = kernel2_event
                .get_profiling_info<sycl::info::event_profiling::command_end>()
            - kernel2_event.get_profiling_info<
                sycl::info::event_profiling::command_start>();
        kernelTime += elapsedTime * 1.e-9;

        queue.memcpy(&stop, d_over, sizeof(bool)).wait();

        k++;
    } while (stop);

    if (verbose)
        printf("Kernel Executed %d times\n", k);

    // copy result from device to host
    tstart_ct1 = std::chrono::steady_clock::now();
    queue.memcpy(h_cost, d_cost, sizeof(int) * no_of_nodes).wait();
    tstop_ct1 = std::chrono::steady_clock::now();
    elapsedTime
        = std::chrono::duration<float, std::milli>(tstop_ct1 - tstart_ct1)
              .count();
    transferTime += elapsedTime * 1.e-3; // convert to seconds

    // Store the result into a file
    string outfile = op.getOptionString("outputFile");
    if (outfile != "")
    {
        FILE *fpo = fopen(outfile.c_str(), "w");
        for (int i = 0; i < no_of_nodes; i++)
            fprintf(fpo, "%d) cost:%d\n", i, h_cost[i]);
        fclose(fpo);
    }

    // cleanup memory
    free(h_graph_mask);
    free(h_updating_graph_mask);
    free(h_graph_visited);
    free(h_cost);
    sycl::free(d_graph_nodes, queue);
    sycl::free(d_graph_edges, queue);
    sycl::free(d_graph_mask, queue);
    sycl::free(d_updating_graph_mask, queue);
    sycl::free(d_graph_visited, queue);
    sycl::free(d_cost, queue);
    sycl::free(d_over, queue);

    char tmp[64];
    sprintf(tmp, "%dV,%dE", no_of_nodes, edge_list_size);
    string atts = string(tmp);
    resultDB.AddResult("bfs_transfer_time", atts, "sec", transferTime);
    resultDB.AddResult("bfs_kernel_time", atts, "sec", kernelTime);
    resultDB.AddResult(
        "bfs_total_time", atts, "sec", transferTime + kernelTime);
    resultDB.AddResult(
        "bfs_rate_nodes", atts, "Nodes/s", no_of_nodes / kernelTime);
    resultDB.AddResult(
        "bfs_rate_edges", atts, "Edges/s", edge_list_size / kernelTime);
    resultDB.AddResult("bfs_rate_parity", atts, "N", transferTime / kernelTime);
    resultDB.AddOverall("Time", "sec", kernelTime + transferTime);
    return transferTime + kernelTime;
}
catch (sycl::exception const &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Bfs graph with unified memory using CUDA. </summary>
///
/// <remarks>	Edward Hu (bodunhu@utexas.edu), 5/19/2020. </remarks>
///
/// <param name="resultDB">		 	[in,out] The result database.
/// </param> <param name="op">			 	[in,out] The operation.
/// </param> <param name="no_of_nodes">   	The no of nodes. </param> <param
/// name="edge_list_size">	Size of the edge list. </param> <param
/// name="source">		 	Source for the. </param> <param
/// name="h_graph_nodes"> 	[in,out] [in,out] If non-null, the graph nodes.
/// </param>
/// <param name="h_graph_edges"> 	[in,out] [in,out] If non-null, the graph
/// edges. </param>
///
/// <returns>	Kernel time. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////

float
BFSGraphUnifiedMemory(ResultDatabase &resultDB,
                      OptionParser   &op,
                      int             no_of_nodes,
                      int             edge_list_size,
                      int             source,
                      Node          *&h_graph_nodes,
                      int           *&h_graph_edges,
                      sycl::queue    &queue)
try
{
    bool       verbose             = op.getOptionBool("verbose");
    bool       quiet               = op.getOptionBool("quiet");
    const bool uvm                 = op.getOptionBool("uvm");
    const bool uvm_prefetch        = op.getOptionBool("uvm-prefetch");
    const bool uvm_advise          = op.getOptionBool("uvm-advise");
    const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");

    int num_of_blocks            = 1;
    int num_of_threads_per_block = no_of_nodes;
    // Make execution Parameters according to the number of nodes
    // Distribute threads across multiple Blocks if necessary
    if (no_of_nodes > MAX_THREADS_PER_BLOCK)
    {
        num_of_blocks = (int)ceil(no_of_nodes / (double)MAX_THREADS_PER_BLOCK);
        num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
    }

    // copy graph nodes to unified memory
    Node *graph_nodes = NULL;
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise)
        graph_nodes = sycl::malloc_shared<Node>(no_of_nodes, queue);
    memcpy(graph_nodes, h_graph_nodes, sizeof(Node) * no_of_nodes);

    if (uvm)
    {
        // do nothing, graph_nodes remains on CPU
    }
    else if (uvm_prefetch)
    {
        queue.prefetch(graph_nodes, sizeof(Node) * no_of_nodes);
    }
    else if (uvm_advise)
    {
        queue.mem_advise(graph_nodes, sizeof(Node) * no_of_nodes, 0);
        queue.mem_advise(graph_nodes, sizeof(Node) * no_of_nodes, 0);
    }
    else if (uvm_prefetch_advise)
    {
        queue.mem_advise(graph_nodes, sizeof(Node) * no_of_nodes, 0);
        queue.mem_advise(graph_nodes, sizeof(Node) * no_of_nodes, 0);
        queue.prefetch(graph_nodes, sizeof(Node) * no_of_nodes);
    }
    else
    {
        std::cerr << "unrecognized uvm flag, exiting..." << std::endl;
        exit(-1);
    }

    // copy graph edges to unified memory
    int *graph_edges = NULL;
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise)
        graph_edges = sycl::malloc_shared<int>(edge_list_size, queue);
    memcpy(graph_edges, h_graph_edges, sizeof(int) * edge_list_size);
    if (uvm)
    {
        // Do nothing, graph_edges remains on CPU
    }
    else if (uvm_prefetch)
    {
        queue.prefetch(graph_edges, sizeof(int) * edge_list_size);
    }
    else if (uvm_advise)
    {
        queue.mem_advise(graph_edges, sizeof(int) * edge_list_size, 0);
        queue.mem_advise(graph_edges, sizeof(int) * edge_list_size, 0);
    }
    else if (uvm_prefetch_advise)
    {
        queue.mem_advise(graph_edges, sizeof(int) * edge_list_size, 0);
        queue.mem_advise(graph_edges, sizeof(int) * edge_list_size, 0);
        queue.prefetch(graph_edges, sizeof(int) * edge_list_size);
    }
    else
    {
        std::cerr << "unrecognized uvm flag, exiting..." << std::endl;
        exit(-1);
    }

    // allocate and initalize the memory
    bool *graph_mask;
    bool *updating_graph_mask;
    bool *graph_visited;
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise)
    {
        graph_mask          = sycl::malloc_shared<bool>(no_of_nodes, queue);
        updating_graph_mask = sycl::malloc_shared<bool>(no_of_nodes, queue);
        graph_visited       = sycl::malloc_shared<bool>(no_of_nodes, queue);
    }

    for (int i = 0; i < no_of_nodes; i++)
    {
        graph_mask[i]          = false;
        updating_graph_mask[i] = false;
        graph_visited[i]       = false;
    }

    // set the source node as true in the mask
    graph_mask[source]    = true;
    graph_visited[source] = true;

    if (uvm)
    {
        // Do nothing. graph_mask, updating_graph_mask, and graph_visited
        // unallocated
    }
    else if (uvm_advise)
    {
        queue.mem_advise(graph_mask, sizeof(bool) * no_of_nodes, 0);
        queue.mem_advise(updating_graph_mask, sizeof(bool) * no_of_nodes, 0);
        queue.mem_advise(graph_visited, sizeof(bool) * no_of_nodes, 0);
    }
    else if (uvm_prefetch)
    {
        queue.prefetch(graph_mask, sizeof(bool) * no_of_nodes);
        queue.prefetch(updating_graph_mask, sizeof(bool) * no_of_nodes);
        queue.prefetch(graph_visited, sizeof(bool) * no_of_nodes);
    }
    else if (uvm_prefetch_advise)
    {
        queue.mem_advise(graph_mask, sizeof(bool) * no_of_nodes, 0);
        queue.mem_advise(updating_graph_mask, sizeof(bool) * no_of_nodes, 0);
        queue.mem_advise(graph_visited, sizeof(bool) * no_of_nodes, 0);
        queue.prefetch(graph_mask, sizeof(bool) * no_of_nodes);
        queue.prefetch(updating_graph_mask, sizeof(bool) * no_of_nodes);
        queue.prefetch(graph_visited, sizeof(bool) * no_of_nodes);
    }

    // allocate and initialize memory for result
    int *cost = nullptr;
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise)
    {
        cost = sycl::malloc_shared<int>(no_of_nodes, queue);
        if (nullptr == cost)
        {
            sycl::free(graph_nodes, queue);
            sycl::free(graph_edges, queue);
            sycl::free(graph_mask, queue);
            sycl::free(updating_graph_mask, queue);
            sycl::free(graph_visited, queue);
            sycl::free(cost, queue);
            return FLT_MAX;
        }
    }

    for (int i = 0; i < no_of_nodes; i++)
        cost[i] = -1;
    cost[source] = 0;

    if (uvm)
    {
        // Do nothing, cost stays on CPU
    }
    else if (uvm_advise)
    {
        queue.mem_advise(cost, sizeof(int) * no_of_nodes, 0);
    }
    else if (uvm_prefetch)
    {
        queue.prefetch(cost, sizeof(int) * no_of_nodes);
    }
    else if (uvm_prefetch_advise)
    {
        queue.mem_advise(cost, sizeof(int) * no_of_nodes, 0);
        queue.prefetch(cost, sizeof(int) * no_of_nodes);
    }
    else
    {
        std::cerr << "Unrecognized uvm option, exiting...";
        exit(-1);
    }

    // bool if execution is over
    bool *over = NULL;
    if (uvm || uvm_advise || uvm_prefetch || uvm_prefetch_advise)
    {
        over = sycl::malloc_shared<bool>(1, queue);
    }

    // events for timing
    sycl::event                                        tstart, tstop;
    std::chrono::time_point<std::chrono::steady_clock> tstart_ct1;
    std::chrono::time_point<std::chrono::steady_clock> tstop_ct1;
    float                                              elapsedTime;

    // setup execution parameters
    sycl::range<3> grid(1, 1, num_of_blocks);
    sycl::range<3> threads(1, 1, num_of_threads_per_block);

    double kernelTime = 0;
    int    k          = 0;
    bool   stop;
    // Call the Kernel until all the elements of Frontier are not false
    do
    {
        stop  = false;
        *over = stop;

        sycl::event kernel_event = queue.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                           [=](sycl::nd_item<3> item_ct1) {
                               Kernel(graph_nodes,
                                      graph_edges,
                                      graph_mask,
                                      updating_graph_mask,
                                      graph_visited,
                                      cost,
                                      no_of_nodes,
                                      item_ct1);
                           });
        kernel_event.wait();
        elapsedTime
            = kernel_event
                .get_profiling_info<sycl::info::event_profiling::command_end>()
            - kernel_event.get_profiling_info<
                sycl::info::event_profiling::command_start>();
        kernelTime += elapsedTime * 1.e-9;

        sycl::event kernel2_event = queue.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                           [=](sycl::nd_item<3> item_ct1) {
                               Kernel2(graph_mask,
                                       updating_graph_mask,
                                       graph_visited,
                                       over,
                                       no_of_nodes,
                                       item_ct1);
                           });
        kernel2_event.wait();
        elapsedTime
            = kernel2_event
                .get_profiling_info<sycl::info::event_profiling::command_end>()
            - kernel2_event.get_profiling_info<
                sycl::info::event_profiling::command_start>();
        kernelTime += elapsedTime * 1.e-9;

        stop = *over;
        k++;
    } while (stop);

    if (verbose && !quiet)
    {
        printf("Kernel Time: %f\n", kernelTime);
        printf("Kernel Executed %d times\n", k);
    }

    // copy result from device to host
    tstart_ct1 = std::chrono::steady_clock::now();
    if (uvm)
    {
        // Do nothing, cost stays on CPU
    }
    else if (uvm_advise)
    {
        queue.mem_advise(cost, sizeof(int) * no_of_nodes, 0);
        queue.mem_advise(cost, sizeof(int) * no_of_nodes, 0);
    }
    else if (uvm_prefetch)
    {
        queue.prefetch(cost, sizeof(int) * no_of_nodes);
    }
    else if (uvm_prefetch_advise)
    {
        queue.mem_advise(cost, sizeof(int) * no_of_nodes, 0);
        queue.mem_advise(cost, sizeof(int) * no_of_nodes, 0);
        queue.prefetch(cost, sizeof(int) * no_of_nodes);
    }
    else
    {
        std::cerr << "Unrecognized uvm option, exiting..." << std::endl;
        exit(-1);
    }
    tstop_ct1 = std::chrono::steady_clock::now();
    elapsedTime
        = std::chrono::duration<float, std::milli>(tstop_ct1 - tstart_ct1)
              .count();
    // transferTime += elapsedTime * 1.e-3; // convert to seconds

    // Store the result into a file.
    //
    string outfile = op.getOptionString("outputFile");
    if (outfile != "")
    {
        FILE *fpo = fopen(outfile.c_str(), "w");
        for (int i = 0; i < no_of_nodes; i++)
            fprintf(fpo, "%d) cost:%d\n", i, cost[i]);
        fclose(fpo);
    }

    // Cleanup memory.
    //
    sycl::free(graph_nodes, queue);
    sycl::free(graph_edges, queue);
    sycl::free(graph_mask, queue);
    sycl::free(updating_graph_mask, queue);
    sycl::free(graph_visited, queue);
    sycl::free(cost, queue);
    sycl::free(over, queue);

    char tmp[64];
    sprintf(tmp, "%dV,%dE", no_of_nodes, edge_list_size);
    string atts = string(tmp);
    resultDB.AddResult("bfs_unifiedmem_total_time", atts, "sec", kernelTime);
    resultDB.AddResult(
        "bfs_unifiedmem_rate_nodes", atts, "Nodes/s", no_of_nodes / kernelTime);
    resultDB.AddResult("bfs_unifiedmem_rate_edges",
                       atts,
                       "Edges/s",
                       edge_list_size / kernelTime);
    return kernelTime;
}
catch (sycl::exception const &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}
