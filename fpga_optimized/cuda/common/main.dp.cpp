////////////////////////////////////////////////////////////////////////////////////////////////////
// file:	altis\src\cuda\common\main.cpp
//
// summary:	Implements the main class
// 
// origin: SHOC (https://github.com/vetter/shoc)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
#include <iostream>
#include <cstdlib>
#include <fstream>

#include "ResultDatabase.h"
#include "OptionParser.h"
#include "Utility.h"
#include "cudacommon.h"

using namespace std;

// Forward Declarations
void addBenchmarkSpecOptions(OptionParser &op);
void RunBenchmark(ResultDatabase &resultDB, OptionParser &op, size_t device_idx);

static std::string 
get_device_type_string(const cl::sycl::device& d) 
{
    const auto t = d.get_info<cl::sycl::info::device::device_type>();
    switch (t)
    {
        case cl::sycl::info::device_type::cpu:
            return "CPU";
        case cl::sycl::info::device_type::gpu:
            return "GPU";
        case cl::sycl::info::device_type::accelerator:
            return "Accelerator";
        default:
            return "Other";
    }
}

static void 
print_device_params(const cl::sycl::device& dev) 
{
        std::cout 
            << "  Name:            " << dev.get_info<cl::sycl::info::device::name>() << endl
            << "  Vendor:          " << dev.get_info<cl::sycl::info::device::vendor>() << endl
            // << "  OpenCL:          " << dev.get_info<cl::sycl::info::device::opencl_c_version>() << endl
            // << "  DeviceType:      " << get_device_type_string(dev) << endl
            // << "  MaxWorkGroupSize:" << dev.get_info<cl::sycl::info::device::max_work_group_size>() << endl
            // << "  MaxComputeUnits: " << dev.get_info<cl::sycl::info::device::max_compute_units>() << endl
            // << "  GlobalMem:       " << dev.get_info<cl::sycl::info::device::global_mem_size>() << endl
            // << "  LocalMem:        " << dev.get_info<cl::sycl::info::device::local_mem_size>() << endl
            << endl;
}

size_t 
EnumerateDevicesAndChoose(bool use_gpu, bool use_fpga, bool use_fpga_emu, bool properties, bool quiet)
{
    int64_t first_cpu = -1;
    int64_t first_gpu = -1;
    int64_t first_fpga = -1;
    int64_t first_fpga_emu = -1;

    std::vector<cl::sycl::device> devices = cl::sycl::device::get_devices();
    for(size_t d = 0ull; d < devices.size(); d++)
    {
        const auto& dev = devices[d];
        switch (dev.get_info<cl::sycl::info::device::device_type>())
        {
            case cl::sycl::info::device_type::cpu:
                if (first_cpu < 0) first_cpu = d;
                break;
            case cl::sycl::info::device_type::gpu:
                if (first_gpu < 0) first_gpu = d;
                break;
            case cl::sycl::info::device_type::accelerator:
                if (dev.get_info<cl::sycl::info::device::name>().find("Emu") != std::string::npos)
                {
                    if (first_fpga_emu < 0) first_fpga_emu = d;
                }
                else
                {
                    if (first_fpga < 0) first_fpga = d;
                }
                break;
            default:
                break;
        }

        if (properties) 
        {
            std::cout << "Device " << d << std::endl;
            print_device_params(dev);
        }
    }

    auto print_and_set = [&](int64_t d) {
        if (d < 0)
        {
            std::cerr << "No device found for given execution params!" << std::endl;
            safe_exit(-1);
        }
        std::cout << "Execute candidate for benchmark is" << std::endl;
        print_device_params(devices[size_t(d)]);
        return d;
    };

    if (use_fpga_emu)
        return print_and_set(first_fpga_emu);
    else if (use_fpga)
        return print_and_set(first_fpga);
    else if (use_gpu)
        return print_and_set(first_gpu);
    else
        return print_and_set(first_cpu);
}

void 
checkCudaFeatureAvailability(OptionParser &op) {
    // Check UVM availability
    if (op.getOptionBool("uvm") || op.getOptionBool("uvm-advise") ||
            op.getOptionBool("uvm-prefetch") || op.getOptionBool("uvm-prefetch-advise")) {
        // Always supported in SYCL.
    }

    // Check Cooperative Group availability
    if (op.getOptionBool("coop")) {
        std::cerr << "Cooperative Groups currently not supported in sycl-build..." << std::endl;
        safe_exit(-1);
    }

    // Check Dynamic Parallelism availability
    if (op.getOptionBool("dyn")) {
        std::cerr << "Dynamic Parallelis currently not supported in sycl-build..." << std::endl;
        safe_exit(-1);
    }

    // Check CUDA Graphs availability
    if (op.getOptionBool("graph")) {
        std::cerr << "Graph parameter currently not supported in sycl-build..." << std::endl;
        safe_exit(-1);
    }
}

int 
main(int argc, char *argv[])
{
    int ret = 0;

    try
    {
        // Get args
        OptionParser op;

        // Add shared options to the parser
        op.addOption("properties", OPT_BOOL, "0",
                "show properties for available platforms and devices", 'p');
        op.addOption("device", OPT_VECINT, "0",
                "specify device(s) to run on", 'd');
        op.addOption("passes", OPT_INT, "10", "specify number of passes", 'n');
        op.addOption("size", OPT_INT, "1", "specify problem size", 's');
        op.addOption("verbose", OPT_BOOL, "0", "enable verbose output", 'v');
        op.addOption("quiet", OPT_BOOL, "0", "enable concise output", 'q');
        op.addOption("configFile", OPT_STRING, "", "path of configuration file", 'c');
        op.addOption("inputFile", OPT_STRING, "", "path of input file", 'i');
        op.addOption("outputFile", OPT_STRING, "", "path of output file", 'o');
        op.addOption("metricsFile", OPT_STRING, "", "path of file to write metrics to", 'm');

        // Add options for turn on/off CUDA features
        op.addOption("uvm", OPT_BOOL, "0", "enable CUDA Unified Virtual Memory, only demand paging");
        op.addOption("uvm-advise", OPT_BOOL, "0", "guide the driver about memory usage patterns");
        op.addOption("uvm-prefetch", OPT_BOOL, "0", "prefetch memory the specified destination device");
        op.addOption("uvm-prefetch-advise", OPT_BOOL, "0", "prefetch memory the specified destination device with memory guidance on");
        op.addOption("coop", OPT_BOOL, "0", "enable CUDA Cooperative Groups");
        op.addOption("dyn", OPT_BOOL, "0", "enable CUDA Dynamic Parallelism");
        op.addOption("graph", OPT_BOOL, "0", "enable CUDA Graphs");

        // Add some additional options for the dpcpp-build
        op.addOption("gpu", OPT_BOOL, "0", "use gpu for kernels");
        op.addOption("fpga", OPT_BOOL, "0", "use fpga for kernels");
        op.addOption("fpga_emu", OPT_BOOL, "0", "use fpga emulator for kernels");

        op.addOption("cgemm", OPT_BOOL, "0", "use custom gemm instead of OneMKL");

        addBenchmarkSpecOptions(op);

        if (!op.parse(argc, argv))
        {
            op.usage();
            return (op.HelpRequested() ? 0 : 1);
        }

        bool properties = op.getOptionBool("properties");
        bool quiet = op.getOptionBool("quiet");
        string metricsfile = op.getOptionString("metricsFile");

        // Initialization
        bool gpu = op.getOptionBool("gpu");
        bool fpga = op.getOptionBool("fpga");
        bool fpga_emu = op.getOptionBool("fpga_emu");
        size_t device = EnumerateDevicesAndChoose(gpu, fpga, fpga_emu, properties, quiet);
        if (properties)
            return 0;

        // Check CUDA feature availability
        checkCudaFeatureAvailability(op);

        ResultDatabase resultDB;

        // Run the benchmark
        RunBenchmark(resultDB, op, device);

        // If quiet, output overall result
        // else output metrics
        if (quiet) {
            resultDB.DumpOverall();
        } else {
            if (metricsfile.empty()) {
                cout << endl;
                resultDB.DumpSummary(cout);
            } else {
                ofstream ofs;
                ofs.open(metricsfile.c_str());
                resultDB.DumpCsv(metricsfile);
                ofs.close();
            }
        }
    }
    catch( std::exception& e )
    {
        std::cerr << "Caught exception in benchmark: " << e.what() << std::endl;
        ret = 1;
    }
    catch( ... )
    {
        std::cerr << "Caught unkown exception in benchmark." << std::endl;
        ret = 1;
    }

    std::cout << "Successfully out of main." << std::endl;

    return ret;
}
