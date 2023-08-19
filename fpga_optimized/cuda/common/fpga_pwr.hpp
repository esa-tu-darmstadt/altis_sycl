#pragma once

#include <array>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

std::string exec(const char *cmd) {
  std::array<char, 16 * 1024> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  return result;
}

} // namespace

static double get_fpga_board_pwr_w() {
  std::string result = exec("bittware_power -c0");
  std::string tbp = "acl0: ";

  int idx = result.find(tbp);
  if (idx < 0)
    return -1.0;

  return strtod(result.c_str() + idx + tbp.size(), nullptr);
}

static volatile bool pollThreadStatus = false;
static pthread_t powerPollThread;
constexpr int32_t ms_target_per_iteration = 20; // Set to 0 for unlimited

/*
Poll the FPGA using the nallatech tool.
*/
static void *powerPollingFunc(void *) {

  unsigned int powerLevel = 0;
  FILE *fp = fopen("fpga_pwr.txt", "a+");

  static std::vector<unsigned int> pwr_vals;
  pwr_vals.resize(10000);

  const auto thread_begin = std::chrono::high_resolution_clock::now();

  int pwr_val_cnt = 0;
  do {
    pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, 0);

    const auto iteration_begin = std::chrono::high_resolution_clock::now();

    const double powerLevel = get_fpga_board_pwr_w();
    if (powerLevel > .0) {
      pwr_vals[pwr_val_cnt] = powerLevel;
      pwr_val_cnt = (pwr_val_cnt + 1) % 10000;
    }

    const auto iteration_end = std::chrono::high_resolution_clock::now();
    const auto it_duration_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(iteration_begin -
                                                              iteration_end)
            .count();

    if (it_duration_ms < ms_target_per_iteration)
      std::this_thread::sleep_for(
          std::chrono::milliseconds(ms_target_per_iteration - it_duration_ms));

    pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
  } while (pollThreadStatus);

  const auto thread_end = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < pwr_val_cnt; i++)
    fprintf(fp, "%.3lf\n", double(pwr_vals[i]));

  fprintf(fp, "AVG %.3lf\n",
          std::chrono::duration_cast<std::chrono::milliseconds>(thread_end -
                                                                thread_begin)
                  .count() /
              double(pwr_val_cnt));
  // std::cout << "Values got: " << pwr_val_cnt << std::endl;
  // std::cout << "Avg Iteration Duration: "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(
  //                  thread_end - thread_begin)
  //                      .count() /
  //                  double(pwr_val_cnt)
  //           << "ms" << std::endl;

  fclose(fp);
  pthread_exit(0);
}

/*
Start power measurement by spawning a pthread that polls the FPGA.
*/
static void pwr_measurement_start() {
  pollThreadStatus = true;

  int iret = pthread_create(&powerPollThread, NULL, powerPollingFunc, nullptr);
  if (iret) {
    fprintf(stderr, "Error - pthread_create() return code: %d\n", iret);
    exit(0);
  }
}

/*
End power measurement. This ends the polling thread.
*/
static void pwr_measurement_end() {
  pollThreadStatus = false;
  pthread_join(powerPollThread, NULL);
}

#ifdef _FPGA
#define FPGA_PWR_MEAS_START // pwr_measurement_start();
#define FPGA_PWR_MEAS_END   // pwr_measurement_end();
#else
#define FPGA_PWR_MEAS_START
#define FPGA_PWR_MEAS_END
#endif
