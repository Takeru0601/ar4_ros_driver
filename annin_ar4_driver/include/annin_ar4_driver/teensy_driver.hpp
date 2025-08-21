#ifndef ANNIN_AR4_DRIVER_TEENSY_DRIVER_HPP
#define ANNIN_AR4_DRIVER_TEENSY_DRIVER_HPP

#include <boost/asio.hpp>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <vector>

namespace annin_ar4_driver {

class TeensyDriver {
public:
  TeensyDriver();

  bool init(std::string ar_model,
            std::string port,
            int baudrate,
            int num_joints,
            bool velocity_control_enabled);

  void update(std::vector<double>& pos_commands,
              std::vector<double>& vel_commands,
              std::vector<double>& joint_positions,
              std::vector<double>& joint_velocities);

  // --- Commands ---
  // 安全デフォルト（モード3・順序0..N-1）で送る
  bool calibrateJoints();

  // 明示シーケンスを使う版（互換維持）
  bool calibrateJoints(const std::string& calib_sequence);

  void getJointPositions(std::vector<double>& joint_positions);
  void getJointVelocities(std::vector<double>& joint_velocities);
  bool resetEStop();
  bool isEStopped();
  bool sendCommand(std::string outMsg);

private:
  // --- Low-level I/O ---
  bool exchange(std::string outMsg);
  bool transmit(std::string msg, std::string& err);
  void receive(std::string& inMsg);

  // --- Parsing helpers ---
  void checkInit(std::string msg);

  template <typename T>
  void parseValuesToVector(const std::string msg, std::vector<T>& values);

  void updateEncoderCalibrations(const std::string msg);
  void updateJointPositions(const std::string msg);
  void updateJointVelocities(const std::string msg);
  void updateEStopStatus(std::string msg);

  // --- Utility ---
  std::string composeCalibrationSequence(int mode) const; // '3' + '012...(N-1)'

  // --- State ---
  rclcpp::Logger logger_{rclcpp::get_logger("teensy_driver")};
  rclcpp::Clock  clock_{RCL_SYSTEM_TIME};

  bool initialised_{false};
  std::string ar_model_;
  std::string version_;

  boost::asio::io_service io_service_;
  boost::asio::serial_port serial_port_{io_service_};

  int num_joints_{0};
  std::vector<double> joint_positions_deg_;
  std::vector<double> joint_velocities_deg_;
  std::vector<int>    enc_calibrations_;
  bool velocity_control_enabled_{true};
  bool is_estopped_{false};
};

}  // namespace annin_ar4_driver

#endif  // ANNIN_AR4_DRIVER_TEENSY_DRIVER_HPP
