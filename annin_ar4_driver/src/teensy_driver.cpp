#include "annin_ar4_driver/teensy_driver.hpp"

#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <type_traits>

#define FW_VERSION "2.1.0"

namespace annin_ar4_driver {

TeensyDriver::TeensyDriver() : serial_port_(io_service_) {}

bool TeensyDriver::init(std::string ar_model, std::string port, int baudrate,
                        int num_joints, bool velocity_control_enabled) {
  // version / model
  version_ = FW_VERSION;
  ar_model_ = ar_model;

  // establish connection with teensy board
  boost::system::error_code ec;
  serial_port_.open(port, ec);
  if (ec) {
    RCLCPP_WARN(logger_, "Failed to connect to serial port %s", port.c_str());
    return false;
  } else {
    serial_port_.set_option(
        boost::asio::serial_port_base::baud_rate(static_cast<uint32_t>(baudrate)));
    serial_port_.set_option(
        boost::asio::serial_port_base::parity(boost::asio::serial_port_base::parity::none));
    // ★ デバイスが落ち着くまで少し待機（初回取りこぼし対策）
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    RCLCPP_INFO(logger_, "Successfully connected to serial port %s", port.c_str());
  }

  initialised_ = false;

  // ★ 初回ハンドシェイクは CRLF 終端に変更（互換性対策）
  std::string msg = "STA" + version_ + "B" + ar_model_ + "\r\n";

  while (!initialised_) {
    RCLCPP_INFO(logger_, "Waiting for response from Teensy on port %s", port.c_str());
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    exchange(msg);
  }
  RCLCPP_INFO(logger_, "Successfully initialised driver on port %s", port.c_str());

  // initialise joint and encoder calibration
  num_joints_ = num_joints;
  joint_positions_deg_.resize(num_joints_);
  joint_velocities_deg_.resize(num_joints_);
  enc_calibrations_.resize(num_joints_);
  velocity_control_enabled_ = velocity_control_enabled;
  is_estopped_ = false;
  return true;
}

// Update between hardware interface and hardware driver
void TeensyDriver::update(std::vector<double>& pos_commands,
                          std::vector<double>& vel_commands,
                          std::vector<double>& joint_states,
                          std::vector<double>& joint_velocities) {
  // log pos_commands
  {
    std::string logInfo = "Joint Pos Cmd: ";
    for (int i = 0; i < num_joints_; i++) {
      std::stringstream ss;
      ss << std::fixed << std::setprecision(2) << pos_commands[i];
      logInfo += std::to_string(i) + ": " + ss.str() + " | ";
    }
    RCLCPP_DEBUG_THROTTLE(logger_, clock_, 500, "%s", logInfo.c_str());
  }

  // log vel_commands
  {
    std::string logInfo = "Joint Vel Cmd: ";
    for (int i = 0; i < num_joints_; i++) {
      std::stringstream ss;
      ss << std::fixed << std::setprecision(2) << vel_commands[i];
      logInfo += std::to_string(i) + ": " + ss.str() + " | ";
    }
    RCLCPP_DEBUG_THROTTLE(logger_, clock_, 500, "%s", logInfo.c_str());
  }

  // construct update message
  std::string outMsg;
  if (velocity_control_enabled_) {
    outMsg = "MV";
    for (int i = 0; i < num_joints_; ++i) {
      outMsg += char('A' + i);
      outMsg += std::to_string(vel_commands[i]);
    }
  } else {
    outMsg = "MT";
    for (int i = 0; i < num_joints_; ++i) {
      outMsg += char('A' + i);
      outMsg += std::to_string(pos_commands[i]);
    }
  }
  outMsg += "\n";

  // run the communication with board
  exchange(outMsg);

  // copy back states
  joint_states     = joint_positions_deg_;
  joint_velocities = joint_velocities_deg_;

  // print joint_states
  {
    std::string logInfo = "Joint Pos: ";
    for (int i = 0; i < num_joints_; i++) {
      std::stringstream ss;
      ss << std::fixed << std::setprecision(2) << joint_states[i];
      logInfo += std::to_string(i) + ": " + ss.str() + " | ";
    }
    RCLCPP_DEBUG_THROTTLE(logger_, clock_, 500, "%s", logInfo.c_str());
  }

  // print joint_velocities
  {
    std::string logInfo = "Joint Vel: ";
    for (int i = 0; i < num_joints_; i++) {
      std::stringstream ss;
      ss << std::fixed << std::setprecision(2) << joint_velocities[i];
      logInfo += std::to_string(i) + ": " + ss.str() + " | ";
    }
    RCLCPP_DEBUG_THROTTLE(logger_, clock_, 500, "%s", logInfo.c_str());
  }
}

// ★ キャリブレーション直前に待機し、失敗したら CRLF で 1 回だけリトライ
bool TeensyDriver::calibrateJoints() {
  std::this_thread::sleep_for(std::chrono::milliseconds(300));  // 落ち着き待ち
  std::string outMsg = "JC\n";
  RCLCPP_INFO(logger_, "Sending calibration command: %s", outMsg.c_str());
  if (sendCommand(outMsg)) {
    return true;
  }
  std::string retryMsg = "JC\r\n";
  RCLCPP_WARN(logger_, "Calibration LF failed, retrying with CRLF...");
  return sendCommand(retryMsg);
}

void TeensyDriver::getJointPositions(std::vector<double>& joint_positions) {
  std::string msg = "JP\n";
  exchange(msg);
  joint_positions = joint_positions_deg_;
}

void TeensyDriver::getJointVelocities(std::vector<double>& joint_velocities) {
  std::string msg = "JV\n";
  exchange(msg);
  joint_velocities = joint_velocities_deg_;
}

bool TeensyDriver::resetEStop() {
  std::string msg = "RE\n";
  exchange(msg);
  return !is_estopped_;
}

bool TeensyDriver::isEStopped() { return is_estopped_; }

bool TeensyDriver::sendCommand(std::string outMsg) { return exchange(outMsg); }

// Send msg to board and collect data
bool TeensyDriver::exchange(std::string outMsg) {
  std::string inMsg;
  std::string errTransmit;

  if (!transmit(outMsg, errTransmit)) {
    RCLCPP_ERROR(logger_, "Error in transmit: %s", errTransmit.c_str());
    return false;
  }

  // receive single-line response and dispatch
  while (true) {
    receive(inMsg);
    if (inMsg.size() < 2) {
      RCLCPP_WARN(logger_, "Short line: '%s'", inMsg.c_str());
      continue;
    }
    std::string header = inMsg.substr(0, 2);

    if (header == "DB") {
      RCLCPP_DEBUG(logger_, "Debug: %s", inMsg.c_str());
      continue;
    } else if (header == "WN") {
      RCLCPP_WARN(logger_, "Warning: %s", inMsg.c_str());
      continue;
    } else if (header == "ER") {
      RCLCPP_INFO(logger_, "ERROR message: %s", inMsg.c_str());
      return false;
    } else if (header == "ST") {
      checkInit(inMsg);
      return true;
    } else if (header == "JC") {
      updateEncoderCalibrations(inMsg);
      return true;
    } else if (header == "JP") {
      updateJointPositions(inMsg);
      return true;
    } else if (header == "JV") {
      updateJointVelocities(inMsg);
      return true;
    } else if (header == "ES") {
      updateEStopStatus(inMsg);
      return true;
    } else {
      RCLCPP_WARN(logger_, "Unknown header %s", header.c_str());
      return false;
    }
  }
}

bool TeensyDriver::transmit(std::string msg, std::string& err) {
  boost::system::error_code ec;
  const auto sendBuffer = boost::asio::buffer(msg.c_str(), msg.size());
  boost::asio::write(serial_port_, sendBuffer, ec);
  if (!ec) return true;
  err = ec.message();
  return false;
}

void TeensyDriver::receive(std::string& inMsg) {
  char c;
  std::string msg;
  bool eol = false;
  while (!eol) {
    boost::asio::read(serial_port_, boost::asio::buffer(&c, 1));
    switch (c) {
      case '\r': break;           // ignore CR
      case '\n': eol = true; break;
      default: msg += c; break;
    }
  }
  inMsg = msg;
}

// Safely parse ST line with explicit field bounds
void TeensyDriver::checkInit(std::string msg) {
  // Format: "ST" + "A"<ack> + "B"<version> + "C"<ar_model_matched> + "D"<ar_model>
  std::size_t a = msg.find('A', 2);
  std::size_t b = msg.find('B', 2);
  std::size_t c = msg.find('C', 2);
  std::size_t d = msg.find('D', 2);

  if (a == std::string::npos || b == std::string::npos ||
      c == std::string::npos || d == std::string::npos ||
      !(a < b && b < c && c < d)) {
    RCLCPP_WARN(logger_, "Malformed ST line: %s", msg.c_str());
    return;
  }

  int ack = 0;
  int ar_model_matched = 0;
  std::string version;
  std::string ar_model;

  try {
    ack               = std::stoi(msg.substr(a + 1, b - (a + 1)));
    version           = msg.substr(b + 1, c - (b + 1));
    ar_model_matched  = std::stoi(msg.substr(c + 1, d - (c + 1)));
    ar_model          = msg.substr(d + 1);
  } catch (const std::exception& e) {
    RCLCPP_WARN(logger_, "Failed to parse ST line: %s (err=%s)", msg.c_str(), e.what());
    return;
  }

  if (!ack) {
    RCLCPP_ERROR(logger_, "Firmware version mismatch %s", version.c_str());
  }
  if (!ar_model_matched) {
    RCLCPP_ERROR(logger_, "Model mismatch %s", ar_model.c_str());
  }
  if (ack && ar_model_matched) {
    initialised_ = true;
  }
}

void TeensyDriver::updateJointPositions(const std::string msg) {
  parseValuesToVector(msg, joint_positions_deg_);
}

void TeensyDriver::updateJointVelocities(const std::string msg) {
  parseValuesToVector(msg, joint_velocities_deg_);
}

void TeensyDriver::updateEStopStatus(std::string msg) {
  is_estopped_ = msg.substr(2) == "1";
}

void TeensyDriver::updateEncoderCalibrations(const std::string msg) {
  parseValuesToVector(msg, enc_calibrations_);
}

template <typename T>
void TeensyDriver::parseValuesToVector(const std::string msg,
                                       std::vector<T>& values) {
  values.clear();
  size_t prevIdx = msg.find('A', 2);
  if (prevIdx == std::string::npos) return;
  ++prevIdx;  // start after 'A'

  for (size_t i = 1;; ++i) {
    char currentIdentifier = char('A' + i);
    size_t currentIdx = msg.find(currentIdentifier, 2);

    try {
      if (currentIdx == std::string::npos) {
        if constexpr (std::is_same<T, int>::value) {
          values.push_back(std::stoi(msg.substr(prevIdx)));
        } else if constexpr (std::is_same<T, double>::value) {
          values.push_back(std::stod(msg.substr(prevIdx)));
        }
        break;
      }
      if constexpr (std::is_same<T, int>::value) {
        values.push_back(std::stoi(msg.substr(prevIdx, currentIdx - prevIdx)));
      } else if constexpr (std::is_same<T, double>::value) {
        values.push_back(std::stod(msg.substr(prevIdx, currentIdx - prevIdx)));
      }
    } catch (const std::invalid_argument&) {
      RCLCPP_WARN(logger_, "Invalid argument, can't parse %s", msg.c_str());
    }
    prevIdx = currentIdx + 1;
  }
}

}  // namespace annin_ar4_driver
