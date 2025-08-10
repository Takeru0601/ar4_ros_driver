#include "annin_ar4_driver/teensy_driver.hpp"

#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <thread>

#define FW_VERSION "2.1.0C1Dmk3"

namespace annin_ar4_driver {

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
    serial_port_.set_option(boost::asio::serial_port_base::baud_rate(
        static_cast<uint32_t>(baudrate)));
    serial_port_.set_option(boost::asio::serial_port_base::parity(
        boost::asio::serial_port_base::parity::none));
    RCLCPP_INFO(logger_, "Successfully connected to serial port %s",
                port.c_str());
  }

  // device settle: some boards miss first bytes right after open
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  initialised_ = false;

  // first handshake uses CRLF for compatibility
  std::string msg = "STA" + version_ + "B" + ar_model_ + "\r\n";

  while (!initialised_) {
    RCLCPP_INFO(logger_, "Waiting for response from Teensy on port %s",
                port.c_str());
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    exchange(msg);
  }
  RCLCPP_INFO(logger_, "Successfully initialised driver on port %s",
              port.c_str());

  // initialise joint and encoder calibration
  num_joints_ = num_joints;
  joint_positions_deg_.resize(num_joints_);
  joint_velocities_deg_.resize(num_joints_);
  enc_calibrations_.resize(num_joints_);
  velocity_control_enabled_ = velocity_control_enabled;
  is_estopped_ = false;
  return true;
}

TeensyDriver::TeensyDriver() : serial_port_(io_service_) {}

// Update between hardware interface and hardware driver
void TeensyDriver::update(std::vector<double>& pos_commands,
                          std::vector<double>& vel_commands,
                          std::vector<double>& joint_positions,
                          std::vector<double>& joint_velocities) {
  // log pos_commands
  std::string logInfo = "Joint Pos Cmd: ";
  for (int i = 0; i < num_joints_; i++) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << pos_commands[i];
    logInfo += std::to_string(i) + ": " + ss.str() + " | ";
  }
  RCLCPP_DEBUG_THROTTLE(logger_, clock_, 500, logInfo.c_str());

  // log vel_commands
  logInfo = "Joint Vel Cmd: ";
  for (int i = 0; i < num_joints_; i++) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << vel_commands[i];
    logInfo += std::to_string(i) + ": " + ss.str() + " | ";
  }
  RCLCPP_DEBUG_THROTTLE(logger_, clock_, 500, logInfo.c_str());

  // construct update message
  std::string outMsg;
  if (velocity_control_enabled_) {
    outMsg += "MV";
    for (int i = 0; i < num_joints_; ++i) {
      outMsg += 'A' + i;
      outMsg += std::to_string(vel_commands[i]);
    }
  } else {
    outMsg += "MT";
    for (int i = 0; i < num_joints_; ++i) {
      outMsg += 'A' + i;
      outMsg += std::to_string(pos_commands[i]);
    }
  }
  outMsg += "\n";

  // run the communication with board
  exchange(outMsg);

  // copy back states
  joint_positions = joint_positions_deg_;
  joint_velocities = joint_velocities_deg_;

  // print joint_positions
  logInfo = "Joint Pos: ";
  for (int i = 0; i < num_joints_; i++) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << joint_positions[i];
    logInfo += std::to_string(i) + ": " + ss.str() + " | ";
  }
  RCLCPP_DEBUG_THROTTLE(logger_, clock_, 500, logInfo.c_str());

  // print joint_velocities
  logInfo = "Joint Vel: ";
  for (int i = 0; i < num_joints_; i++) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << joint_velocities[i];
    logInfo += std::to_string(i) + ": " + ss.str() + " | ";
  }
  RCLCPP_DEBUG_THROTTLE(logger_, clock_, 500, logInfo.c_str());
}

bool TeensyDriver::calibrateJoints() {
  std::string outMsg = "JC\n";  // protocol: no sequence argument
  RCLCPP_INFO(logger_, "Sending calibration command: %s", outMsg.c_str());
  return sendCommand(outMsg);
}

void TeensyDriver::getJointPositions(std::vector<double>& joint_positions) {
  // get current joint positions
  std::string msg = "JP\n";
  exchange(msg);
  joint_positions = joint_positions_deg_;
}

bool TeensyDriver::resetEStop() {
  std::string msg = "RE\n";
  exchange(msg);
  return !is_estopped_;
}

bool TeensyDriver::isEStopped() { return is_estopped_; }

void TeensyDriver::getJointVelocities(std::vector<double>& joint_velocities) {
  // get current joint velocities
  std::string msg = "JV\n";
  exchange(msg);
  joint_velocities = joint_velocities_deg_;
}

bool TeensyDriver::sendCommand(std::string outMsg) { return exchange(outMsg); }

// Send msg to board and collect data
bool TeensyDriver::exchange(std::string outMsg) {
  std::string inMsg;
  std::string errTransmit = "";

  // send
  if (!transmit(outMsg, errTransmit)) {
    RCLCPP_ERROR(logger_, "Error in transmit: %s", errTransmit.c_str());
    return false;
  }

  // receive single-line response and dispatch
  while (true) {
    receive(inMsg);
    // Always dump the received line for visibility
    RCLCPP_INFO(logger_, "RX: %s", inMsg.c_str());

    if (inMsg.size() < 2) {
      RCLCPP_WARN(logger_, "Short line: '%s'", inMsg.c_str());
      continue;
    }
    std::string header = inMsg.substr(0, 2);

    if (header == "DB") {
      RCLCPP_DEBUG(logger_, "Debug message: %s", inMsg.c_str());
    } else if (header == "WN") {
      RCLCPP_WARN(logger_, "Warning: %s", inMsg.c_str());
    } else {
      if (header == "ST") {
        // init acknowledgement
        checkInit(inMsg);
      } else if (header == "JC") {
        // encoder calibration values
        updateEncoderCalibrations(inMsg);
      } else if (header == "JP") {
        // joint positions (deg)
        updateJointPositions(inMsg);
      } else if (header == "JV") {
        // joint velocities (deg/s)
        updateJointVelocities(inMsg);
      } else if (header == "ES") {
        // estop status
        updateEStopStatus(inMsg);
      } else if (header == "ER") {
        // error message
        RCLCPP_INFO(logger_, "ERROR message: %s", inMsg.c_str());
        return false;
      } else {
        // unknown header
        RCLCPP_WARN(logger_, "Unknown header %s", header.c_str());
        return false;
      }
      return true;
    }
  }
  return true;
}

bool TeensyDriver::transmit(std::string msg, std::string& err) {
  boost::system::error_code ec;
  const auto sendBuffer = boost::asio::buffer(msg.c_str(), msg.size());
  boost::asio::write(serial_port_, sendBuffer, ec);

  if (!ec) {
    return true;
  } else {
    err = ec.message();
    return false;
  }
}

void TeensyDriver::receive(std::string& inMsg) {
  char c;
  std::string msg = "";
  bool eol = false;
  while (!eol) {
    boost::asio::read(serial_port_, boost::asio::buffer(&c, 1));
    switch (c) {
      case '\r':
        // ignore CR, wait for LF
        break;
      case '\n':
        eol = true;
        break;
      default:
        msg += c;
    }
  }
  inMsg = msg;
}

// Safely parse ST line with explicit field bounds
void TeensyDriver::checkInit(std::string msg) {
  // Format: "ST" + "A" + <ack> + "B" + <version> + "C" + <ar_model_matched> + "D" + <ar_model>
  auto a = msg.find('A', 2);
  auto b = msg.find('B', 2);
  auto c = msg.find('C', 2);
  auto d = msg.find('D', 2);

  if (a == std::string::npos || b == std::string::npos ||
      c == std::string::npos || d == std::string::npos ||
      b <= a || c <= b || d <= c) {
    RCLCPP_WARN(logger_, "Malformed ST line: %s", msg.c_str());
    return;
  }

  int ack = 0;
  int ar_model_matched = 0;
  std::string version;
  std::string ar_model;

  try {
    ack = std::stoi(msg.substr(a + 1, b - (a + 1)));
    version = msg.substr(b + 1, c - (b + 1));
    ar_model_matched = std::stoi(msg.substr(c + 1, d - (c + 1)));
    ar_model = msg.substr(d + 1);
  } catch (const std::exception& e) {
    RCLCPP_WARN(logger_, "Failed to parse ST line: %s (err=%s)", msg.c_str(),
                e.what());
    return;
  }
