#include "annin_ar4_driver/teensy_driver.hpp"

#include <chrono>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#define FW_VERSION "2.1.0"

namespace annin_ar4_driver {

TeensyDriver::TeensyDriver() : serial_port_(io_service_) {}

bool TeensyDriver::init(std::string ar_model, std::string port, int baudrate,
                        int num_joints, bool velocity_control_enabled) {
  // version / model
  version_  = FW_VERSION;
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
    RCLCPP_INFO(logger_, "Successfully connected to serial port %s", port.c_str());
  }

  initialised_ = false;

  // 初回ハンドシェイク（LF 統一）
  std::string msg = "STA" + version_ + "B" + ar_model_ + "\n";

  while (!initialised_) {
    RCLCPP_INFO(logger_, "Waiting for response from Teensy on port %s", port.c_str());
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    exchange(msg);  // ST 応答で initialised_ を立てる
  }
  RCLCPP_INFO(logger_, "Successfully initialised driver on port %s", port.c_str());

  // initialise joint and encoder calibration
  num_joints_ = num_joints;
  joint_positions_deg_.resize(num_joints_);
  joint_velocities_deg_.resize(num_joints_);
  enc_calibrations_.resize(num_joints_);
  velocity_control_enabled_ = velocity_control_enabled;
  is_estopped_              = false;
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

  // construct update message (LF 統一)
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

// Calibration (LF 統一)
bool TeensyDriver::calibrateJoints() {
  std::string outMsg = "JC\n";
  RCLCPP_INFO(logger_, "Sending calibration command: %s", outMsg.c_str());
  return sendCommand(outMsg);
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

// ---------- 受信補助：既知ヘッダ検出 & フレーム分割 ----------
static inline bool is_known_header(const std::string& s, size_t i) {
  if (i + 1 >= s.size()) return false;
  auto tag = s.substr(i, 2);
  return (tag == "ST" || tag == "JC" || tag == "JP" || tag == "JV" ||
          tag == "ES" || tag == "DB" || tag == "WN" || tag == "ER");
}

// Send msg to board and collect data
bool TeensyDriver::exchange(std::string outMsg) {
  std::string inMsg;
  std::string errTransmit;

  if (!transmit(outMsg, errTransmit)) {
    RCLCPP_ERROR(logger_, "Error in transmit: %s", errTransmit.c_str());
    return false;
  }

  // 1行ずつ受け、必要なら行内を既知ヘッダで分割して順次処理
  while (true) {
    receive(inMsg);
    if (inMsg.empty()) {
      RCLCPP_DEBUG(logger_, "Empty line received");
      continue;
    }

    // 行頭2文字がヘッダでない場合や、複数フレーム縦続きの場合に備え、
    // 行中の全ての既知ヘッダ開始位置を拾ってフレーム化
    std::vector<size_t> idxs;
    for (size_t i = 0; i + 1 < inMsg.size(); ++i) {
      if (is_known_header(inMsg, i)) idxs.push_back(i);
    }
    if (idxs.empty()) {
      RCLCPP_DEBUG(logger_, "No known header in: '%s' (waiting next line)", inMsg.c_str());
      continue;  // ヘッダ無しはスキップして次行へ
    }
    idxs.push_back(inMsg.size());  // 末尾番兵

    bool handled_any = false;

    for (size_t k = 0; k + 1 < idxs.size(); ++k) {
      size_t beg = idxs[k];
      size_t end = idxs[k + 1];
      const std::string frame  = inMsg.substr(beg, end - beg);
      const std::string header = frame.substr(0, 2);

      if (header == "DB") {
        RCLCPP_DEBUG(logger_, "Debug: %s", frame.c_str());
        continue;  // ログはスキップ
      } else if (header == "WN") {
        RCLCPP_WARN(logger_, "Warning: %s", frame.c_str());
        continue;  // 警告もスキップ
      } else if (header == "ER") {
        RCLCPP_INFO(logger_, "ERROR message: %s", frame.c_str());
        return false;
      } else if (header == "ST") {
        checkInit(frame);
        handled_any = true;
        if (initialised_) return true;  // 初期化の本命
      } else if (header == "JC") {
        updateEncoderCalibrations(frame);
        handled_any = true;
      } else if (header == "JP") {
        updateJointPositions(frame);
        handled_any = true;
      } else if (header == "JV") {
        updateJointVelocities(frame);
        handled_any = true;
      } else if (header == "ES") {
        updateEStopStatus(frame);
        handled_any = true;
      } else {
        // ここは原則来ないが、来ても次行へ
        RCLCPP_WARN(logger_, "Unknown header %s in frame '%s'", header.c_str(), frame.c_str());
        continue;
      }
    }

    if (handled_any) return true;

    // ここに来るのは異常系（既知ヘッダがあったのにどれも処理されなかった）
    RCLCPP_WARN(logger_, "Unhandled frame(s) in line: '%s'", inMsg.c_str());
    // 次行へ継続（強制終了しない）
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
  // 1行読み：CR でも LF でも終端扱い（JVAES0等の連結を回避）
  char c;
  std::string msg;
  bool eol = false;
  while (!eol) {
    boost::asio::read(serial_port_, boost::asio::buffer(&c, 1));
    switch (c) {
      case '\r': eol = true; break;  // CRで切る
      case '\n': eol = true; break;  // LFで切る
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
    ack              = std::stoi(msg.substr(a + 1, b - (a + 1)));
    version          = msg.substr(b + 1, c - (b + 1));
    ar_model_matched = std::stoi(msg.substr(c + 1, d - (c + 1)));
    ar_model         = msg.substr(d + 1);
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

// --- Robust value parsing helpers ---
static inline void trim_spaces_bounds(const std::string& s, size_t& beg, size_t& end) {
  while (beg < end && std::isspace(static_cast<unsigned char>(s[beg]))) ++beg;
  while (end > beg && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
}

template <typename T>
void TeensyDriver::parseValuesToVector(const std::string msg,
                                       std::vector<T>& values) {
  values.clear();
  size_t prevIdx = msg.find('A', 2);  // first label after header (A)
  if (prevIdx == std::string::npos) return;
  ++prevIdx;  // start after 'A'

  for (size_t i = 1;; ++i) {
    const char currentIdentifier = char('A' + i);
    const size_t currentIdx      = msg.find(currentIdentifier, 2);

    try {
      if (currentIdx == std::string::npos) {
        // last slice
        size_t beg = prevIdx, end = msg.size();
        trim_spaces_bounds(msg, beg, end);
        if (beg < end) {
          if constexpr (std::is_same<T, int>::value) {
            values.push_back(std::stoi(msg.substr(beg, end - beg)));
          } else if constexpr (std::is_same<T, double>::value) {
            values.push_back(std::stod(msg.substr(beg, end - beg)));
          }
        }
        break;
      } else {
        size_t beg = prevIdx, end = currentIdx;
        trim_spaces_bounds(msg, beg, end);
        if (beg < end) {
          if constexpr (std::is_same<T, int>::value) {
            values.push_back(std::stoi(msg.substr(beg, end - beg)));
          } else if constexpr (std::is_same<T, double>::value) {
            values.push_back(std::stod(msg.substr(beg, end - beg)));
          }
        } else {
          RCLCPP_WARN(logger_, "Empty numeric field around '%c' in '%s'",
                      currentIdentifier, msg.c_str());
        }
      }
    } catch (const std::invalid_argument&) {
      RCLCPP_WARN(logger_, "Invalid argument, can't parse '%s'", msg.c_str());
    } catch (const std::out_of_range&) {
      RCLCPP_WARN(logger_, "Out of range, can't parse '%s'", msg.c_str());
    }
    prevIdx = currentIdx + 1;
  }
}

void TeensyDriver::updateJointPositions(const std::string msg) {
  parseValuesToVector(msg, joint_positions_deg_);
}

void TeensyDriver::updateJointVelocities(const std::string msg) {
  parseValuesToVector(msg, joint_velocities_deg_);
}

void TeensyDriver::updateEStopStatus(std::string msg) {
  is_estopped_ = (msg.size() >= 3) && (msg.substr(2) == "1");
}

void TeensyDriver::updateEncoderCalibrations(const std::string msg) {
  parseValuesToVector(msg, enc_calibrations_);
}

}  // namespace annin_ar4_driver
