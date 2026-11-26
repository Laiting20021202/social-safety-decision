#include <algorithm>
#include <array>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <hardware_interface/system_interface.hpp>
#include <openarm/canbus/can_socket.hpp>
#include <openarm/damiao_motor/dm_motor_control.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>

namespace openarm_right_curl {
namespace dm = openarm::damiao_motor;
using Encoder = dm::CanPacketEncoder;
using Decoder = dm::CanPacketDecoder;
using Result = hardware_interface::return_type;
using Callback = hardware_interface::CallbackReturn;
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;

class RightArm final : public hardware_interface::SystemInterface {
  static constexpr size_t elbow = 3;
  static constexpr size_t axes = 7;
  static constexpr size_t devices = 8;
  static constexpr const char *command_key = "openarm_right_joint4/position";
  const rclcpp::Logger log_ = rclcpp::get_logger("openarm_encoder_interface");
  bool right_ = false, powered_ = false;
  bool commissioned_tabletop_ = false;
  std::unique_ptr<openarm::canbus::CANSocket> bus_;
  std::vector<dm::Motor> motors_;
  std::array<double, devices> pos_{}, vel_{}, effort_{}, velocity_scale_{};
  std::array<int, devices> status_{};
  std::array<Clock::time_point, devices> received_{};
  std::array<double, axes> start_{}, kp_{}, kd_{};
  std::array<double, axes> commissioned_start_{};
  double command_ = NAN, target_ = NAN;

  bool send(const dm::CANPacket &packet) {
    canfd_frame frame{};
    frame.can_id = packet.send_can_id;
    frame.len = packet.data.size();
    frame.flags = CANFD_BRS;
    std::copy(packet.data.begin(), packet.data.end(), frame.data);
    const auto deadline = Clock::now() + 5ms;
    do {
      if (bus_->write_canfd_frame(frame)) return true;
      if (errno != ENOBUFS && errno != EAGAIN) return false;
      std::this_thread::sleep_for(150us);
    } while (Clock::now() < deadline);
    return false;
  }

  // Read existing motor parameters. No drive parameters or encoder zeros are written.
  double parameter(const dm::Motor &motor, dm::RID rid) {
    canfd_frame frame{};
    if (!send(Encoder::create_query_param_command(motor, static_cast<int>(rid))) ||
        !bus_->is_data_available(30000) || !bus_->read_canfd_frame(frame) ||
        frame.can_id != motor.get_recv_can_id() || frame.len != 8 ||
        frame.data[0] != motor.get_send_can_id() || frame.data[2] != 0x33 ||
        frame.data[3] != static_cast<int>(rid)) return NAN;
    auto data = Decoder::parse_motor_param_data({frame.data, frame.data + 8});
    return data.valid ? data.value : NAN;
  }

  bool sample() {
    for (const auto &motor : motors_) {
      if (!send(Encoder::create_refresh_command(motor))) return false;
    }
    // Our SocketCAN socket does not receive its own requests. A command frame
    // on this socket indicates another writer, even on a different ROS domain.
    for (size_t count = 0; bus_->is_data_available(1200); ++count) {
      canfd_frame frame{};
      if (count >= 64 || !bus_->read_canfd_frame(frame) || frame.len != 8 ||
          frame.can_id < 0x11 || frame.can_id > 0x18) return false;
      const size_t i = frame.can_id - 0x11;
      if ((frame.data[0] & 15u) != i + 1) return false;
      auto state = Decoder::parse_motor_state_data(motors_[i], {frame.data, frame.data + 8});
      if (!state.valid) return false;
      pos_[i] = i == axes ? state.position * (0.044 / -1.0472) : state.position;
      vel_[i] = state.velocity * velocity_scale_[i];
      effort_[i] = state.torque;
      status_[i] = frame.data[0] >> 4;
      received_[i] = Clock::now();
    }
    const auto now = Clock::now();
    return std::all_of(received_.begin(), received_.end(),
                       [now](auto stamp) { return now - stamp < 100ms; });
  }

  bool write_pose() {
    for (size_t i = 0; i < axes; ++i) {
      dm::MITParam hold{kp_[i], kd_[i], i == elbow ? target_ : start_[i], 0.0, 0.0};
      if (!send(Encoder::create_mit_control_command(motors_[i], hold))) return false;
    }
    return true;
  }

  bool release() {
    if (!powered_) return true;
    powered_ = false;
    for (int attempt = 0; attempt < 3; ++attempt) {
      for (size_t i = 0; i < axes; ++i) send(Encoder::create_disable_command(motors_[i]));
      std::this_thread::sleep_for(2ms);
      if (sample() && std::all_of(status_.begin(), status_.begin() + axes, [](int s) { return s == 0; })) {
        RCLCPP_INFO(log_, "Seven right motors confirm disabled; encoder acquisition continues.");
        return true;
      }
    }
    RCLCPP_ERROR(log_, "Could not confirm right motor disable. Check arm onsite.");
    return false;
  }

  Result fault(const char *reason) {
    RCLCPP_ERROR(log_, "%s: %s", info_.name.c_str(), reason);
    release();
    return Result::ERROR;
  }

public:
  ~RightArm() override { release(); }

  Callback on_init(const hardware_interface::HardwareInfo &info) override {
    if (SystemInterface::on_init(info) != Callback::SUCCESS) return Callback::ERROR;
    const auto side = info.hardware_parameters.at("side");
    if ((side != "left" && side != "right") || info.joints.size() != devices) return Callback::ERROR;
    right_ = side == "right";
    commissioned_tabletop_ = right_ && info.hardware_parameters.count("tabletop_profile_id");
    const std::array<dm::MotorType, devices> types{dm::MotorType::DM8009, dm::MotorType::DM8009,
        dm::MotorType::DM4340, dm::MotorType::DM4340, dm::MotorType::DM4310, dm::MotorType::DM4310,
        dm::MotorType::DM4310, dm::MotorType::DM4310};
    for (size_t i = 0; i < devices; ++i) {
      const std::string index = std::to_string(i + 1);
      const auto expected = "openarm_" + side + "_" + (i < axes ? "joint" + index : "finger_joint1");
      if (info.joints[i].name != expected) return Callback::ERROR;
      motors_.emplace_back(types[i], i + 1, 0x11 + i);
      if (right_ && i < axes) {
        kp_[i] = std::stod(info.hardware_parameters.at("kp" + index));
        kd_[i] = std::stod(info.hardware_parameters.at("kd" + index));
        if (commissioned_tabletop_) {
          const double q = std::stod(info.hardware_parameters.at("tabletop_start" + index));
          if (!std::isfinite(q) || q < (i == elbow ? -.02 : -.16) || q > (i == 0 ? .60 : .16))
            return Callback::ERROR;
          commissioned_start_[i] = q;
        }
      }
    }
    return Callback::SUCCESS;
  }

  Callback on_configure(const rclcpp_lifecycle::State &) override {
    const std::string interface = right_ ? "can0" : "can1";
    bus_ = std::make_unique<openarm::canbus::CANSocket>(interface, true);
    if (bus_->is_data_available(150000)) {
      RCLCPP_ERROR(log_, "%s is already carrying traffic", interface.c_str());
      return Callback::ERROR;
    }
    for (size_t i = 0; i < devices; ++i) {
      const auto &motor = motors_[i];
      const auto limits = dm::Motor::get_limit_param(motor.get_motor_type());
      const auto p = parameter(motor, dm::RID::PMAX);
      const auto v = parameter(motor, dm::RID::VMAX);
      const auto t = parameter(motor, dm::RID::TMAX);
      if (p != limits.pMax || t != limits.tMax || !std::isfinite(v) || v <= 0.0 ||
          (right_ && i < axes && parameter(motor, dm::RID::CTRL_MODE) != 1.0)) {
        RCLCPP_ERROR(log_, "%s motor %zu: no valid scaling/mode response", interface.c_str(), i + 1);
        return Callback::ERROR;
      }
      velocity_scale_[i] = v / limits.vMax;
    }
    if (!sample()) return Callback::ERROR;
    RCLCPP_INFO(log_, "%s: eight encoder replies; sensor acquisition ready", interface.c_str());
    return Callback::SUCCESS;
  }

  std::vector<hardware_interface::StateInterface> export_state_interfaces() override {
    std::vector<hardware_interface::StateInterface> states;
    for (size_t i = 0; i < devices; ++i) {
      const auto &name = info_.joints[i].name;
      states.emplace_back(name, "position", &pos_[i]);
      if (i < axes) {
        states.emplace_back(name, "velocity", &vel_[i]);
        states.emplace_back(name, "effort", &effort_[i]);
      }
    }
    return states;
  }

  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override {
    std::vector<hardware_interface::CommandInterface> commands;
    if (right_) commands.emplace_back(info_.joints[elbow].name, "position", &command_);
    return commands;
  }

  Result perform_command_mode_switch(const std::vector<std::string> &start,
                                     const std::vector<std::string> &stop) override {
    if (!right_) return Result::OK;
    if (std::find(stop.begin(), stop.end(), command_key) != stop.end() && !release()) return Result::ERROR;
    if (std::find(start.begin(), start.end(), command_key) == start.end()) return Result::OK;
    if (!sample()) return fault("missing encoder feedback or competing bus traffic");
    for (size_t i = 0; i < axes; ++i) {
      const bool in_pose = commissioned_tabletop_
        ? (std::abs(pos_[i] - commissioned_start_[i]) <= .04 &&
           pos_[i] >= (i == elbow ? -.02 : -.16) && pos_[i] <= (i == 0 ? .60 : .16))
        : std::abs(pos_[i]) <= .16;
      if (status_[i] != 0 || std::abs(vel_[i]) > .1 || !in_pose)
        return fault("curl requires stationary encoders inside its configured starting envelope");
      start_[i] = pos_[i];
    }
    target_ = command_ = start_[elbow];
    powered_ = true;
    if (!write_pose()) return fault("could not preload measured hold positions");
    for (size_t i = 0; i < axes; ++i) {
      if (!send(Encoder::create_enable_command(motors_[i]))) return fault("motor enable failed");
    }
    if (!write_pose() || !sample()) return fault("no feedback after measured hold");
    for (size_t i = 0; i < axes; ++i) {
      if (status_[i] != 1) return fault("right motor did not acknowledge enable");
    }
    RCLCPP_INFO(log_, "Holding measured right arm posture; J4 accepts bounded trajectories.");
    return Result::OK;
  }

  Result read(const rclcpp::Time &, const rclcpp::Duration &) override {
    if (!sample()) return fault("missing or unexpected CAN feedback");
    if (powered_) {
      for (size_t i = 0; i < axes; ++i) {
        if (status_[i] != 1 || std::abs(pos_[i] - (i == elbow ? target_ : start_[i])) > 0.10)
          return fault("right motor status or tracking error outside test limits");
      }
    }
    return Result::OK;
  }

  Result write(const rclcpp::Time &, const rclcpp::Duration &period) override {
    if (!powered_) return Result::OK;
    if (!std::isfinite(command_) || command_ < std::max(-0.02, start_[elbow] - 0.025) ||
        command_ > start_[elbow] + 0.46 || command_ > 2.443461)
      return fault("elbow command outside curl envelope");
    const double step = 0.14 * std::clamp(period.seconds(), 0.0, 0.04);
    target_ = std::clamp(command_, target_ - step, target_ + step);
    return write_pose() ? Result::OK : fault("CAN output failed");
  }

  Callback on_deactivate(const rclcpp_lifecycle::State &) override { release(); return Callback::SUCCESS; }
  Callback on_shutdown(const rclcpp_lifecycle::State &) override { release(); return Callback::SUCCESS; }
  Callback on_error(const rclcpp_lifecycle::State &) override { release(); return Callback::SUCCESS; }
};
}  // namespace openarm_right_curl

PLUGINLIB_EXPORT_CLASS(openarm_right_curl::RightArm, hardware_interface::SystemInterface)
