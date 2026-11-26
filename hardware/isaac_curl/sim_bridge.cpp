#include <cmath>
#include <chrono>
#include <thread>
#include <map>
#include <memory>
#include <vector>
#include <hardware_interface/system_interface.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64.hpp>

namespace laiting_isaac {
using State = sensor_msgs::msg::JointState;
using Result = hardware_interface::return_type;
using Callback = hardware_interface::CallbackReturn;
class Bridge final : public hardware_interface::SystemInterface {
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<State>::SharedPtr rx_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr tx_;
  std::map<std::string, size_t> index_;
  std::vector<double> q_, v_;
  size_t elbow_ = 0;
  double cmd_ = 0.;
  double upper_ = 0.;
  int64_t received_ = 0;
  bool active_ = false;
 public:
  Callback on_init(const hardware_interface::HardwareInfo &info) override {
    if (SystemInterface::on_init(info) != Callback::SUCCESS) return Callback::ERROR;
    upper_ = std::stod(info.hardware_parameters.at("upper_elbow_rad"));
    if (!std::isfinite(upper_) || upper_ <= 0 || upper_ > 1.0) return Callback::ERROR;
    q_.assign(info.joints.size(), NAN);
    v_=q_;
    for(size_t i=0;i<q_.size();++i) index_[info.joints[i].name]=i;
    elbow_=index_.at("openarm_right_joint4");
    node_=std::make_shared<rclcpp::Node>("isaac_measured_joint_bridge",
      rclcpp::NodeOptions().use_global_arguments(false));
    rx_=node_->create_subscription<State>("/laiting/isaac/encoders",rclcpp::SensorDataQoS(),
      [this](State::ConstSharedPtr msg){
        if (msg->name.size()!=msg->position.size() || msg->velocity.size()!=msg->position.size()) return;
        if (std::abs((node_->now()-rclcpp::Time(msg->header.stamp)).seconds())>.30) return;
        std::vector<double> pos(q_.size(),NAN), vel(q_.size(),NAN);
        for(size_t j=0;j<msg->name.size();++j){
          auto it=index_.find(msg->name[j]);
          if(it!=index_.end()){ pos[it->second]=msg->position[j]; vel[it->second]=msg->velocity[j]; }
        }
        for(size_t i=0;i<pos.size();++i) if(!std::isfinite(pos[i]) || !std::isfinite(vel[i])) return;
        std::copy(pos.begin(),pos.end(),q_.begin());
        std::copy(vel.begin(),vel.end(),v_.begin());
        received_=node_->now().nanoseconds();
      });
    tx_=node_->create_publisher<std_msgs::msg::Float64>("/laiting/isaac/elbow_target",1);
    return Callback::SUCCESS;
  }
  Callback on_configure(const rclcpp_lifecycle::State &) override {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(12);
    while(!received_ && std::chrono::steady_clock::now()<deadline){
      rclcpp::spin_some(node_);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return received_ ? Callback::SUCCESS : Callback::ERROR;
  }
  std::vector<hardware_interface::StateInterface> export_state_interfaces() override {
    std::vector<hardware_interface::StateInterface> out;
    for(const auto &j:info_.joints) {
      auto i=index_[j.name];
      out.emplace_back(j.name,"position",&q_[i]); out.emplace_back(j.name,"velocity",&v_[i]);
    }
    return out;
  }
  std::vector<hardware_interface::CommandInterface> export_command_interfaces() override {
    std::vector<hardware_interface::CommandInterface> out;
    out.emplace_back("openarm_right_joint4","position",&cmd_);
    return out;
  }
  Result perform_command_mode_switch(const std::vector<std::string>& start,const std::vector<std::string>& stop) override {
    if(!stop.empty()) active_=false;
    if(!start.empty()){
      if(!std::isfinite(q_[elbow_]) || node_->now().nanoseconds()-received_>300000000) return Result::ERROR;
      cmd_=q_[elbow_]; active_=true;
    }
    return Result::OK;
  }
  Result read(const rclcpp::Time&,const rclcpp::Duration&) override {
    rclcpp::spin_some(node_);
    if (received_ && node_->now().nanoseconds()-received_>350000000) {
      active_=false;
      return Result::ERROR;
    }
    return Result::OK;
  }
  Result write(const rclcpp::Time&,const rclcpp::Duration&) override {
    if(active_ && std::isfinite(cmd_) && std::isfinite(q_[elbow_]) && cmd_>=-.025 && cmd_<=upper_){
      std_msgs::msg::Float64 m; m.data=cmd_; tx_->publish(m);
    }
    return Result::OK;
  }
};
}
PLUGINLIB_EXPORT_CLASS(laiting_isaac::Bridge,hardware_interface::SystemInterface)
