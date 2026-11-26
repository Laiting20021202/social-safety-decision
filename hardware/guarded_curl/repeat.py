"""Plan a right-elbow curl in MoveIt. Stream only short trajectory segments while permission is fresh."""
import argparse
import csv
import json
import os
from pathlib import Path
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatusArray, GoalStatus
from builtin_interfaces.msg import Duration
from controller_manager_msgs.srv import ListControllers, SwitchController
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
from moveit_msgs.srv import GetMotionPlan
from rclpy.duration import Duration as ROSDuration
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from frames import SENSOR, fresh, stamp_ns
from registration import curl_span, physical_profile_matches, tabletop_start

CTRL='right_joint_trajectory_controller'
JOINT='openarm_right_joint4'
TOPIC=f'/{CTRL}/joint_trajectory'
STEP=.12
SPEED=.06


class NoClearPath(RuntimeError):
    """The latest planning scene has no approved path to this endpoint."""


class Repeat(Node):
    def __init__(self, execute, output):
        super().__init__('laiting_guarded_curl')
        self.js=None
        self.q=None
        self.vel=0.
        self.guard={}
        self.guard_time=0.
        self.denials=0
        self.active_action=False
        self.description=''
        self.create_subscription(String,'/robot_description',lambda msg:setattr(self,'description',msg.data),
                                 QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.plan_srv=self.create_client(GetMotionPlan,'/plan_kinematic_path')
        self.inventory=self.create_client(ListControllers,'/controller_manager/list_controllers')
        self.switch=self.create_client(SwitchController,'/controller_manager/switch_controller')
        self.create_subscription(JointState,'/joint_states',self.joints,SENSOR)
        self.create_subscription(String,'/laiting/safety/state',self.safety,10)
        self.create_subscription(GoalStatusArray, f'/{CTRL}/follow_joint_trajectory/_action/status',
                                 self.action_status,10)
        self.tx=None
        if execute:
            qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT,
                           lifespan=ROSDuration(seconds=.20))
            self.tx=self.create_publisher(JointTrajectory,TOPIC,qos)
        self.output=output
        self.writer=csv.writer(output)
        self.writer.writerow(['wall_ns','event','target_rad','encoder_rad','velocity_rad_s','reason'])
        self.owns=False
        self.rest=None
        self.span=curl_span()

    def log(self,event,target=''):
        reason=self.guard.get('reason','no_guard')
        if reason=='clear' and not self.clear(): reason='guard_heartbeat_timeout'
        if self.js is not None and not self.feedback(): reason='encoder_timeout'
        self.writer.writerow([time.time_ns(),event,target,self.q,self.vel,reason])
        self.output.flush()
        if event not in ('step','paused'): print(event,target,'encoder',self.q,reason,flush=True)

    def joints(self,m):
        if JOINT not in m.name or not fresh(stamp_ns(m.header),.30): return
        if self.js is not None and stamp_ns(m.header)<=stamp_ns(self.js.header): return
        i=m.name.index(JOINT)
        if i>=len(m.velocity) or i>=len(m.position) or not np.isfinite([m.position[i],m.velocity[i]]).all():
            return
        self.q=m.position[i]; self.vel=m.velocity[i]; self.js=m

    def safety(self,m):
        try: g=json.loads(m.data)
        except (ValueError,TypeError): g={}
        self.guard=g
        self.guard_time=time.monotonic()
        if not self.clear(): self.denials+=1

    def action_status(self,m):
        self.active_action=any(s.status in (GoalStatus.STATUS_EXECUTING,GoalStatus.STATUS_ACCEPTED,
                                           GoalStatus.STATUS_CANCELING) for s in m.status_list)

    def clear(self):
        return (self.guard.get('permit') is True and self.guard.get('registered') is True
                and self.guard.get('reason')=='clear' and fresh(self.guard.get('stamp_ns',0),.36)
                and time.monotonic()-self.guard_time<.32)

    def feedback(self):
        return self.js is not None and fresh(stamp_ns(self.js.header),.25)

    def spin(self,secs):
        end=time.monotonic()+secs
        while rclpy.ok() and time.monotonic()<end:
            rclpy.spin_once(self,timeout_sec=min(.02,max(0.,end-time.monotonic())))

    def call(self,client,req,timeout):
        if not client.wait_for_service(timeout_sec=3):
            raise RuntimeError(f'{client.srv_name} unavailable')
        future=client.call_async(req)
        end=time.monotonic()+timeout
        while rclpy.ok() and not future.done() and time.monotonic()<end:
            rclpy.spin_once(self,timeout_sec=.02)
        if not future.done(): raise RuntimeError(f'{client.srv_name} timed out')
        return future.result()

    def plan(self,target):
        if not self.feedback(): raise RuntimeError('stale encoders before MoveIt request')
        start=self.q
        request=GetMotionPlan.Request()
        p=request.motion_plan_request
        p.group_name='right_curl'; p.pipeline_id='ompl'; p.planner_id='RRTConnect'
        p.start_state.joint_state=self.js; p.start_state.is_diff=True
        p.allowed_planning_time=2.; p.num_planning_attempts=3
        p.max_velocity_scaling_factor=.75; p.max_acceleration_scaling_factor=.5
        con=JointConstraint(joint_name=JOINT,position=target,tolerance_above=.001,
                            tolerance_below=.001,weight=1.)
        p.goal_constraints=[Constraints(joint_constraints=[con])]
        current=self.call(self.plan_srv,request,5).motion_plan_response
        if current.error_code.val in (MoveItErrorCodes.FAILURE, MoveItErrorCodes.PLANNING_FAILED,
                MoveItErrorCodes.START_STATE_IN_COLLISION, MoveItErrorCodes.GOAL_IN_COLLISION,
                MoveItErrorCodes.TIMED_OUT, MoveItErrorCodes.MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE):
            raise NoClearPath(f'MoveIt has no approved path ({current.error_code.val})')
        if current.error_code.val!=MoveItErrorCodes.SUCCESS:
            raise RuntimeError(f'MoveIt planning error {current.error_code.val}')
        traj=current.trajectory.joint_trajectory
        if list(traj.joint_names)!=[JOINT] or not traj.points:
            raise RuntimeError('MoveIt returned joints outside the curl group')
        path=np.array([pt.positions[0] for pt in traj.points])
        # A safe prefix can be streamed only if the entire approved 1-DOF path is monotonic.
        sign=1 if target>=start else -1
        if (not np.isfinite(path).all() or abs(path[0]-start)>.025 or abs(path[-1]-target)>.004
                or np.any(sign*np.diff(path)<-.0001)):
            raise RuntimeError('unexpected MoveIt path shape or endpoints')
        self.log('plan',round(float(path[-1]),5))
        return path

    def enable(self,value):
        req=SwitchController.Request()
        req.strictness=SwitchController.Request.STRICT
        req.timeout=Duration(sec=2)
        if value: req.activate_controllers=[CTRL]
        else: req.deactivate_controllers=[CTRL]
        if not self.call(self.switch,req,4).ok: raise RuntimeError('NUC rejected controller switch')
        self.owns=value
        self.log('enabled' if value else 'released')

    def send(self,q):
        m=JointTrajectory()
        m.header.stamp=self.get_clock().now().to_msg()
        m.joint_names=[JOINT]
        m.points=[JointTrajectoryPoint(positions=[float(q)],time_from_start=Duration(nanosec=int(STEP*1e9)))]
        self.tx.publish(m)

    def hold(self):
        if self.owns and self.feedback() and not self.active_action:
            # Latch once. Repeatedly sampling gravity sag as a new target would lower the arm.
            q=float(np.clip(self.q,self.rest,self.rest+self.span+.025))
            self.send(q)
            self.log('hold',round(q,5))

    def leg(self,goal,deadline):
        path=None
        command=None
        done_since=None
        held=False
        accepted_denials=self.denials
        progress_q=None
        progress_at=0.
        retry_at=0.
        last=time.monotonic()
        while rclpy.ok() and time.monotonic()<deadline:
            self.spin(.048)
            now=time.monotonic()
            dt=min(now-last,.07); last=now
            if self.active_action or self.count_publishers(TOPIC)!=1:
                raise RuntimeError('another trajectory writer is present')
            if not self.clear() or not self.feedback() or accepted_denials!=self.denials:
                if not held: self.hold()
                else: self.log('paused')
                held=True; path=None; done_since=None; accepted_denials=self.denials
                continue
            if path is None:
                if now < retry_at:
                    continue
                try:
                    path=self.plan(goal)
                except NoClearPath:
                    if not held:
                        self.hold()
                    self.log('awaiting_path')
                    retry_at=time.monotonic()+.75
                    held=True
                    continue
                if accepted_denials!=self.denials or not self.clear():
                    path=None
                    continue
                command=float(path[0]); held=False
                progress_q,progress_at=self.q,now
                self.log('stream',round(goal,5))
            if abs(command-self.q)>.055:
                raise RuntimeError('right elbow no longer follows short commands')
            # In the one dimensional joint group every value between these bounds is on the checked path.
            target=float(path[-1])
            next_command=float(np.clip(target, command-SPEED*dt, command+SPEED*dt))
            # With a closed-loop trajectory controller each short segment starts at the
            # latest measured state. Never build up a long lead over its actual position.
            command=float(np.clip(next_command,self.q-.035,self.q+.035))
            if abs(self.q-progress_q)>.004 or abs(self.q-goal)<.033:
                progress_q,progress_at=self.q,now
            elif now-progress_at>2.5:
                raise RuntimeError('no measured elbow progress')
            self.send(command)
            self.log('step',round(command,6))
            if abs(command-target)<.001 and abs(self.q-goal)<.033 and abs(self.vel)<.045:
                if done_since is None: done_since=now
                if now-done_since>.35:
                    self.log('reached',round(goal,5)); return
            else:
                done_since=None
        raise RuntimeError('run deadline reached')

    def run(self,cycles,timeout,execute):
        self.spin(1.4)
        if not self.feedback(): raise RuntimeError('no live /joint_states')
        expected=tabletop_start()
        self.rest=max(0.,self.q) if expected is None else expected[3]
        peak=self.rest+self.span
        if not execute:
            self.plan(peak); return
        if not self.clear(): raise RuntimeError('motion inhibited: '+self.guard.get('reason','no fresh guard'))
        measured=dict(zip(self.js.name,self.js.position))
        right=np.array([measured.get(f'openarm_right_joint{i}',np.nan) for i in range(1,8)])
        if expected is None: expected=np.zeros(7)
        if not np.isfinite(right).all() or np.max(np.abs(right-expected))>.16 or self.q<-.02 or abs(self.vel)>.06:
            raise RuntimeError('right arm is outside the commissioned starting posture')
        elevated_site = os.environ.get('LAITING_CAMERA_PROFILE')!='isaac_fixed_openarm_v1' and expected[0]>.16
        if elevated_site:
            if (not physical_profile_matches(self.description) or np.max(np.abs(right-expected))>.04
                    or not -.16<=right[0]<=.60 or np.max(np.abs(right[1:]))>.16):
                raise RuntimeError('NUC and station must match the surveyed tabletop start')
        elif os.environ.get('LAITING_CAMERA_PROFILE')!='isaac_fixed_openarm_v1' and np.max(np.abs(right))>.16:
            raise RuntimeError('measured start exceeds the physical CAN driver entry envelope')
        ctrls=self.call(self.inventory,ListControllers.Request(),4).controller
        if dict((c.name,c.state) for c in ctrls).get(CTRL)!='inactive':
            raise RuntimeError('controller already owned; existing hold preserved')
        if self.active_action or self.count_publishers(TOPIC)!=1:
            raise RuntimeError('another writer has control')
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            if self.clear() and self.feedback():
                try:
                    self.plan(peak)
                    if self.clear():
                        break
                except NoClearPath:
                    self.log('awaiting_path')
            self.spin(.75)
        else:
            raise RuntimeError('no approved initial path before deadline')
        if self.active_action or self.count_publishers(TOPIC)!=1:
            raise RuntimeError('another writer has control')
        self.enable(True)
        try:
            for cycle in range(cycles):
                self.leg(peak,deadline)
                self.leg(self.rest,deadline)
                self.log('cycle',cycle+1)
        finally:
            if self.owns:
                if not elevated_site and self.feedback() and abs(self.q-self.rest)<.035 and abs(self.vel)<.05 and not self.active_action:
                    self.enable(False)
                else:
                    self.hold()
                    self.log('support_retained')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true',help='Explicitly request the physical command channel')
    p.add_argument('--cycles',type=int,default=2)
    p.add_argument('--timeout',type=float,default=120.)
    p.add_argument('--log',type=Path)
    args=p.parse_args()
    if args.cycles<1 or not 1<=args.timeout<=3600: p.error('positive cycles and a 1–3600 second timeout required')
    path=args.log or Path(f'/results/curl-{"execution" if args.execute else "plan"}-{time.time_ns()}.csv')
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    with path.open('x') as output:
        node=Repeat(args.execute,output)
        try: node.run(args.cycles,args.timeout,args.execute)
        except (KeyboardInterrupt,RuntimeError) as exc:
            node.log('stopped')
            raise SystemExit(str(exc) or 'operator interrupt')
        finally:
            node.destroy_node()
            rclpy.try_shutdown()


if __name__=='__main__':
    main()
