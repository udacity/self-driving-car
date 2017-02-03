#!/usr/bin/env python
import multiprocessing
import time

import numpy as np


CLOCK_MONOTONIC_RAW = 4  # see <linux/time.h>
CLOCK_BOOTTIME = 7


# TODO: Look at the OpenPilot repo if more accurate timing is neeed.
def clock_gettime(clk_id):
    return time.time()


def sec_since_boot():
    return clock_gettime(CLOCK_BOOTTIME)


class Ratekeeper(object):

    def __init__(self, rate, print_delay_threshold=0.):
        """Rate in Hz for ratekeeping. print_delay_threshold must be nonnegative."""
        self._interval = 1. / rate
        self._next_frame_time = sec_since_boot() + self._interval
        self._print_delay_threshold = print_delay_threshold
        self._frame = 0
        self._remaining = 0
        self._process_name = multiprocessing.current_process().name

    @property
    def frame(self):
        return self._frame

    # Maintain loop rate by calling this at the end of each loop
    def keep_time(self):
        self.monitor_time()
        if self._remaining > 0:
            time.sleep(self._remaining)

    # this only monitor the cumulative lag, but does not enforce a rate
    def monitor_time(self):
        remaining = self._next_frame_time - sec_since_boot()
        self._next_frame_time += self._interval
        self._frame += 1
        self._remaining = remaining


def car_plant(pos, speed, grade, gas, brake):
    # vehicle parameters
    mass = 1700
    aero_cd = 0.3
    force_peak = mass * 3.
    force_brake_peak = -mass * 10.  # 1g
    power_peak = 100000   # 100kW
    speed_base = power_peak / force_peak
    rolling_res = 0.01
    g = 9.81
    air_density = 1.225
    gas_to_peak_linear_slope = 3.33
    brake_to_peak_linear_slope = 0.2
    creep_accel_v = [1., 0.]
    creep_accel_bp = [0., 1.5]

    # *** longitudinal model ***
    # find speed where peak torque meets peak power
    force_brake = brake * force_brake_peak * brake_to_peak_linear_slope
    if speed < speed_base:  # torque control
        force_gas = gas * force_peak * gas_to_peak_linear_slope
    else:  # power control
        force_gas = gas * power_peak / speed * gas_to_peak_linear_slope

    force_grade = - grade * mass  # positive grade means uphill

    creep_accel = np.interp(speed, creep_accel_bp, creep_accel_v)
    force_creep = creep_accel * mass

    force_resistance = -(rolling_res * mass * g + 0.5 *
                         speed**2 * aero_cd * air_density)
    force = force_gas + force_brake + force_resistance + force_grade + force_creep
    acceleration = force / mass

    # TODO: lateral model
    return speed, acceleration


class Plant(object):
    messaging_initialized = False

    def __init__(self, lead_relevancy=False, rate=100,
                 speed=0.0, distance_lead=2.0,
                 verbosity=0):
        self.rate = rate
        self.civic = False
        self.brake_only = False
        self.verbosity = verbosity

        self.angle_steer = 0.
        self.gear_choice = 0
        self.speed, self.speed_prev = 0., 0.

        self.esp_disabled = 0
        self.main_on = 1
        self.user_gas = 0
        self.computer_brake, self.user_brake = 0, 0
        self.brake_pressed = 0
        self.distance, self.distance_prev = 0., 0.
        self.speed, self.speed_prev = speed, speed
        self.steer_error, self.brake_error, self.steer_not_allowed = 0, 0, 0
        self.gear_shifter = 4   # D gear
        self.pedal_gas = 0
        self.cruise_setting = 0

        self.seatbelt, self.door_all_closed = True, True
        # v_cruise is reported from can, not the one used for controls
        self.steer_torque, self.v_cruise, self.acc_status = 0, 0, 0

        self.lead_relevancy = lead_relevancy

        # lead car
        self.distance_lead, self.distance_lead_prev = distance_lead, distance_lead

        self.rk = Ratekeeper(rate, print_delay_threshold=100)
        self.ts = 1. / rate

    def speed_sensor(self, speed):
        if speed < 0.3:
            return 0
        else:
            return speed

    def current_time(self):
        return float(self.rk.frame) / self.rate

    def step(self, brake=0, gas=0, steer_torque=0, v_lead=0.0, cruise_buttons=None, grade=0.0):
        # dbc_f, sgs, ivs, msgs, cks_msgs, frqs = initialize_can_struct(self.civic, self.brake_only)

        distance_lead = self.distance_lead_prev + v_lead * self.ts

        # ******** run the car ********
        speed, acceleration = car_plant(
            self.distance_prev, self.speed_prev, grade, gas, brake)
        distance = self.distance_prev + speed * self.ts
        speed = self.speed_prev + self.ts * acceleration
        if speed <= 0:
            speed = 0
            acceleration = 0

        # ******** lateral ********
        self.angle_steer -= steer_torque

        # *** radar model ***
        if self.lead_relevancy:
            d_rel = np.maximum(0., self.distance_lead - distance)
            v_rel = v_lead - speed
        else:
            d_rel = 200.
            v_rel = 0.

        # print at 5hz
        if (self.rk.frame % (self.rate / 5)) == 0:
            msg_tmpl = ("%6.2f m  %6.2f m/s  %6.2f m/s2   %.2f ang "
                        "  gas: %.2f  brake: %.2f  steer: %5.2f "
                        "  lead_rel: %6.2f m  %6.2f m/s")
            msg = msg_tmpl % (distance, speed, acceleration, self.angle_steer,
                              gas, brake, steer_torque, d_rel, v_rel)

            if self.verbosity > 2:
                print(msg)

        # ******** publish the car ********
        vls = [self.speed_sensor(speed), self.speed_sensor(speed),
               self.speed_sensor(speed), self.speed_sensor(speed),
               self.angle_steer, 0, self.gear_choice, speed != 0,
               0, 0, 0, 0,
               self.v_cruise, not self.seatbelt, self.seatbelt, self.brake_pressed,
               self.user_gas, cruise_buttons, self.esp_disabled, 0,
               self.user_brake, self.steer_error, self.speed_sensor(
                   speed), self.brake_error,
               self.brake_error, self.gear_shifter, self.main_on, self.acc_status,
               self.pedal_gas, self.cruise_setting,
               # left_blinker, right_blinker, counter
               0, 0, 0,
               # interceptor_gas
               0, 0]

        # TODO: Use vls for something
        assert vls is not None

        # ******** update prevs ********
        self.speed_prev = speed
        self.distance_prev = distance
        self.distance_lead_prev = distance_lead

        car_in_front = distance_lead - \
            distance if self.lead_relevancy else 200.

        self.rk.keep_time()
        return (speed, acceleration, car_in_front, steer_torque)
